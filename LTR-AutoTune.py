import pickle
import re

import lightgbm as lgb
import numpy as np
import pandas as pd
from flaml import AutoML
from sklearn.model_selection import train_test_split

TREC_QREL_COLUMNS = ['qid', 'iteration', 'docNo', 'rel']
TREC_RES_COLUMNS = ['qid', 'iteration', 'docNo', 'rank', 'docScore', 'method']


def resdf_to_trec_format(df):
    _df = df[['qid', 'docNo', 'docScore']]
    _df = _df.assign(iteration='Q0', method='LTR', rank=_df.groupby('qid')['docScore'].rank(ascending=False))
    _df['rank'] = _df['rank'].astype(int)
    return _df[TREC_RES_COLUMNS]


def rreplace(string, old, new, count=1):
    lis = string.rsplit(old, count)
    return new.join(lis)


def filter_alpha_numeric(s):
    return ' '.join(re.sub(r'[^A-Za-z\d ]+', '', s).lower().split())


def calc_ndcg(qrels_file, results_file, k, logger=None, base=2, gdeval=True, non_judged_score=0):
    """
    Setting gdeval will yield identical results to the official evaluation script that was published for TREC.
    Note that the calculation in that script differs from (probably) any published research version.
    :param qrels_file:
    :param results_file:
    :param k:
    :param logger:
    :param base:
    :param gdeval:
    :return:
    """
    if isinstance(qrels_file, str):
        qrels_df = pd.read_csv(qrels_file, delim_whitespace=True, names=TREC_QREL_COLUMNS)
    elif isinstance(qrels_file, pd.DataFrame):
        qrels_df = qrels_file
    else:
        raise TypeError
    # Reading and sorting the qrels, to later speed-up indexing and locating
    qrels_df = qrels_df.sort_values(['qid', 'rel', 'docNo'], ascending=[True, False, True]).set_index(['qid', 'docNo'])
    qrels_df['rel'].clip(lower=0, inplace=True)

    # Store beginning and end indices for each query - used for speed up
    qid_res_len = qrels_df.groupby('qid').apply(len)
    qid_end_loc = qid_res_len.cumsum()
    qid_start_loc = qid_end_loc - qid_res_len
    qrels_df = qrels_df.droplevel(0)

    nonjudged = 0

    if isinstance(results_file, str):
        results_df = pd.read_csv(results_file, delim_whitespace=True, names=TREC_RES_COLUMNS)
    elif isinstance(results_file, pd.DataFrame):
        results_df = results_file
    else:
        raise TypeError
    if gdeval:
        results_df = results_df.sort_values(['qid', 'docScore', 'docNo'], ascending=[True, False, False]).groupby(
            'qid').head(k)
        discount = np.log(np.arange(1, k + 1) + 1)
    else:
        results_df = results_df.sort_values(['qid', 'rank']).groupby('qid').head(k)
        discount = np.concatenate((np.ones(base), np.log(np.arange(base, k) + 1) / np.log(base)))
    result = {}
    for qid, _df in results_df.groupby('qid'):
        docs = _df['docNo'].to_numpy()
        try:
            _qrels_df = qrels_df.iloc[qid_start_loc.loc[qid]: qid_end_loc.loc[qid]]
        except KeyError as err:
            if logger is None:
                print(f'query id {err} doesn\'t exist in the qrels file, skipping it')
            else:
                logger.warning(f'query id {err} doesn\'t exist in the qrels file, skipping it')
            continue
        nonjudged += _qrels_df.reindex(docs)['rel'].isna().sum()
        if gdeval:
            dcg = 2 ** _qrels_df.reindex(docs)['rel'].fillna(non_judged_score).to_numpy() - 1
            idcg = ((2 ** _qrels_df['rel'].head(k) - 1) / discount[:len(_qrels_df)]).sum()
        else:
            dcg = _qrels_df.reindex(docs)['rel'].fillna(non_judged_score).to_numpy()
            idcg = (_qrels_df['rel'].head(k) / discount[:len(_qrels_df)]).sum()
        result[qid] = (dcg / discount[:len(dcg)]).sum() / idcg
    res_df = pd.DataFrame.from_dict(result, orient='index', columns=[f'nDCG@{k}'])
    #     res_df.to_csv(rreplace(results_file, 'run', f'ndcg@{k}', 1), sep='\t', float_format='%.6f', header=False)
    #     print(res_df.to_string(float_format='%.5f'))
    #     print(f'Mean: {res_df.mean()[0]:.5f}')
    print(f"Total queries {results_df['qid'].nunique()}")
    print(f"Average non judged docs per query in top {k} {nonjudged / results_df['qid'].nunique():.3f}")
    return res_df


def df_to_dataset(df):
    return lgb.Dataset(df[df.columns[4:]], label=df['label'].clip(lower=0), group=df.groupby('qid')['docNo'].count())


def train_eval(params):
    model = lgb.train(params=params, train_set=df_to_dataset(train_df), valid_sets=[df_to_dataset(valid_df)])
    res_df = valid_df.assign(docScore=model.predict(valid_df[valid_df.columns[4:]]))
    res_df = res_df.sort_values(['qid', 'docScore'], ascending=[True, False])
    res_df = resdf_to_trec_format(res_df)
    result = calc_ndcg(qrels_df, res_df, 10, gdeval=True)
    return model, res_df, result


def eval_trained_model(model, eval_df):
    res_df = eval_df.assign(docScore=model.predict(eval_df[eval_df.columns[4:]]))
    res_df = res_df.sort_values(['qid', 'docScore'], ascending=[True, False])
    res_df = resdf_to_trec_format(res_df)
    result = calc_ndcg(qrels_df, res_df, 10, gdeval=True)
    return model, res_df, result


def main():
    # Load Model
    #     try:
    #         with open("automl_script.pkl", "rb") as f:
    #             automl = pickle.load(f)
    #     except FileNot

    automl = AutoML()

    settings = {
        "time_budget": 1 * 24 * 60 * 60,  # total running time in seconds
        "metric": 'ndcg',  # primary metrics for regression can be chosen from: ['mae','mse','r2']
        "estimator_list": ['xgboost'],  # list of ML learners
        "task": 'rank',  # task type  
        "log_file_name": 'automl_script_xgboost.log',  # flaml log file
        "seed": 169,  # random seed
        'split_type': 'group',
    }

    start_params = {
        'lgbm':
            {'boosting_type': 'dart',
             'n_estimators': 1500,
             'num_leaves': 10,
             'min_child_samples': 70,
             'learning_rate': 0.05,
             'log_max_bin': 7,
             'colsample_bytree': 0.9,
             'reg_alpha': 0.0,
             'subsample': 0.5,
             'subsample_freq': 10,
             'eval_at': 10,
             'reg_lambda': 0.0},
        'xgboost':
            {'n_estimators': 70,
             'max_leaves': 183,
             'min_child_weight': 128.0,
             'learning_rate': 0.139,
             'subsample': 0.842,
             'colsample_bylevel': 0.288,
             'colsample_bytree': 0.953,
             'reg_alpha': 0.0298,
             'reg_lambda': 0.001}
    }

    # Load Model
    with open("automl_script_xgboost.pkl", "rb") as f:
        automl1 = pickle.load(f)

    for k, v in automl1.best_config_per_estimator.items():
        if isinstance(v, dict):
            for _k, _v in v.items():
                start_params[k][_k] = _v
        else:
            start_params[k] = v

    automl.fit(X_train=train_df[train_df.columns[4:]].to_numpy(), y_train=train_df['label'].clip(lower=0).to_numpy(),
               groups=train_df.groupby('qid')['docNo'].count().to_numpy(),
               X_val=valid_df[valid_df.columns[4:]].to_numpy(),
               y_val=valid_df['label'].clip(lower=0), **settings, starting_points=start_params)

    # Save the model
    with open("automl_script_xgboost.pkl", "wb") as f:
        pickle.dump(automl, f, pickle.HIGHEST_PROTOCOL)

    print(automl.best_estimator)
    print()
    print(automl.best_config)
    print()
    print(automl.best_config_per_estimator)


if __name__ == '__main__':
    full_features_df = pd.read_csv('uqv100_all-unique_features.csv', header=None)
    full_features_df.columns = ['label', 'qid', 'docNo'] + list(range(len(full_features_df.columns) - 3))
    full_features_df = full_features_df.assign(topic=full_features_df['qid'].str.split('-').str[0])[
        ['label', 'topic', 'qid', 'docNo'] + full_features_df.columns[3:].tolist()]

    qrels_df = pd.read_csv('uqv100-allQueries-posit.qrels', delim_whitespace=True, names=TREC_QREL_COLUMNS)
    qdf = pd.read_csv('QueriesSurvey.csv', sep=',', header=None, names=['qid', 'query'])
    test_topics = qdf['qid'].str.split('-').str[0].unique()

    test_df = full_features_df.loc[full_features_df['topic'].isin(test_topics)]
    full_train_df = full_features_df.loc[~full_features_df['topic'].isin(test_topics)]

    train_topics, validation_topics = train_test_split(full_train_df['topic'].unique(), train_size=0.8,
                                                       random_state=169)
    train_df = full_train_df.loc[full_train_df['topic'].isin(train_topics)]
    valid_df = full_train_df.loc[full_train_df['topic'].isin(validation_topics)]

    print(f'Test set shape: {test_df.shape}')
    print(f'Validation set shape: {valid_df.shape}')
    print(f'Train set shape: {train_df.shape}')

    main()
