import re

import numpy as np
import pandas as pd

TREC_RES_COLUMNS = ['qid', 'iteration', 'docNo', 'rank', 'docScore', 'method']
TREC_QREL_COLUMNS = ['qid', 'iteration', 'docNo', 'rel']


def filter_alpha_numeric(s):
    return ' '.join(re.sub(r'[^A-Za-z0-9 ]+', '', s).lower().split())


def calc_ndcg(qrels_file, results_file, k, base=2, gdeval=True):
    """
    Setting gdeval will produce identical results to the official evaluation script that was published for TREC.
    Note that the calculation in that script differs from (probably) any published research version.
    :param qrels_file: a path to a TREC style qrels file
    :param results_file: a path to a TREC style run file (result)
    :param k: integer to be used as cutoff
    :param base: a number to be used as the log base for calculation if gdeval=False
    :param gdeval: boolean parameter that indicates whether to use the gdeval calculation or not
    :return:
    """

    # Reading and sorting the qrels, to later speed-up indexing and locating
    qrels_df = pd.read_csv(qrels_file, delim_whitespace=True, names=TREC_QREL_COLUMNS). \
        sort_values(['qid', 'rel', 'docNo'], ascending=[True, False, True]).set_index(['qid', 'docNo'])
    qrels_df['rel'].clip(lower=0, inplace=True)

    # Store beginning and end indices for each query - used for speed up
    qid_res_len = qrels_df.groupby('qid').apply(len)
    qid_end_loc = qid_res_len.cumsum()
    qid_start_loc = qid_end_loc - qid_res_len
    qrels_df = qrels_df.droplevel(0)

    results_df = pd.read_csv(results_file, delim_whitespace=True, names=TREC_RES_COLUMNS)

    # if calculating according to the gdeval script, the results are sorted by docScore and ties are broken by docNo
    if gdeval:
        results_df = results_df.sort_values(['qid', 'docScore', 'docNo'], ascending=[True, False, False]).groupby(
            'qid').head(k)
        discount = np.log(np.arange(1, k + 1) + 1)
    # Otherwise, sort by doc ranks
    else:
        results_df = results_df.sort_values(['qid', 'rank']).groupby('qid').head(k)
        discount = np.concatenate((np.ones(base), np.log(np.arange(base, k) + 1) / np.log(base)))

    result = {}
    for qid, _df in results_df.groupby('qid'):
        docs = _df['docNo'].to_numpy()
        try:
            _qrels_df = qrels_df.iloc[qid_start_loc.loc[qid]: qid_end_loc.loc[qid]]
        except KeyError as err:
            print(f'query id {err} doesn\'t exist in the qrels file, skipping it')
            continue

        if gdeval:
            dcg = 2 ** _qrels_df.reindex(docs)['rel'].fillna(0).to_numpy() - 1
            idcg = ((2 ** _qrels_df['rel'].head(k) - 1) / discount[:len(_qrels_df)]).sum()
        else:
            dcg = _qrels_df.reindex(docs)['rel'].fillna(0).to_numpy()
            idcg = (_qrels_df['rel'].head(k) / discount[:len(_qrels_df)]).sum()
        result[qid] = (dcg / discount[:len(dcg)]).sum() / idcg
    res_df = pd.DataFrame.from_dict(result, orient='index', columns=[f'nDCG@{k}'])
    #     res_df.to_csv(rreplace(results_file, 'run', f'ndcg@{k}', 1), sep='\t', float_format='%.6f', header=False)
    print(res_df.to_string(float_format='%.5f'))
    print(f'Mean: {res_df.mean()[0]:.5f}')
    return res_df


def read_survey_data(ranks_df_path, ratings_df_path):
    col_names = ['rid', 'topic', 'query', 'value', 'wid', 'batch', 'duration', 'EndDate', 'query_mean']
    dtypes = {'rid': str, 'topic': str, 'query': str, 'value': int, 'wid': str, 'batch': str, 'duration': int,
              'query_mean': float}
    ranks_df = pd.read_csv(ranks_df_path, index_col='rid', names=col_names, dtype=dtypes, header=0,
                           parse_dates=['EndDate'])
    rates_df = pd.read_csv(ratings_df_path, index_col='rid', names=col_names, dtype=dtypes, header=0,
                           parse_dates=['EndDate'])
    ranks_df['value'] = ranks_df['value'].max() - ranks_df['value'] + 1

    return ranks_df, rates_df
