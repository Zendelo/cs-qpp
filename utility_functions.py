import linecache
import os
import re
from bisect import bisect_left

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


def get_file_len(file_path):
    """Opens a file and counts the number of lines in it"""
    return sum(1 for _ in open(file_path))


def read_line(file_path, n):
    """Return a specific line n from a file, if the line doesn't exist, returns an empty string"""
    return linecache.getline(file_path, n)


def binary_search(list_, target):
    """Return the index of first value equal to target, if non found will raise a ValueError"""
    i = bisect_left(list_, target)
    if i != len(list_) and list_[i] == target:
        return i
    raise ValueError


def ensure_file(file):
    """Ensure a single file exists, returns the absolute path of the file if True or raises FileNotFoundError if not"""
    # tilde expansion
    file_path = os.path.normpath(os.path.expanduser(file))
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"The file {file_path} doesn't exist")
    return file_path


def ensure_dir(file_path, create_if_not=True):
    """
    The function ensures the dir exists,
    if it doesn't it creates it and returns the path or raises FileNotFoundError
    """
    # tilde expansion
    file_path = os.path.normpath(os.path.expanduser(file_path))
    if os.path.isfile(file_path):
        directory = os.path.dirname(file_path)
    else:
        directory = file_path
    if not os.path.exists(directory):
        if create_if_not:
            try:
                os.makedirs(directory)
            except FileExistsError:
                # This exception was added for multiprocessing, in case multiple process try to create the directory
                pass
        else:
            raise FileNotFoundError(f"The directory {directory} doesnt exist, create it or pass create_if_not=True")
    return directory


# def read_message(buffer, n, message_type):
#     """
#     The function used to read and parse a protobuf message, specifically for delimited protobuf files
#     """
#     message = message_type()
#     msg_len, new_pos = _DecodeVarint32(buffer, n)
#     n = new_pos
#     msg_buf = buffer[n:n + msg_len]
#     n += msg_len
#     message.ParseFromString(msg_buf)
#     return n, message


def transform_list_to_counts_dict(_list):
    counts = [_list.count(i) for i in _list]
    return {i: j for i, j in zip(_list, counts)}


def jaccard_similarity(set_1, set_2):
    return len(set_1.intersection(set_2)) / len(set_1.union(set_2))


def overlap_coefficient(set_1, set_2):
    return len(set_1.intersection(set_2)) / min(len(set_1), len(set_2))


def duplicate_qrel_file_to_qids(qrel_file, qids):
    qrels_df = pd.read_table(qrel_file, delim_whitespace=True, names=['qid', 'iter', 'doc', 'rel'], dtype=str)
    topics_dict = {qid: qid.split('-', 1)[0] for qid in qids}  # {qid:topic}
    result = []
    for qid, topic in topics_dict.items():
        result.append(
            qrels_df.loc[qrels_df['qid'] == topic,
            ['iter', 'doc', 'rel']].assign(qid=qid).loc[:, ['qid', 'iter', 'doc', 'rel']])
    res_df = pd.concat(result, axis=0)
    res_df.to_csv(qrel_file.replace('.qrels', '_mod.qrels'), sep=' ', index=False, header=False)


def add_topic_to_qdf(qdf: pd.DataFrame):
    """This function used to add a topic column to a queries DF"""
    columns = qdf.columns.to_list()
    if 'topic' not in columns:
        if 'qid' in columns:
            qdf = qdf.assign(topic=lambda x: x.qid.apply(lambda y: y.split('-')[0]))
        else:
            qdf = qdf.reset_index().assign(topic=lambda x: x.qid.apply(lambda y: y.split('-')[0]))
    columns = qdf.columns.to_list()
    return qdf.loc[:, columns[-1:] + columns[:-1]]


# def msgpack_encode(vector):
#     return msgpack.packb(vector)
#
#
# def msgpack_decode(serialized_vector):
#     return msgpack.unpackb(serialized_vector)


def read_trec_res_file(file_name):
    """
    Assuming data is in trec format results file with 6 columns, 'Qid entropy cross_entropy Score
    '"""
    data_df = pd.read_csv(file_name, delim_whitespace=True, header=None, index_col=0,
                          names=['qid', 'Q0', 'docNo', 'docRank', 'docScore', 'ind'],
                          dtype={'qid': str, 'Q0': str, 'docNo': str, 'docRank': int, 'docScore': float,
                                 'ind': str})
    data_df = data_df.filter(['qid', 'docNo', 'docRank', 'docScore'], axis=1)
    data_df.index = data_df.index.astype(str)
    data_df.sort_values(by=['qid', 'docRank'], ascending=True, inplace=True)
    return data_df

# def plot_roc(y_test, y_pred, predictor_name):
#     fpr, tpr, threshold = roc_curve(y_test, y_pred)
#     plt.title(f'ROC {predictor_name}')
#     plt.plot(fpr, tpr, 'b', label=f'ROC (AUC = {auc(fpr, tpr):0.2f})')
#     # plt.plot(fpr, threshold, label='Threshold')
#     plt.legend(loc='lower right')
#     plt.plot([0, 1], [0, 1], 'r--')
#     plt.xlim([0, 1])
#     plt.ylim([0, 1])
#     plt.ylabel('True Positive Rate (Sensitivity/Recall)')
#     plt.xlabel('False Positive Rate (1-Specificity)')
#     plt.show()
