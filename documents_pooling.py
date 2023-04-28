import argparse
from glob import glob

import numpy as np
import pandas as pd

TREC_QREL_COLUMNS = ['qid', 'iteration', 'docNo', 'rel']
TREC_RES_COLUMNS = ['qid', 'iteration', 'docNo', 'rank', 'docScore', 'method']


def calc_ndcg(qrels_file, results_file, k, logger=None, base=2, gdeval=True, non_judged_score=0,
              return_non_judged=False):
    """
    Setting gdeval will yield identical results to the official evaluation script that was published for TREC.
    Note that the calculation in that script differs from (probably) any published research version.
    :param non_judged_score:
    :param return_non_judged:
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
        raise ValueError('qrels_file must be a string or a pandas dataframe')
    # Reading and sorting the qrels, to later speed-up indexing and locating
    qrels_df = qrels_df.sort_values(['qid', 'rel', 'docNo'], ascending=[True, False, True]).set_index(['qid', 'docNo'])
    qrels_df['rel'].clip(lower=0, inplace=True)

    # Store beginning and end indices for each query - used for speed up
    qid_res_len = qrels_df.groupby('qid').apply(len)
    qid_end_loc = qid_res_len.cumsum()
    qid_start_loc = qid_end_loc - qid_res_len
    qrels_df = qrels_df.droplevel(0)

    nonjudged_docs = {}
    nonjudged = 0

    if isinstance(results_file, str):
        results_df = pd.read_csv(results_file, delim_whitespace=True, names=TREC_RES_COLUMNS)
    elif isinstance(results_file, pd.DataFrame):
        results_df = results_file
    else:
        raise ValueError('results_file must be a string or a pandas dataframe')

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
        # _nonj_idx = _qrels_df.reindex(docs)['rel'].isna().to_numpy()
        _nonj_idx = _df.loc[~_df['docNo'].isin(_qrels_df.index), 'docNo']
        nonjudged_docs[qid] = _nonj_idx.to_list()
        # nonjudged_docs += _nonj_idx.to_list()
        nonjudged += len(_nonj_idx)
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
    if return_non_judged:
        return res_df, nonjudged_docs
    else:
        return res_df


def get_run_files():
    run_files = glob('runs/selected_runs/*run')
    return run_files


def main(parser):
    # parser.add_argument('qrels_file', help='qrels file')
    # parser.add_argument('results_file', help='results file')
    # parser.add_argument('-k', type=int, default=10, help='k - cutoff')
    # parser.add_argument('-b', type=int, default=2, help='base of the logarithm')
    # parser.add_argument('-g', action='store_true', help='use gdeval')
    # parser.add_argument('-n', type=float, default=0, help='non-judged document score')
    # parser.add_argument('-r', action='store_true', help='return non-judged docs')
    # args = parser.parse_args()
    # res_df = calc_ndcg(args.qrels_file, args.results_file, args.k, base=args.b, gdeval=args.g, non_judged_score=args.n,
    #                    return_non_judged=args.r)
    # print(res_df.to_string(float_format='%.5f'))
    # print(f'Mean: {res_df.mean()[0]:.5f}')
    qrels_file = 'uqv100-allQueries-posit.qrels'
    # qrels_df = pd.read_csv(qrels_file, delim_whitespace=True, names=TREC_QREL_COLUMNS)
    run_files = get_run_files()
    docs_to_judge = {}
    for run_file in run_files:
        res_df, non_judged_docs = calc_ndcg(qrels_file=qrels_file, results_file=run_file, k=10, base=2, gdeval=True,
                                            return_non_judged=True)
        docs_to_judge[run_file] = non_judged_docs


if __name__ == '__main__':
    PARSER = argparse.ArgumentParser(description='Calculate nDCG@k')
    main(PARSER)
