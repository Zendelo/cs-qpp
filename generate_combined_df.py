import os.path

import pandas as pd

from utility_functions import read_survey_data

# This topics classification was done with ChatGPT
TOPICS_CLASSIFICATION = {'259': 'Understand',
                         '223': 'Remember',
                         '238': 'Remember',
                         '297': 'Understand',
                         '213': 'Understand',
                         '266': 'Remember',
                         '234': 'Understand',
                         '253': 'Analyze',
                         '286': 'Analyze',
                         '261': 'Analyze',
                         '283': 'Remember',
                         '244': 'Understand'}


def read_queries_df(ndcg_df, ranks_sr, rates_sr):
    qdf = pd.read_csv('data/QueriesSurvey.csv', header=None, names=['qid', 'query'], index_col='qid').applymap(
        str.strip)
    qdf.index = qdf.index.str.strip()
    qdf = qdf.merge(ndcg_df, left_index=True, right_index=True)
    qdf['topic'] = qdf.index.str.split('-').str[0]
    qdf = qdf.assign(avg_rate=qdf['query'].map(rates_sr), avg_rank=qdf['query'].map(ranks_sr))
    return qdf


def read_unique_users_queries(unique_user_queries_path):
    unique_user_queries = pd.read_csv(unique_user_queries_path, header=None, names=['qid', 'query'], index_col='query')
    return unique_user_queries.assign(rid=unique_user_queries.qid.str.rsplit('-', 1).str[1])


def read_ndcg_df(ndcg_df_path):
    return pd.read_csv(ndcg_df_path, sep='\t', header=None, names=['qid', f'nDCG@10'], index_col='qid')


def read_all_users_queries(all_user_queries_path, unique_user_queries_df, users_ndcg_df):
    all_user_queries = pd.read_csv(all_user_queries_path, header=None,
                                   names=['qid', 'user_query']).assign(
        topic=lambda x: x.qid.apply(lambda y: y.split('-')[0])).sort_values('qid')
    all_user_queries = all_user_queries.assign(rid=all_user_queries.qid.str.rsplit('-', 1).str[1])
    all_user_queries = all_user_queries.assign(method=all_user_queries.qid.str.split('-').str[1].str.capitalize())
    all_user_queries = all_user_queries.assign(topic=all_user_queries.qid.str.split('-').str[0])

    all_user_queries['ref_qid'] = all_user_queries['user_query'].apply(lambda x: unique_user_queries_df.loc[x, 'qid'])
    all_user_queries[f'user_nDCG@10'] = all_user_queries['ref_qid'].apply(lambda x: users_ndcg_df.loc[x, f'nDCG@10'])
    # all_user_queries.set_index(['method', 'topic', 'rid'])
    return all_user_queries


def combine_dataframes(ranks_df, rates_df, qdf, all_user_queries):
    comb_df = pd.concat([ranks_df.assign(method='Ranking'), rates_df.assign(method='Rating')]).sort_values(
        ['topic', 'method', 'query_mean', 'value'])
    comb_df['value'].clip(lower=1, inplace=True)  # clipping al rating to start from 1

    # assign batch order per wid by the time finished
    comb_df = comb_df.assign(batch_order=comb_df.groupby(['wid'])['EndDate'].rank(method="dense"))

    comb_df = comb_df.assign(qid=comb_df['query'].map(qdf.reset_index().set_index('query')['qid']))

    _comb_df = comb_df.reset_index().set_index(['method', 'topic', 'rid']).merge(
        all_user_queries.set_index(['method', 'topic', 'rid'])[[f'user_nDCG@10', 'user_query']], left_index=True,
        right_index=True)
    return _comb_df.reset_index()


def main():
    ranks_df_file = 'data/ranks_df_long.csv'
    ratings_df_file = 'data/ratings_df_long.csv'
    ndcg_df_path = 'data/user_queries/PL2.DFR.SD.ndcg@10'
    all_user_queries_path = 'data/user_queries/all_normalized_user_queries.csv'
    unique_user_queries_path = 'data/user_queries/unique_normalized_user_queries.csv'

    ranks_df, rates_df = read_survey_data(ranks_df_file, ratings_df_file)
    ndcg_df = pd.read_csv(ndcg_df_path, sep='\t', header=None, names=['qid', f'nDCG@10'], index_col='qid')
    ranks_sr = ranks_df.groupby(['query'])['value'].mean()
    rates_sr = rates_df.groupby(['query'])['value'].mean()
    qdf = read_queries_df(ndcg_df, ranks_sr, rates_sr)

    unique_user_queries_df = read_unique_users_queries(unique_user_queries_path)
    users_ndcg_df = read_ndcg_df(ndcg_df_path)
    all_user_queries = read_all_users_queries(all_user_queries_path, unique_user_queries_df, users_ndcg_df)
    comb_df = combine_dataframes(ranks_df, rates_df, qdf, all_user_queries)
    # add column with topic classification
    comb_df['topic_class'] = comb_df['topic'].map(TOPICS_CLASSIFICATION)
    print(comb_df.head())
    print(comb_df.info())
    comb_df.to_parquet('data/comb_df.parquet.zstd', compression='zstd')
    print(f'combined df written to {os.path.abspath("data/comb_df.parquet.zstd")}')
    # filter only the accepted queries
    # all_user_queries = all_user_queries.loc[all_user_queries['rid'].isin(comb_df.index)]


if __name__ == '__main__':
    main()
