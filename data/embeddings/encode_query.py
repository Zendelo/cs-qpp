import argparse
from typing import List

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# from sklearn.metrics.pairwise import cosine_similarity

SIMILARITIES_PATH = "similarities.npy"
DEFAULT_SBERT_MODEL = "sentence-transformers/all-mpnet-base-v2"


def load_queries(in_path: str) -> pd.DataFrame:
    df = pd.read_csv(in_path, sep=",", names=["qid", "query"])
    df["qid"] = df["qid"].str.strip()
    df["query"] = df["query"].str.strip()
    return df


def build_embeddings(model: str, queries: List[str]) -> np.ndarray:
    model = SentenceTransformer(model)
    embeddings = model.encode(queries, show_progress_bar=True)
    return embeddings


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--path", required=True)
    parser.add_argument("--model", default=DEFAULT_SBERT_MODEL)
    parser.add_argument("--output", required=True)

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    df = load_queries(args.path)

    embeddings = build_embeddings(args.model, df["query"].tolist())
    df.assign(embeddings=embeddings.tolist()).to_pickle(f'{args.output}_df.pkl')
    print(df.shape)
    assert len(df) == len(embeddings)
    print(f"Embedding shape: {embeddings.shape}")
    if args.output is not None:
        np.save(args.output, embeddings)

    # similarities = cosine_similarity(embeddings, metric="cosine")
    # np.save(SIMILARITIES_PATH, similarities)


if __name__ == "__main__":
    main()
