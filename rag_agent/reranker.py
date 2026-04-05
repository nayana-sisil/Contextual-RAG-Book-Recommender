import numpy as np
import pandas as pd
from typing import List, Tuple


class BookReranker:

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self._model = None  

    def _load(self):
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
                print(f"[Reranker] Loading {self.model_name}...")
                self._model = CrossEncoder(self.model_name, max_length=512)
                print("[Reranker] Ready.")
            except ImportError:
                raise ImportError(
                    "sentence-transformers not installed.\n"
                    "Run: pip install sentence-transformers"
                )

    def rerank(
        self,
        query: str,
        books_df: pd.DataFrame,
        text_column: str = "description",
        top_k: int = 16,
    ) -> pd.DataFrame:
        
        self._load()

        if books_df.empty:
            return books_df

        descriptions = books_df[text_column].fillna("").tolist()
        pairs = [(query, desc[:512]) for desc in descriptions]  

        scores = self._model.predict(pairs, show_progress_bar=False)

        books_df = books_df.copy()
        books_df["rerank_score"] = scores

        return (
            books_df
            .sort_values("rerank_score", ascending=False)
            .head(top_k)
            .reset_index(drop=True)
        )

    def top_score(self, df: pd.DataFrame) -> float:
        if "rerank_score" in df.columns and not df.empty:
            return float(df["rerank_score"].max())
        return 0.0


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    """Map cross-encoder logits to 0–1 via sigmoid."""
    return 1 / (1 + np.exp(-scores))


if __name__ == "__main__":
    import pandas as pd
    reranker = BookReranker()

    test_df = pd.DataFrame({
        "title":       ["A Little Life", "Python Cookbook", "Grief Is the Thing with Feathers"],
        "description": [
            "Four friends navigate trauma and grief over decades in New York.",
            "Recipes and techniques for Python programming.",
            "A grieving father is visited by a crow who helps him through loss.",
        ]
    })

    query = "a book about grief and unexpected friendship"
    result = reranker.rerank(query, test_df, text_column="description", top_k=3)
    print(result[["title", "rerank_score"]])