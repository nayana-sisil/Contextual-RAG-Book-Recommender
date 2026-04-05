import os
import re
import json
import pandas as pd
import numpy as np
from typing import Optional
from dotenv import load_dotenv

from langchain.tools import Tool
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_chroma import Chroma
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch

from observability_new import setup_langsmith, RunTracker
from reranker import BookReranker
from llm_local import EXPLAIN_PROMPT, QUERY_ANALYSIS_PROMPT

load_dotenv()

_books_df: Optional[pd.DataFrame] = None
_db_books: Optional[Chroma]       = None
_reranker: Optional[BookReranker] = None
_llm                               = None
_tracker:  RunTracker              = RunTracker()


def get_llm(model_name: str = "google/flan-t5-base") -> HuggingFacePipeline:
    """
    Load flan-t5 (seq2seq) correctly.
    Uses the seq2seq task directly via AutoModelForSeq2SeqLM so we avoid
    the text-generation vs text2text-generation pipeline confusion entirely.
    """
    print(f"[LLM] Loading {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
    )

    
    class FlanT5LLM:
        def invoke(self, prompt: str) -> str:
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                max_length=512,
                truncation=True,
            )
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                num_beams=2,
                early_stopping=True,
            )
            return tokenizer.decode(outputs[0], skip_special_tokens=True)

    print(f"[LLM] {model_name} ready.")
    return FlanT5LLM()


def initialize(
    csv_path: str          = "../dataset/books_with_emotions.csv",
    txt_path: str          = "../tagged_description.txt",
    llm_model: str         = "google/flan-t5-base",
    langsmith_project: str = "bookmind-rag",
):
    global _books_df, _db_books, _reranker, _llm

    setup_langsmith(project=langsmith_project)

    print("[Init] Loading book dataset...")
    _books_df = pd.read_csv(csv_path)
    _books_df["large_thumbnail"] = _books_df["thumbnail"].fillna("") + "&fife=w800"
    _books_df["large_thumbnail"] = np.where(
        _books_df["large_thumbnail"].str.strip() == "&fife=w800",
        "cover-not-found.jpg",
        _books_df["large_thumbnail"],
    )
    print(f"[Init] Loaded {len(_books_df)} books.")

    print("[Init] Building ChromaDB vector store...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )

    with open(txt_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    book_chunks = re.split(r'(?=\d{13} )', raw_text)
    book_chunks = [b.strip() for b in book_chunks if b.strip()]

    _db_books = Chroma.from_texts(
        book_chunks,
        embedding=embeddings,
        collection_name="bookmind_v2",
    )
    print(f"[Init] Vector store ready with {len(book_chunks)} chunks.")

    _reranker = BookReranker()
    _llm      = get_llm(llm_model)

    print("[Init] All components ready.\n")
    return _books_df


def _vector_search(query: str, k: int = 50) -> pd.DataFrame:
    _tracker.log_step("vector-search")
    results   = _db_books.similarity_search(query, k=k)
    isbn_list = []
    for r in results:
        tokens = r.page_content.strip().split()
        if tokens and tokens[0].isdigit():
            isbn_list.append(int(tokens[0]))

    matches = _books_df[_books_df["isbn13"].isin(isbn_list)].copy()
    _tracker.candidates = len(matches)
    return matches


def _metadata_filter(df: pd.DataFrame, category: str, tone: str) -> pd.DataFrame:
    _tracker.log_step("metadata-filter")

    if category and category.lower() not in ("all", ""):
        df = df[df["simple_categories"] == category]

    tone_map = {
        "happy":       "joy",
        "surprising":  "surprise",
        "angry":       "anger",
        "suspenseful": "fear",
        "sad":         "sadness",
    }
    tone_col = tone_map.get(tone.lower() if tone else "", None)
    if tone_col and tone_col in df.columns:
        df = df.sort_values(by=tone_col, ascending=False)

    _tracker.after_filter = len(df)
    return df.head(80)


def _rerank(query: str, df: pd.DataFrame, top_k: int = 16) -> pd.DataFrame:
    _tracker.log_step("cross-encoder-rerank")
    ranked            = _reranker.rerank(query, df, text_column="description", top_k=top_k)
    _tracker.after_rerank = len(ranked)
    _tracker.top_score    = _reranker.top_score(ranked)
    return ranked


def _explain_book(query: str, title: str, description: str) -> str:
    _tracker.llm_calls += 1
    prompt = EXPLAIN_PROMPT.format(
        query=query, title=title, description=description[:400]
    )
    try:
        return _llm.invoke(prompt).strip()
    except Exception:
        return f"'{title}' matches your request through its themes of {query[:50]}."


def _analyze_query(query: str) -> dict:
    prompt = QUERY_ANALYSIS_PROMPT.format(query=query)
    try:
        raw = _llm.invoke(prompt).strip()
        raw = re.sub(r"```json|```", "", raw).strip()
        return json.loads(raw)
    except Exception:
        return {"themes": [], "tone": "neutral", "category": "all", "keywords": []}



def make_tools(query_ref: dict):

    def tool_vector_search(input_str: str) -> str:
        df = _vector_search(input_str, k=50)
        query_ref["df"] = df
        return f"Found {len(df)} candidate books via vector search."

    def tool_metadata_filter(input_str: str) -> str:
        parts    = input_str.split("|")
        cat      = parts[0].strip() if len(parts) > 0 else "All"
        tone     = parts[1].strip() if len(parts) > 1 else "All"
        df       = query_ref.get("df", _books_df)
        filtered = _metadata_filter(df, cat, tone)
        query_ref["df"] = filtered
        return f"After filtering: {len(filtered)} books remain."

    def tool_rerank(input_str: str) -> str:
        df    = query_ref.get("df", _books_df)
        final = _rerank(input_str, df, top_k=16)
        query_ref["final_df"] = final
        top3  = final[["title", "rerank_score"]].head(3).to_string(index=False)
        return f"Reranked to top 16. Top 3:\n{top3}"

    def tool_explain(input_str: str) -> str:
        df = query_ref.get("final_df", query_ref.get("df", pd.DataFrame()))
        if df.empty:
            return "No books to explain."
        explanations = []
        for _, row in df.head(5).iterrows():
            exp = _explain_book(input_str, row["title"], str(row.get("description", "")))
            explanations.append(f"• {row['title']}: {exp}")
        query_ref["explanations"] = dict(
            zip(
                df["title"].head(5).tolist(),
                [e.split(": ", 1)[-1] for e in explanations],
            )
        )
        return "\n".join(explanations)

    return [
        Tool(name="vector_search",   func=tool_vector_search,   description="Search for books semantically similar to a query. Input: the search query string."),
        Tool(name="metadata_filter", func=tool_metadata_filter, description="Filter books by category and emotional tone. Input format: 'category|tone'."),
        Tool(name="rerank",          func=tool_rerank,          description="Rerank candidate books with a cross-encoder. Input: the original user query."),
        Tool(name="explain_books",   func=tool_explain,         description="Generate explanations for top books. Input: the original user query."),
    ]



def run_agent(
    query:    str,
    category: str = "All",
    tone:     str = "All",
) -> dict:
    """
    Runs the full RAG pipeline in a fixed order:
      1. vector_search
      2. metadata_filter
      3. rerank
      4. explain_books

    Returns:
        {
          "books":        pd.DataFrame of final recommendations,
          "explanations": dict of title -> explanation string,
          "metrics":      dict from RunTracker,
          "reasoning":    str,
        }
    """
    global _tracker
    _tracker       = RunTracker()
    _tracker.query = query

    print(f"\n{'='*60}")
    print(f"[Pipeline] Query: {query!r}")
    print(f"[Pipeline] Category: {category} | Tone: {tone}")
    print(f"{'='*60}")

    query_ref: dict = {}
    tools = make_tools(query_ref)

    print("[Step 1] Running vector search...")
    result1 = tools[0].func(query)
    print(f"         {result1}")

    print("[Step 2] Running metadata filter...")
    result2 = tools[1].func(f"{category}|{tone}")
    print(f"         {result2}")

    print("[Step 3] Running reranker...")
    result3 = tools[2].func(query)
    print(f"         {result3}")

    print("[Step 4] Generating explanations...")
    result4 = tools[3].func(query)
    print(f"         Explanations generated.")

    final_df     = query_ref.get("final_df", pd.DataFrame())
    explanations = query_ref.get("explanations", {})

    if not explanations and not final_df.empty:
        _tracker.log_step("explain-fallback")
        for _, row in final_df.head(8).iterrows():
            explanations[row["title"]] = _explain_book(
                query, row["title"], str(row.get("description", ""))
            )

    _tracker.reasoning = f"Direct pipeline for: {query[:80]}"

    print(f"\n[Pipeline] {_tracker.summary()}")

    return {
        "books":        final_df,
        "explanations": explanations,
        "metrics":      _tracker.to_dict(),
        "reasoning":    _tracker.reasoning,
    }



if __name__ == "__main__":
    initialize()
    result = run_agent(
        query    = "a story about grief and unexpected friendship in a small town",
        category = "Fiction",
        tone     = "Sad",
    )

    print("\n── Top 3 Results ──────────────────────────────────────")
    if not result["books"].empty:
        print(result["books"][["title", "authors", "rerank_score"]].head(3).to_string(index=False))
    else:
        print("No results found.")

    print("\n── Explanations ───────────────────────────────────────")
    for title, explanation in list(result["explanations"].items())[:3]:
        print(f"• {title}:\n  {explanation}\n")

    print("── Metrics ────────────────────────────────────────────")
    print(result["metrics"])