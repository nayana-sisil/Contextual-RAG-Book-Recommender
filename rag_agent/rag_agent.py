import os
import re
import json
import time
import pandas as pd
import numpy as np
from typing import Optional
from dotenv import load_dotenv

from langchain.tools import Tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

from observability_new import setup_langsmith, RunTracker
from reranker import BookReranker
from llm_local import EXPLAIN_PROMPT, QUERY_ANALYSIS_PROMPT

load_dotenv()

# ── Globals ───────────────────────────────────────────────────────────────────

_books_df: Optional[pd.DataFrame] = None
_db_books: Optional[Chroma]       = None
_reranker: Optional[BookReranker] = None
_llm                               = None
_tracker:  RunTracker              = RunTracker()

# ── Agentic thresholds (tune these) ──────────────────────────────────────────

SCORE_THRESHOLD = 0.0   # below this → rewrite query
MIN_RESULTS     = 3     # below this → rewrite query
MAX_RETRIES     = 2     # max rewrite attempts before giving up


# ── LLM ───────────────────────────────────────────────────────────────────────

class FlanT5LLM:
    """
    Thin wrapper around flan-t5-base.
    Exposes a simple .invoke(prompt) interface used for:
      - query rewriting
      - book explanations
      - query analysis
    """

    def __init__(self, model_name: str = "google/flan-t5-base"):
        print(f"[LLM] Loading {model_name} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model     = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
        )
        self.model.eval()
        print(f"[LLM] {model_name} ready.")

    def invoke(self, prompt: str, max_new_tokens: int = 128) -> str:
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=512,
            truncation=True,
        )
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=2,
                early_stopping=True,
            )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()


# ── Initialization ────────────────────────────────────────────────────────────

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
    _llm      = FlanT5LLM(llm_model)

    print("[Init] All components ready.\n")
    return _books_df


# ── Core pipeline steps ───────────────────────────────────────────────────────

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
    df     = df.drop_duplicates(subset=["title"]).copy()
    ranked = _reranker.rerank(query, df, text_column="description", top_k=top_k)
    _tracker.after_rerank = len(ranked)
    _tracker.top_score    = _reranker.top_score(ranked)
    return ranked


def _explain_book(query: str, title: str, description: str) -> str:
    _tracker.llm_calls += 1
    prompt = (
        f"Book title: {title}\n"
        f"Book description: {description[:300]}\n"
        f"In one sentence, explain what this book is about."
    )
    try:
        result = _llm.invoke(prompt, max_new_tokens=80)
        # Reject if model just echoes the query
        if not result or result.lower().strip() == query.lower().strip():
            raise ValueError("echo")
        return result
    except Exception:
        return description[:150].strip() + "..." if description else f"Matches themes of: {query[:60]}"


def _analyze_query(query: str) -> dict:
    prompt = QUERY_ANALYSIS_PROMPT.format(query=query)
    try:
        raw = _llm.invoke(prompt).strip()
        raw = re.sub(r"```json|```", "", raw).strip()
        return json.loads(raw)
    except Exception:
        return {"themes": [], "tone": "neutral", "category": "all", "keywords": []}


# ── Agentic behavior 1: Self-reflection ──────────────────────────────────────

def _reflection(df: pd.DataFrame, top_score: float) -> dict:
    """
    Decide if results are good enough or if we should retry.
    Returns: { satisfied, reason, action }
    """
    if df.empty or len(df) < MIN_RESULTS:
        return {
            "satisfied": False,
            "reason":    f"Only {len(df)} results (minimum: {MIN_RESULTS})",
            "action":    "rewrite_query",
        }
    if top_score < SCORE_THRESHOLD:
        return {
            "satisfied": False,
            "reason":    f"Top score {top_score:.3f} below threshold {SCORE_THRESHOLD}",
            "action":    "rewrite_query",
        }
    return {
        "satisfied": True,
        "reason":    f"Score {top_score:.3f} ≥ threshold, {len(df)} results found",
        "action":    "done",
    }


# ── Agentic behavior 2: Query rewriting ──────────────────────────────────────

def _rewrite_query(original_query: str, attempt: int) -> str:
    """
    Use LLM to produce a broader/alternative query.
    Falls back to manual expansion if LLM output is poor.
    """
    strategies = [
        f"Rewrite this book search to be more general using different words: {original_query}",
        f"Suggest an alternative way to search for books about: {original_query}",
    ]
    prompt    = strategies[min(attempt - 1, len(strategies) - 1)]
    rewritten = _llm.invoke(prompt, max_new_tokens=64)

    print(f"[Agent] LLM rewrite: {rewritten!r}")

    # Reject if too short or identical
    if (
        not rewritten
        or len(rewritten.strip()) < 8
        or rewritten.lower().strip() == original_query.lower().strip()
    ):
        fallbacks = [
            f"emotional literary fiction {original_query}",
            f"novels about {original_query} themes loss connection",
        ]
        rewritten = fallbacks[min(attempt - 1, len(fallbacks) - 1)]
        print(f"[Agent] Fallback rewrite: {rewritten!r}")

    return rewritten


# ── Tool wrappers ─────────────────────────────────────────────────────────────

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
        df = query_ref.get("df", _books_df)

        # ── Guard: nothing to rerank ──────────────────────────
        if df.empty:
            query_ref["final_df"] = df
            return "No books to rerank — filter returned 0 results."

        final = _rerank(input_str, df, top_k=16)
        query_ref["final_df"] = final

        # ── Guard: rerank returned empty ──────────────────────
        if final.empty or "rerank_score" not in final.columns:
            return "Reranker returned no results."

        top3 = final[["title", "rerank_score"]].head(3).to_string(index=False)
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
        Tool(name="vector_search",   func=tool_vector_search,   description="Search for books semantically similar to a query."),
        Tool(name="metadata_filter", func=tool_metadata_filter, description="Filter books by category and tone. Input: 'category|tone'."),
        Tool(name="rerank",          func=tool_rerank,          description="Rerank candidate books with a cross-encoder."),
        Tool(name="explain_books",   func=tool_explain,         description="Generate explanations for top books."),
    ]


# ── Main agentic entry point ──────────────────────────────────────────────────

def run_agent(
    query:    str,
    category: str = "All",
    tone:     str = "All",
) -> dict:
    """
    Agentic RAG with self-reflection + query rewriting.

    Flow:
    ┌──────────────────────────────────────────────────────┐
    │  vector_search → metadata_filter → rerank            │
    │                      ↓                               │
    │                  [reflect]                           │
    │                satisfied? ──yes──→ explain → return  │
    │                    │ no                              │
    │             rewrite_query → retry (max MAX_RETRIES)  │
    └──────────────────────────────────────────────────────┘

    Returns:
        {
          "books":         pd.DataFrame,
          "explanations":  dict[title, str],
          "metrics":       dict,
          "reasoning":     str,
          "query_history": list[str],
          "reflections":   list[dict],
        }
    """
    global _tracker
    _tracker       = RunTracker()
    _tracker.query = query

    start_time     = time.time()
    active_query   = query
    query_history  = [query]
    reflection_log = []
    final_df       = pd.DataFrame()
    attempt        = 0

    print(f"\n{'='*60}")
    print(f"[Agent] Original query : {query!r}")
    print(f"[Agent] Category: {category} | Tone: {tone}")
    print(f"{'='*60}")

    query_ref: dict = {}
    tools = make_tools(query_ref)

    # ── Agentic loop ──────────────────────────────────────────────────────────
    while attempt <= MAX_RETRIES:
        print(f"\n[Agent] ── Attempt {attempt + 1}/{MAX_RETRIES + 1} ──────────────")
        print(f"[Agent] Query: {active_query!r}")

        # Step 1: Vector search
        r1 = tools[0].func(active_query)
        print(f"  [Step 1] Vector search    → {r1}")

        # Step 2: Metadata filter
        r2 = tools[1].func(f"{category}|{tone}")
        print(f"  [Step 2] Metadata filter  → {r2}")

        # Step 3: Rerank
        r3 = tools[2].func(active_query)
        print(f"  [Step 3] Rerank           → {r3}")

        df        = query_ref.get("final_df", pd.DataFrame())
        top_score = _tracker.top_score if not df.empty else -999.0

        # Step 4: Reflect
        reflection = _reflection(df, top_score)
        reflection_log.append({
            "attempt":   attempt + 1,
            "query":     active_query,
            "top_score": round(top_score, 4),
            "n_results": len(df),
            **reflection,
        })
        print(f"  [Reflect] satisfied={reflection['satisfied']} — {reflection['reason']}")

        if reflection["satisfied"] or attempt >= MAX_RETRIES:
            final_df = df
            if not reflection["satisfied"]:
                print(f"[Agent] Max retries reached — accepting best available results.")
            else:
                print(f"[Agent] Results accepted ✓")
            break

        # Not satisfied → rewrite and retry
        attempt      += 1
        active_query  = _rewrite_query(query, attempt)
        query_history.append(active_query)
        # Reset query_ref for next attempt
        query_ref.clear()
        tools = make_tools(query_ref)

    # ── Step 5: Explain (once, on final accepted results) ─────────────────────
    explanations: dict = {}

    # ── Guard: no results at all ──────────────────────────────────────────────
    if final_df.empty:
        reasoning = (
            f"No results found after {attempt + 1} attempt(s). "
            f"Try selecting 'All' for category and tone, or broaden your query."
        )
        print(f"[Agent] No results found.")
    else:
        print(f"\n[Agent] Generating explanations for top {min(5, len(final_df))} books...")
        r4 = tools[3].func(query)   # always explain against ORIGINAL query
        explanations = query_ref.get("explanations", {})

        if len(query_history) > 1:
            trail     = " → ".join(f'"{q}"' for q in query_history)
            reasoning = (
                f"Query rewritten {len(query_history) - 1} time(s): {trail}. "
                f"Final top score: {_tracker.top_score:.3f}."
            )
        else:
            reasoning = (
                f"Query accepted on first attempt. "
                f"Final top score: {_tracker.top_score:.3f}."
            )

    _tracker.reasoning = reasoning
    total_s = round(time.time() - start_time, 2)

    print(f"\n[Agent] Finished in {total_s}s | {_tracker.summary()}")

    return {
        "books":         final_df,
        "explanations":  explanations,
        "metrics":       _tracker.to_dict(),
        "reasoning":     reasoning,
        "query_history": query_history,
        "reflections":   reflection_log,
    }


# ── CLI entry ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    initialize()

    result = run_agent(
        query    = "a story about grief and unexpected friendship in a small town",
        category = "Fiction",
        tone     = "Sad",
    )

    print("\n── Query History ──────────────────────────────────────")
    for i, q in enumerate(result["query_history"]):
        label = "original" if i == 0 else f"rewrite {i}"
        print(f"  [{label}] {q}")

    print("\n── Reflection Log ─────────────────────────────────────")
    for r in result["reflections"]:
        print(f"  Attempt {r['attempt']}: score={r['top_score']:.3f}, n={r['n_results']}, satisfied={r['satisfied']}")
        print(f"    reason: {r['reason']}")

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

    print("\n── Reasoning ──────────────────────────────────────────")
    print(result["reasoning"])