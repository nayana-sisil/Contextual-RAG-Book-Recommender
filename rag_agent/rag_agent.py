import os
import re
import json
import pandas as pd
import numpy as np
from typing import Optional
from dotenv import load_dotenv
 
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
 
from observability import setup_langsmith, RunTracker
from reranker import BookReranker
from llm_local import get_llm, EXPLAIN_PROMPT, QUERY_ANALYSIS_PROMPT

load_dotenv()

_books_df:  Optional[pd.DataFrame] = None
_db_books:  Optional[Chroma]       = None
_reranker:  Optional[BookReranker] = None
_llm                               = None
_tracker:   RunTracker             = RunTracker()


#intialize

def initialize(
    csv_path:    str = "./dataset/books_with_emotions.csv",
    txt_path:    str = "./tagged_description.txt",
    llm_model:   str = "google/flan-t5-base",
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
 
    _llm = get_llm(model=llm_model)
 
    print("[Init] All components ready.\n")
    return _books_df

#semantic similarity search

def _vector_search(query: str, k: int = 50) -> pd.DataFrame:
    _tracker.log_step("vector-search")
    results = _db_books.similarity_search(query, k=k)
    isbn_list = []
    for r in results:
        tokens = r.page_content.strip().split()
        if tokens and tokens[0].isdigit():
            isbn_list.append(int(tokens[0]))
 
    matches = _books_df[_books_df["isbn13"].isin(isbn_list)].copy()
    _tracker.candidates = len(matches)
    return matches


#metadata filtering

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

#reranker

def _rerank(query: str, df: pd.DataFrame, top_k: int = 16) -> pd.DataFrame:
    """Cross-encoder rerank for final precision."""
    _tracker.log_step("cross-encoder-rerank")
    ranked = _reranker.rerank(query, df, text_column="description", top_k=top_k)
    _tracker.after_rerank = len(ranked)
    _tracker.top_score    = _reranker.top_score(ranked)
    return ranked


#explain about book
 
def _explain_book(query: str, title: str, description: str) -> str:
    _tracker.llm_calls += 1
    prompt = EXPLAIN_PROMPT.format(
        query=query, title=title, description=description[:400]
    )
    try:
        return _llm.invoke(prompt).strip()
    except Exception:
        return f"'{title}' matches your request through its themes of {query[:50]}."

#analysie query and  find tone

def _analyze_query(query: str) -> dict:
    prompt = QUERY_ANALYSIS_PROMPT.format(query=query)
    try:
        raw = _llm.invoke(prompt).strip()
        raw = re.sub(r"```json|```", "", raw).strip()
        return json.loads(raw)
    except Exception:
        return {"themes": [], "tone": "neutral", "category": "all", "keywords": []}


#tool wrapper

def make_tools(query_ref: dict):
    
    def tool_vector_search(input_str: str) -> str:
        df = _vector_search(input_str, k=50)
        query_ref["df"] = df
        return f"Found {len(df)} candidate books via vector search."
 
    def tool_metadata_filter(input_str: str) -> str:
        parts   = input_str.split("|")
        cat     = parts[0].strip() if len(parts) > 0 else "All"
        tone    = parts[1].strip() if len(parts) > 1 else "All"
        df      = query_ref.get("df", _books_df)
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
            zip(df["title"].head(5).tolist(), [e.split(": ", 1)[-1] for e in explanations])
        )
        return "\n".join(explanations)
 
    return [
        Tool(
            name="vector_search",
            func=tool_vector_search,
            description=(
                "Search for books semantically similar to a query. "
                "Use this FIRST with the user's query to get candidate books. "
                "Input: the search query string."
            ),
        ),
        Tool(
            name="metadata_filter",
            func=tool_metadata_filter,
            description=(
                "Filter books by category and emotional tone. "
                "Input format: 'category|tone' e.g. 'Fiction|sad' or 'All|happy'. "
                "Call AFTER vector_search."
            ),
        ),
        Tool(
            name="rerank",
            func=tool_rerank,
            description=(
                "Rerank the candidate books using a cross-encoder for higher precision. "
                "Input: the original user query. Call AFTER metadata_filter."
            ),
        ),
        Tool(
            name="explain_books",
            func=tool_explain,
            description=(
                "Generate a short explanation for why each top book matches the user's query. "
                "Input: the original user query. Call LAST, after rerank."
            ),
        ),
    ]


 
 



















