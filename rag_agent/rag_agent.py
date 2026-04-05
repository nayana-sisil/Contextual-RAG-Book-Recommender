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


 



















