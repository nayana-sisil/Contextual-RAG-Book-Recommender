<div align="center">

# 📚 BookMind
### Contextual RAG Book Recommender

*An AI that understands what you're in the mood for, not just what you search for.*

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-0.2+-1C3C3C?style=flat-square&logo=langchain&logoColor=white)](https://langchain.com)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Store-E57CD8?style=flat-square)](https://trychroma.com)
[![HuggingFace](https://img.shields.io/badge/🤗_HuggingFace-Transformers-FFD21E?style=flat-square)](https://huggingface.co)
[![Gradio](https://img.shields.io/badge/Gradio-UI-FF7C00?style=flat-square&logo=gradio)](https://gradio.app)

**[▶ Try the Live Demo](https://huggingface.co/spaces/nayanasisil2700/Contextual-RAG-Book-Recommender)**

</div>

---

## Overview

BookMind is an end-to-end **Agentic Retrieval-Augmented Generation (RAG)** system for book discovery. It goes beyond simple keyword matching by combining semantic vector search, emotion-aware filtering, cross-encoder reranking, and LLM-powered explanations — all orchestrated by an agent that can reflect on its own results and rewrite its query if needed.

Built on a corpus of **6,810 books** (cleaned to ~5,197 usable entries) sourced from Google Books metadata.

---

## Features

| | Feature | Description |
|---|---|---|
| 🔍 | **Semantic Search** | Queries embedded with `all-MiniLM-L6-v2` and matched against book descriptions in ChromaDB |
| 🎭 | **Emotion-Aware Filtering** | Books tagged across 6 emotional dimensions using a fine-tuned BERT model |
| 📂 | **Zero-Shot Classification** | Untagged books classified as Fiction or Nonfiction using `facebook/bart-large-mnli` |
| 🎯 | **Cross-Encoder Reranking** | Candidates reranked using `ms-marco-MiniLM-L-6-v2` for precision |
| 🔄 | **Self-Reflection and Query Rewriting** | Agent evaluates result quality and rewrites the query (up to 2 retries) if scores fall below threshold |
| ✨ | **LLM Explanations** | `flan-t5-base` generates a personalised "why this book" sentence for each result |
| 📡 | **LangSmith Observability** | Full pipeline tracing and run metrics |
| 🖼️ | **Gradio Dashboard** | Book-themed UI with live pipeline trace, reasoning sidebar, and cover image gallery |

---

## Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│                    RAG Agent Loop                        │
│                                                         │
│  1. Vector Search      (ChromaDB + MiniLM embeddings)   │
│         │                                               │
│  2. Metadata Filter    (category + emotional tone)      │
│         │                                               │
│  3. Cross-Encoder Rerank  (ms-marco-MiniLM)             │
│         │                                               │
│  4. Self-Reflection    (score threshold check)          │
│         │                                               │
│    satisfied? ──yes──► LLM Explain ──► Return results   │
│         │                                               │
│        no ──► Query Rewrite (flan-t5) ──► Retry        │
│                      (max 2 retries)                    │
└─────────────────────────────────────────────────────────┘
```

---

## Repository Structure

```
bookmind/
│
├── dataset/
│   ├── raw_dataset.csv              # Original 6,810-book dataset
│   ├── books_cleaned.csv            # After EDA cleaning (5,197 books)
│   ├── books_with_categories.csv    # After zero-shot category classification
│   └── books_with_emotions.csv      # After sentiment/emotion tagging
│
├── rag_agent/
│   ├── rag_agent.py                 # Core agent: vector search, filter, rerank, reflect, explain
│   ├── reranker.py                  # Cross-encoder reranker wrapper
│   ├── llm_local.py                 # flan-t5-base LLM wrapper + prompt templates
│   └── observability_new.py         # LangSmith setup + RunTracker
│
├── EDA.ipynb                        # Exploratory data analysis & cleaning pipeline
├── sentiment_analysis.ipynb         # Emotion scoring with BERT
├── text_classfication.ipynb         # Zero-shot Fiction/Nonfiction classification
├── vector_search.ipynb              # ChromaDB embedding & search experiments
├── gradio_dashboard.py              # Full Gradio UI
├── tagged_description.txt           # ISBN-prefixed descriptions for vector indexing
└── cover-not-found.jpg              # Fallback cover image
```

---

## Data Pipeline

The dataset goes through four sequential preprocessing stages starting from 6,810 raw books.

### 1 — EDA and Cleaning `EDA.ipynb`

- Dropped rows missing `description`, `num_pages`, `average_rating`, or `published_year`
- Removed books with fewer than 25 words in their description (eliminates stubs like *"Donation."* or *"Fantasy-roman."*)
- Merged `title` and `subtitle` into `title_and_subtitle`
- Computed `age_of_book = 2026 - published_year`
- Created `tagged_description` = `isbn13 + " " + description` (used as the vector index unit)
- **Final cleaned corpus: 5,197 books**

### 2 — Category Classification `text_classification.ipynb`

The raw dataset had **531 unique category strings**. These were mapped to simplified labels:

| Original | Simplified |
|---|---|
| Fiction | Fiction |
| Juvenile Fiction | Children's Fiction |
| Biography & Autobiography, History, Science | Nonfiction |
| Comics, Drama, Poetry | Fiction |
| Juvenile Nonfiction | Children's Nonfiction |

For the remaining **~1,454 books** with unmapped categories, `facebook/bart-large-mnli` was used in zero-shot mode. Accuracy validated at **77.8%** on a held-out set of 600 books.

### 3 — Emotion Tagging `sentiment_analysis.ipynb`

Each description was split into sentences and classified by `bhadresh-savani/bert-base-uncased-emotion` across 6 labels: `joy`, `sadness`, `anger`, `fear`, `love`, `surprise`. Scores were averaged to produce a book-level emotion vector and a `dominant_emotion` column for tone-based filtering.

### 4 — Vector Indexing `vector_search.ipynb`

`tagged_description` strings were embedded with `sentence-transformers/all-MiniLM-L6-v2` and stored in ChromaDB. Semantic search is performed at query time against this index.

---

## Agent Design

The agent uses a tool-based loop with self-reflection:

| Tool | Description |
|---|---|
| `vector_search` | Retrieves top 50 semantically similar books from ChromaDB |
| `metadata_filter` | Filters by category and sorts by emotion score; keeps top 80 |
| `rerank` | Applies cross-encoder to rerank candidates; keeps top 16 |
| `explain_books` | Calls flan-t5-base to generate a "why this book" sentence per result |

**Reflection and Retry logic:** after reranking, the agent checks whether there are at least `MIN_RESULTS = 3` books and whether the top cross-encoder score is ≥ `SCORE_THRESHOLD = -0.5`. If not, it rewrites the query with flan-t5 and retries the full pipeline up to `MAX_RETRIES = 2` times.

Every run returns:

```python
{
  "books":         pd.DataFrame,    # Final ranked books
  "explanations":  dict,            # Per-book LLM explanations
  "metrics":       dict,            # Timing, scores, step counts
  "reasoning":     str,             # Human-readable agent reasoning
  "query_history": list[str],       # Original + any rewrites
  "reflections":   list[dict],      # Per-attempt reflection decisions
}
```

---

## Models

| Model | Role |
|---|---|
| `sentence-transformers/all-MiniLM-L6-v2` | Query and document embeddings |
| `bhadresh-savani/bert-base-uncased-emotion` | Emotion classification (6 labels) |
| `facebook/bart-large-mnli` | Zero-shot Fiction/Nonfiction classification |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | Candidate reranking |
| `google/flan-t5-base` | Query rewriting and book explanations |

> All models run locally on CPU. No GPU required, though GPU will be faster for the sentiment and classification stages.

---

## Setup

**Prerequisites:** Python 3.10+, ~4GB disk space for model downloads on first run.

```bash
# Install dependencies
pip install -r requirements.txt

# Run the dashboard
python gradio_dashboard.py
```

App is available at `http://127.0.0.1:7860`. Models (~1.5GB total) download automatically on first launch.

**LangSmith tracing (optional)** — create a `.env` file:

```env
LANGCHAIN_API_KEY=your_langsmith_api_key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=bookmind-rag
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
```

If no key is provided, tracing is silently disabled and the app runs normally.

**Standalone usage:**

```python
from rag_agent.rag_agent import initialize, run_agent

books_df = initialize(
    csv_path  = "dataset/books_with_emotions.csv",
    txt_path  = "tagged_description.txt",
    llm_model = "google/flan-t5-base",
)

result = run_agent(
    query    = "a melancholic story about grief and unexpected friendship",
    category = "Fiction",
    tone     = "Sad",
)

print(result["books"][["title", "authors", "rerank_score"]].head(5))
print(result["reasoning"])
```

---

## Example Queries

| Query | Category | Tone |
|---|---|---|
| *"a melancholic wartime love story with exquisite prose"* | Fiction | Sad |
| *"books to teach children about nature and animals"* | Children's Fiction | Happy |
| *"a gripping thriller with a female detective"* | Fiction | Suspenseful |
| *"philosophy of consciousness and the nature of self"* | Nonfiction | Any |
| *"a story about redemption and second chances"* | Any | Any |

---

## Limitations and Future Work

> **Small LLM explanations** — `flan-t5-base` is a small model and explanations can sometimes be generic. A larger instruction-tuned model would improve quality.

> **Classification accuracy** — Zero-shot classification sits at ~78%. A fine-tuned classifier trained on book descriptions would improve this.

> **Reranking latency** — On CPU, reranking 80 candidates averages 2–3 seconds. Batching or a lighter cross-encoder would reduce this.

> **Future directions** — User preference memory, collaborative filtering signals, multi-turn conversation support.

---

## Acknowledgements

Dataset sourced from the Google Books API. Emotion model by [bhadresh-savani](https://huggingface.co/bhadresh-savani/bert-base-uncased-emotion). Reranker and zero-shot classifier from HuggingFace.

---

<div align="center">

MIT License · **[▶ Live Demo on Hugging Face Spaces](https://huggingface.co/spaces/nayanasisil2700/Contextual-RAG-Book-Recommender)**

</div>
