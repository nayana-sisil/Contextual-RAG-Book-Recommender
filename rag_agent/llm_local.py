"""
llm_local.py
------------
Free local LLM setup for BookMind Agentic RAG.
Primary: Ollama (llama3.2 or phi3) — runs fully on CPU, no API key.
Fallback: HuggingFace pipeline (also free, also CPU).

Install Ollama: https://ollama.com/download
Then run: ollama pull llama3.2
"""

import os
from typing import Optional

# ── Try Ollama first ──────────────────────────────────────────────────────────
def get_llm(model: str = "llama3.2", temperature: float = 0.1):
    """
    Returns a LangChain-compatible LLM.
    Tries Ollama first; falls back to HuggingFace pipeline.

    Args:
        model:       Ollama model name. Options: "llama3.2", "phi3", "mistral"
        temperature: Lower = more factual answers (0.1 recommended for RAG)
    """
    try:
        from langchain_ollama import OllamaLLM
        llm = OllamaLLM(model=model, temperature=temperature)
        # Quick ping to confirm Ollama is running
        llm.invoke("ping")
        print(f"[LLM] Using Ollama: {model}")
        return llm
    except Exception as e:
        print(f"[LLM] Ollama not available ({e}). Falling back to HuggingFace pipeline...")
        return _get_hf_llm()


def _get_hf_llm():
    """
    HuggingFace fallback — uses a small instruction-tuned model that runs on CPU.
    Downloads ~500MB on first run, cached afterwards.
    """
    from langchain_huggingface import HuggingFacePipeline
    from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

    model_id = "google/flan-t5-base"   # 250MB, fast on CPU, good for short answers
    print(f"[LLM] Loading HuggingFace model: {model_id} (first run downloads ~250MB)")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model     = AutoModelForSeq2SeqLM.from_pretrained(model_id)

    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=200,
        temperature=0.1,
        do_sample=False,
    )
    print("[LLM] Using HuggingFace: flan-t5-base")
    return HuggingFacePipeline(pipeline=pipe)


# ── Prompt templates ──────────────────────────────────────────────────────────
EXPLAIN_PROMPT = """You are a literary expert. Given a user's book request and a book's description,
write 2 sentences explaining WHY this specific book matches what the user wants.
Be specific about themes, tone, and emotional resonance. Be concise.

User's request: {query}

Book title: {title}
Book description: {description}

Why this book matches (2 sentences max):"""


QUERY_ANALYSIS_PROMPT = """Analyze this book search query and extract key information.
Return a JSON object with these fields:
- themes: list of 2-3 main themes (e.g. ["grief", "friendship", "redemption"])
- tone: dominant emotional tone from ["happy", "sad", "suspenseful", "angry", "surprising", "neutral"]
- category: "fiction", "non-fiction", or "all"
- keywords: list of 3-5 important words for vector search

Query: {query}

JSON response only, no explanation:"""


if __name__ == "__main__":
    llm = get_llm()
    result = llm.invoke("In one sentence, what is a book about grief?")
    print("LLM test:", result)