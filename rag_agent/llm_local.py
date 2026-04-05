import os
from typing import Optional, List, Any
from langchain_core.language_models.llms import LLM


def get_llm():
    
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    model_id = "google/flan-t5-base"
    print(f"[LLM] Loading HuggingFace model: {model_id} (first run downloads ~250MB)")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

    class FlanT5LLM(LLM):
        def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs: Any) -> str:
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.1,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            return response
        
        @property
        def _llm_type(self) -> str:
            return "flan-t5-base"
    
    print("[LLM] Using HuggingFace: flan-t5-base")
    return FlanT5LLM()


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