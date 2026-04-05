import os
import time
import functools
from typing import Any, Callable, Optional
from dotenv import load_dotenv

load_dotenv(".env")


def setup_langsmith(project: str = "bookmind-rag") -> bool:
    api_key = os.getenv("LANGCHAIN_API_KEY", "")
    if not api_key:
        print("[Observability] API key missing — tracing disabled.")
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        return False

    os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2", "true")
    os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", project)
    os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")

    print(f"[Observability] LangSmith tracing ON → project: '{project}'")
    print("                View at: https://smith.langchain.com")
    return True


class RunTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.start_time = time.time()
        self.steps = []
        self.tools_called = []
        self.candidates = 0
        self.after_filter = 0
        self.after_rerank = 0
        self.top_score = 0.0
        self.llm_calls = 0
        self.query = ""
        self.reasoning = ""

    def log_step(self, name: str, status: str = "done"):
        elapsed = int((time.time() - self.start_time) * 1000)
        self.steps.append({"step": name, "duration_ms": elapsed, "status": status})
        self.tools_called.append(name)
        print(f"  [{status.upper()}] {name} — {elapsed}ms total")

    def total_seconds(self) -> float:
        return round(time.time() - self.start_time, 2)

    def to_dict(self) -> dict:
        return {
            "steps": self.steps,
            "tools_called": list(dict.fromkeys(self.tools_called)),
            "candidates": self.candidates,
            "after_filter": self.after_filter,
            "after_rerank": self.after_rerank,
            "top_score": round(self.top_score, 3),
            "llm_calls": self.llm_calls,
            "total_s": self.total_seconds(),
            "query": self.query,
            "reasoning": self.reasoning,
        }

    def summary(self) -> str:
        d = self.to_dict()
        return (
            f"Run complete in {d['total_s']}s | "
            f"candidates: {d['candidates']} → filter: {d['after_filter']} "
            f"→ rerank: {d['after_rerank']} | top score: {d['top_score']}"
        )


def timed(tracker: RunTracker, step_name: str):
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            t0 = time.time()
            result = fn(*args, **kwargs)
            ms = int((time.time() - t0) * 1000)
            tracker.steps.append({"step": step_name, "duration_ms": ms, "status": "done"})
            print(f"  [DONE] {step_name} — {ms}ms")
            return result
        return wrapper
    return decorator


if __name__ == "__main__":
    setup_langsmith()
    tracker = RunTracker()
    tracker.query = "test query"
    tracker.log_step("vector-search")
    tracker.log_step("rerank")
    print(tracker.summary())
    print(tracker.to_dict())