"""
gradio_dashboard_v2.py
----------------------
BookMind Agentic RAG — Gradio UI
Matches the editorial book-shop design from the mockup.

Run:
    python gradio_dashboard_v2.py

First run will download embedding models (~90MB).
If Ollama is installed and llama3.2 is pulled, the LLM explain feature is live.
Otherwise it gracefully falls back to HuggingFace flan-t5-base.
"""

import json
import math
import pandas as pd
import gradio as gr

from rag_agent import initialize, run_agent

# ── Startup ───────────────────────────────────────────────────────────────────
print("Starting BookMind Agentic RAG...")
books_df = initialize(
    csv_path  = "../dataset/books_with_emotions.csv",
    txt_path  = "../tagged_description.txt",
    llm_model = "google/flan-t5-base",   # ← correct HuggingFace model ID
)

CATEGORIES = ["All"] + sorted(books_df["simple_categories"].dropna().unique().tolist())
TONES      = ["All", "Happy", "Surprising", "Angry", "Suspenseful", "Sad"]


# ── CSS — editorial warm palette ──────────────────────────────────────────────
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,700;1,400&family=DM+Sans:wght@300;400;500&display=swap');

:root {
    --ink:     #1a1208;
    --paper:   #f5f0e8;
    --cream:   #ede8dc;
    --warm:    #c8a96e;
    --rust:    #b85c38;
    --sage:    #5a7a5c;
    --dim:     #7a6f60;
    --card:    #faf7f2;
    --border:  rgba(26,18,8,0.12);
}

body, .gradio-container {
    font-family: 'DM Sans', sans-serif !important;
    background: var(--paper) !important;
    color: var(--ink) !important;
}

/* Header */
#bookmind-header {
    background: var(--ink);
    padding: 18px 32px;
    border-bottom: 3px solid var(--warm);
    margin-bottom: 0;
}
#bookmind-header h1 {
    font-family: 'Playfair Display', serif !important;
    font-size: 22px !important;
    color: var(--paper) !important;
    font-weight: 400 !important;
    margin: 0 !important;
}
#bookmind-header h1 em { color: var(--warm); }
#agentic-badge {
    font-size: 10px;
    font-weight: 500;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: rgba(255,255,255,0.35);
    border: 1px solid rgba(255,255,255,0.15);
    padding: 2px 8px;
    border-radius: 2px;
    display: inline-block;
    margin-left: 12px;
}

/* Search bar */
#search-area {
    background: var(--cream);
    padding: 20px 32px;
    border-bottom: 1px solid var(--border);
}
#query-input textarea {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 15px !important;
    background: var(--card) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: 3px !important;
    color: var(--ink) !important;
    padding: 12px 18px !important;
}
#query-input textarea:focus { border-color: var(--warm) !important; }

#category-dd, #tone-dd {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 13px !important;
}
#find-btn {
    background: var(--rust) !important;
    color: white !important;
    border: none !important;
    border-radius: 3px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    font-size: 14px !important;
}
#find-btn:hover { background: #a04d2e !important; }

/* Trace bar */
#trace-bar {
    background: var(--ink);
    padding: 8px 32px;
    font-family: 'Courier New', monospace;
    font-size: 11px;
    color: rgba(255,255,255,0.45);
    border-bottom: 1px solid rgba(200,169,110,0.2);
    min-height: 36px;
}

/* Sidebar */
#sidebar {
    background: var(--cream);
    border-right: 1px solid var(--border);
    padding: 20px 16px;
    min-height: 600px;
}
#reasoning-box {
    background: var(--card);
    border: 1px solid var(--border);
    border-left: 3px solid var(--warm);
    border-radius: 4px;
    padding: 12px;
    margin-bottom: 14px;
    font-family: 'Playfair Display', serif !important;
    font-size: 13px !important;
    font-style: italic;
    color: var(--dim) !important;
    line-height: 1.65;
}
#metrics-box {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 12px !important;
    color: var(--dim) !important;
    background: transparent !important;
    border: none !important;
}
#tools-box {
    font-size: 11px !important;
    font-family: 'DM Sans', sans-serif !important;
    color: var(--dim) !important;
    background: transparent !important;
}
#langsmith-box {
    font-size: 11px !important;
    color: var(--sage) !important;
    background: rgba(90,122,92,0.06) !important;
    border: 1px solid rgba(90,122,92,0.2) !important;
    border-radius: 3px !important;
    padding: 6px 10px !important;
}

/* Book gallery */
#book-gallery {
    padding: 0 !important;
}
#book-gallery .gallery-item {
    border-radius: 4px !important;
    overflow: hidden !important;
    border: 1px solid var(--border) !important;
    background: var(--card) !important;
    transition: transform 0.2s, box-shadow 0.2s !important;
}
#book-gallery .gallery-item:hover {
    transform: translateY(-4px) !important;
    box-shadow: 0 8px 24px rgba(26,18,8,0.1) !important;
}
#book-gallery .caption {
    font-family: 'Playfair Display', serif !important;
    font-size: 11px !important;
    color: var(--ink) !important;
    padding: 6px 8px !important;
    background: var(--card) !important;
    line-height: 1.4 !important;
}

/* Results header */
#results-header {
    font-family: 'Playfair Display', serif !important;
    font-size: 18px !important;
    color: var(--ink) !important;
    font-weight: 700 !important;
    margin-bottom: 4px !important;
}
#results-meta {
    font-size: 12px !important;
    color: var(--dim) !important;
}

/* Sidebar labels */
.sidebar-label {
    font-size: 10px;
    font-weight: 500;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--dim);
    margin-bottom: 10px;
    margin-top: 16px;
}
"""


# ── Helper: build trace bar HTML ──────────────────────────────────────────────
def build_trace_html(metrics: dict) -> str:
    steps_done = {s["step"] for s in metrics.get("steps", [])}
    all_steps  = ["vector-search", "metadata-filter", "cross-encoder-rerank", "explain-books"]
    step_labels = {
        "vector-search":        "Vector search",
        "metadata-filter":      "Metadata filter",
        "cross-encoder-rerank": "Cross-encoder rerank",
        "explain-books":        "LLM explain",
    }

    parts = []
    for step in all_steps:
        cls    = "done" if step in steps_done else "pending"
        label  = step_labels.get(step, step)
        color  = "#7ab87d" if cls == "done" else "rgba(255,255,255,0.25)"
        bg     = "rgba(90,122,92,0.12)" if cls == "done" else "rgba(255,255,255,0.04)"
        border = "rgba(90,122,92,0.4)"  if cls == "done" else "rgba(255,255,255,0.08)"
        dot    = f'<span style="width:6px;height:6px;border-radius:50%;background:{color};display:inline-block;margin-right:5px;flex-shrink:0"></span>'
        parts.append(
            f'<span style="display:inline-flex;align-items:center;padding:3px 9px;'
            f'border-radius:2px;background:{bg};border:1px solid {border};'
            f'color:{color};font-size:11px;white-space:nowrap">{dot}{label}</span>'
        )

    arrow = '<span style="color:rgba(255,255,255,0.2);font-size:12px;margin:0 2px">›</span>'
    total = metrics.get("total_s", 0)
    time_badge = (
        f'<span style="margin-left:auto;color:rgba(255,255,255,0.3);font-size:11px">'
        f'{total}s</span>'
    )

    inner = arrow.join(parts) + time_badge
    return (
        f'<div style="background:#1a1208;padding:9px 32px;display:flex;align-items:center;'
        f'gap:6px;border-bottom:1px solid rgba(200,169,110,0.2);overflow-x:auto;'
        f'white-space:nowrap;font-family:monospace">'
        f'<span style="font-size:10px;letter-spacing:1px;color:rgba(255,255,255,0.25);'
        f'text-transform:uppercase;margin-right:8px;flex-shrink:0">Agent trace</span>'
        f'{inner}</div>'
    )


# ── Helper: build sidebar metrics HTML ────────────────────────────────────────
def build_metrics_html(metrics: dict) -> str:
    rows = [
        ("Candidates retrieved", metrics.get("candidates",   0), ""),
        ("After filter",         metrics.get("after_filter", 0), ""),
        ("After rerank",         metrics.get("after_rerank", 0), "green"),
        ("Top score",            metrics.get("top_score",    0.0), "amber"),
        ("LLM calls",            metrics.get("llm_calls",    0), ""),
        ("Total time",           f'{metrics.get("total_s", 0)}s', ""),
    ]
    html = '<div style="font-size:12px;font-family:DM Sans,sans-serif;">'
    colors = {"green": "#5a7a5c", "amber": "#b8860b", "": "#1a1208"}
    for label, val, cls in rows:
        col = colors[cls]
        html += (
            f'<div style="display:flex;justify-content:space-between;'
            f'padding:7px 0;border-bottom:1px solid rgba(26,18,8,0.1)">'
            f'<span style="color:#7a6f60">{label}</span>'
            f'<span style="font-weight:500;color:{col}">{val}</span></div>'
        )
    html += "</div>"
    return html


def build_tools_html(tools: list) -> str:
    color_map = {
        "vector-search":        ("#2a6496", "rgba(42,100,150,0.06)"),
        "metadata-filter":      ("#8b4513", "rgba(139,69,19,0.06)"),
        "cross-encoder-rerank": ("#5a7a5c", "rgba(90,122,92,0.06)"),
        "explain-books":        ("#6a1a6a", "rgba(106,26,106,0.06)"),
    }
    label_map = {
        "vector-search":        "vector-search",
        "metadata-filter":      "metadata-filter",
        "cross-encoder-rerank": "cross-encoder",
        "explain-books":        "llm-explain",
    }
    chips = ""
    for t in tools:
        fg, bg = color_map.get(t, ("#555", "rgba(0,0,0,0.04)"))
        label  = label_map.get(t, t)
        chips += (
            f'<span style="display:inline-block;font-size:10px;font-weight:500;'
            f'letter-spacing:0.5px;padding:3px 8px;border-radius:2px;margin:3px 3px 0 0;'
            f'border:1px solid {fg};color:{fg};background:{bg}">{label}</span>'
        )
    return f'<div style="margin-top:4px">{chips}</div>'


# ── Main recommendation function ──────────────────────────────────────────────
def recommend(query: str, category: str, tone: str):
    """Called by Gradio on button click. Yields progressive updates."""

    if not query.strip():
        yield (
            "<div style='padding:20px;color:#7a6f60;font-style:italic'>"
            "Please enter a description of the book you're looking for.</div>",
            [],
            "<div style='font-style:italic;color:#7a6f60;font-size:13px'>Waiting for query...</div>",
            "",
            "",
            "",
        )
        return

    # Show "thinking" state
    thinking_trace = build_trace_html({"steps": [], "total_s": "..."})
    yield (
        thinking_trace,
        [],
        "<div style='font-style:italic;color:#7a6f60;font-size:13px;font-family:Playfair Display,serif'>"
        "Analysing your query...</div>",
        "",
        "",
        "",
    )

    # Run agentic pipeline
    result      = run_agent(query, category, tone)
    final_df    = result["books"]
    explanations = result["explanations"]
    metrics     = result["metrics"]
    reasoning   = result["reasoning"]

    # Build gallery items
    gallery = []
    for _, row in final_df.iterrows():
        title   = row.get("title",   "Unknown")
        authors = row.get("authors", "")
        desc    = str(row.get("description", ""))
        score   = row.get("rerank_score", 0)
        explain = explanations.get(title, "")

        # Truncate description for caption
        desc_short = " ".join(desc.split()[:20]) + "..." if len(desc.split()) > 20 else desc

        # Format authors
        auth_parts = authors.split(";") if authors else []
        if len(auth_parts) == 2:
            auth_str = f"{auth_parts[0]} & {auth_parts[1]}"
        elif len(auth_parts) > 2:
            auth_str = f"{auth_parts[0]} et al."
        else:
            auth_str = authors

        # Score badge in caption
        score_str = f"[{score:.2f}] " if score else ""
        caption   = f"{score_str}{title} — {auth_str}\n{desc_short}"
        if explain:
            caption += f"\n✦ {explain[:100]}..."

        thumb = row.get("large_thumbnail", "cover-not-found.jpg")
        gallery.append((thumb, caption))

    # Build sidebar content
    trace_html   = build_trace_html(metrics)
    metrics_html = build_metrics_html(metrics)
    tools_html   = build_tools_html(metrics.get("tools_called", []))

    reasoning_short = reasoning[:200] + "..." if len(reasoning) > 200 else reasoning
    if not reasoning_short:
        reasoning_short = (
            f"Searched {metrics.get('candidates',0)} books, "
            f"filtered to {metrics.get('after_filter',0)}, "
            f"reranked to {metrics.get('after_rerank',0)} final recommendations."
        )

    langsmith_html = (
        '<div style="display:flex;align-items:center;gap:6px;margin-top:14px;'
        'padding:7px 10px;background:rgba(90,122,92,0.06);border:1px solid rgba(90,122,92,0.2);'
        'border-radius:3px;font-size:11px;color:#5a7a5c">'
        '<span style="width:7px;height:7px;border-radius:50%;background:#5a7a5c;flex-shrink:0"></span>'
        'LangSmith trace logged</div>'
    )

    n = len(final_df)
    results_header = (
        f'<div style="font-family:Playfair Display,serif;font-size:20px;'
        f'font-weight:700;color:#1a1208;margin-bottom:2px">Recommendations</div>'
        f'<div style="font-size:12px;color:#7a6f60">{n} books · ranked by relevance</div>'
    )

    yield (
        trace_html,
        gallery,
        f'<div style="font-family:Playfair Display,serif;font-size:13px;'
        f'font-style:italic;color:#7a6f60;line-height:1.65">{reasoning_short}</div>',
        metrics_html,
        tools_html + langsmith_html,
        results_header,
    )


# ── Gradio layout ─────────────────────────────────────────────────────────────
with gr.Blocks(css=CSS, theme=gr.themes.Base(), title="BookMind — Agentic RAG") as app:

    # Header
    gr.HTML("""
    <div id="bookmind-header">
      <h1>Book<em>Mind</em><span id="agentic-badge">Agentic RAG</span></h1>
    </div>
    """)

    # Search row
    with gr.Group(elem_id="search-area"):
        with gr.Row():
            query_input = gr.Textbox(
                label       = "Describe the story you're looking for...",
                placeholder = "e.g. A story about grief and finding unexpected friendship in a small town",
                lines       = 1,
                scale       = 5,
                elem_id     = "query-input",
            )
            category_dd = gr.Dropdown(
                choices  = CATEGORIES,
                value    = "All",
                label    = "Category",
                scale    = 1,
                elem_id  = "category-dd",
            )
            tone_dd = gr.Dropdown(
                choices  = TONES,
                value    = "All",
                label    = "Emotional tone",
                scale    = 1,
                elem_id  = "tone-dd",
            )
            find_btn = gr.Button("Find books", variant="primary", scale=1, elem_id="find-btn")

    # Agent trace bar
    trace_bar = gr.HTML(
        value=(
            '<div style="background:#1a1208;padding:9px 32px;font-family:monospace;'
            'font-size:11px;color:rgba(255,255,255,0.25);border-bottom:1px solid '
            'rgba(200,169,110,0.2)">Agent trace will appear here after search...</div>'
        )
    )

    # Main content: sidebar + book grid
    with gr.Row():

        # Sidebar
        with gr.Column(scale=1, elem_id="sidebar"):
            gr.HTML('<div class="sidebar-label">Agent reasoning</div>')
            reasoning_box = gr.HTML(
                value='<div style="font-family:Playfair Display,serif;font-size:13px;'
                      'font-style:italic;color:#7a6f60">Run a search to see agent reasoning...</div>',
                elem_id="reasoning-box",
            )
            gr.HTML('<div class="sidebar-label">Run metrics</div>')
            metrics_box = gr.HTML(elem_id="metrics-box")
            gr.HTML('<div class="sidebar-label">Tools called</div>')
            tools_box = gr.HTML(elem_id="tools-box")

        # Book grid
        with gr.Column(scale=4):
            results_header = gr.HTML(
                value='<div style="font-family:Playfair Display,serif;font-size:20px;'
                      'font-weight:700;color:#1a1208;margin-bottom:2px">Recommendations</div>',
            )
            book_gallery = gr.Gallery(
                label   = "",
                columns = 4,
                rows    = 4,
                height  = "auto",
                elem_id = "book-gallery",
                show_label = False,
                object_fit = "cover",
            )

    # Wire up
    find_btn.click(
        fn      = recommend,
        inputs  = [query_input, category_dd, tone_dd],
        outputs = [trace_bar, book_gallery, reasoning_box, metrics_box, tools_box, results_header],
    )
    query_input.submit(
        fn      = recommend,
        inputs  = [query_input, category_dd, tone_dd],
        outputs = [trace_bar, book_gallery, reasoning_box, metrics_box, tools_box, results_header],
    )


if __name__ == "__main__":
    app.launch(
        server_name = "0.0.0.0",
        server_port = 7860,
        share       = False,    # set True to get a public gradio.live link
    )