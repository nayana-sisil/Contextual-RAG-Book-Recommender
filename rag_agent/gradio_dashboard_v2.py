"""
gradio_dashboard_v2.py
----------------------
BookMind Agentic RAG — Gradio UI
Editorial dark-ink aesthetic with working LLM explanations.

Run:
    python gradio_dashboard_v2.py
"""

import pandas as pd
import gradio as gr

from rag_agent import initialize, run_agent

# ── Startup ───────────────────────────────────────────────────────────────────
print("Starting BookMind Agentic RAG...")
books_df = initialize(
    csv_path  = "../dataset/books_with_emotions.csv",
    txt_path  = "../tagged_description.txt",
    llm_model = "google/flan-t5-base",
)

CATEGORIES = ["All"] + sorted(books_df["simple_categories"].dropna().unique().tolist())
TONES      = ["All", "Happy", "Surprising", "Angry", "Suspenseful", "Sad"]


# ── CSS ───────────────────────────────────────────────────────────────────────
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Libre+Baskerville:ital,wght@0,400;0,700;1,400&family=DM+Mono:wght@300;400&family=Outfit:wght@300;400;500;600&display=swap');

:root {
    --ink:       #0e0c09;
    --paper:     #f7f3ec;
    --cream:     #eee9df;
    --gold:      #c9a84c;
    --gold-dim:  #8a6e2f;
    --rust:      #a84232;
    --rust-h:    #8a3228;
    --sage:      #4a6741;
    --dim:       #6b6358;
    --card:      #faf8f4;
    --border:    rgba(14,12,9,0.1);
    --shadow:    0 2px 16px rgba(14,12,9,0.08);
}

*, *::before, *::after { box-sizing: border-box; }

body, .gradio-container, .gradio-container * {
    font-family: 'Outfit', sans-serif !important;
}

/* ── Remove Gradio chrome ── */
.gradio-container {
    background: var(--paper) !important;
    max-width: 100% !important;
    padding: 0 !important;
    margin: 0 !important;
}
footer { display: none !important; }
.contain { padding: 0 !important; }

/* ── Header ── */
#bm-header {
    background: var(--ink);
    padding: 0 40px;
    height: 56px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    border-bottom: 2px solid var(--gold);
    position: sticky;
    top: 0;
    z-index: 100;
}
.bm-logo {
    font-family: 'Libre Baskerville', serif !important;
    font-size: 20px;
    color: var(--paper);
    letter-spacing: -0.3px;
}
.bm-logo em { color: var(--gold); font-style: italic; }
.bm-badge {
    font-family: 'DM Mono', monospace !important;
    font-size: 9px;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--gold-dim);
    border: 1px solid var(--gold-dim);
    padding: 2px 7px;
    border-radius: 2px;
    margin-left: 10px;
}
.bm-status {
    font-family: 'DM Mono', monospace !important;
    font-size: 10px;
    color: rgba(255,255,255,0.3);
    display: flex;
    align-items: center;
    gap: 6px;
}
.bm-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: var(--sage);
    box-shadow: 0 0 6px var(--sage);
}

/* ── Search strip ── */
#bm-search {
    background: var(--cream);
    padding: 16px 40px;
    border-bottom: 1px solid var(--border);
    display: flex;
    gap: 12px;
    align-items: flex-end;
}

/* Override Gradio input styles */
#bm-query textarea,
#bm-query input {
    font-family: 'Libre Baskerville', serif !important;
    font-size: 14px !important;
    background: var(--card) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: 4px !important;
    color: var(--ink) !important;
    padding: 11px 16px !important;
    box-shadow: var(--shadow) !important;
    transition: border-color 0.2s !important;
}
#bm-query textarea:focus,
#bm-query input:focus {
    border-color: var(--gold) !important;
    outline: none !important;
}
#bm-query label span,
#bm-cat label span,
#bm-tone label span {
    font-family: 'DM Mono', monospace !important;
    font-size: 9px !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
    color: var(--dim) !important;
    margin-bottom: 5px !important;
}
#bm-cat select, #bm-tone select,
#bm-cat .wrap, #bm-tone .wrap {
    font-family: 'Outfit', sans-serif !important;
    font-size: 13px !important;
    background: var(--card) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: 4px !important;
    color: var(--ink) !important;
}
#bm-btn {
    background: var(--rust) !important;
    color: white !important;
    border: none !important;
    border-radius: 4px !important;
    font-family: 'Outfit', sans-serif !important;
    font-weight: 500 !important;
    font-size: 13px !important;
    letter-spacing: 0.3px !important;
    padding: 11px 22px !important;
    height: 44px !important;
    cursor: pointer !important;
    transition: background 0.15s !important;
    white-space: nowrap !important;
}
#bm-btn:hover { background: var(--rust-h) !important; }

/* ── Agent trace bar ── */
#bm-trace {
    background: #0a0805;
    padding: 0 40px;
    min-height: 40px;
    border-bottom: 1px solid rgba(201,168,76,0.15);
}

/* ── Main layout ── */
#bm-main {
    display: flex;
    min-height: calc(100vh - 160px);
}

/* ── Sidebar ── */
#bm-sidebar {
    width: 260px;
    flex-shrink: 0;
    background: var(--cream);
    border-right: 1px solid var(--border);
    padding: 20px 18px;
}
.bm-sidebar-section {
    margin-bottom: 20px;
}
.bm-sidebar-label {
    font-family: 'DM Mono', monospace !important;
    font-size: 9px;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--dim);
    margin-bottom: 8px;
    padding-bottom: 6px;
    border-bottom: 1px solid var(--border);
}
#bm-reasoning {
    font-family: 'Libre Baskerville', serif !important;
    font-size: 12px !important;
    font-style: italic !important;
    color: var(--dim) !important;
    line-height: 1.7 !important;
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
}
#bm-metrics {
    font-size: 12px !important;
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
}
#bm-tools {
    font-size: 11px !important;
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
}
#bm-rewrites {
    font-size: 11px !important;
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
}

/* ── Book grid ── */
#bm-grid {
    flex: 1;
    padding: 24px 32px;
    background: var(--paper);
}
#bm-results-hdr {
    margin-bottom: 18px;
}

/* Gallery overrides */
#bm-gallery {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
}
#bm-gallery .grid-wrap {
    gap: 16px !important;
}
#bm-gallery .thumbnail-item {
    border-radius: 6px !important;
    overflow: hidden !important;
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    box-shadow: var(--shadow) !important;
    transition: transform 0.2s ease, box-shadow 0.2s ease !important;
    cursor: pointer !important;
}
#bm-gallery .thumbnail-item:hover {
    transform: translateY(-5px) !important;
    box-shadow: 0 12px 32px rgba(14,12,9,0.14) !important;
}
#bm-gallery img {
    width: 100% !important;
    object-fit: cover !important;
    aspect-ratio: 2/3 !important;
    display: block !important;
}
#bm-gallery .caption-label {
    font-family: 'Outfit', sans-serif !important;
    font-size: 11px !important;
    padding: 8px 10px !important;
    color: var(--ink) !important;
    background: var(--card) !important;
    line-height: 1.45 !important;
}

/* ── Book detail panel ── */
#bm-detail {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    padding: 0 !important;
    margin-top: 0 !important;
    box-shadow: var(--shadow) !important;
    font-size: 13px !important;
}

/* ── No results ── */
.bm-empty {
    text-align: center;
    padding: 60px 20px;
    color: var(--dim);
    font-family: 'Libre Baskerville', serif;
    font-style: italic;
    font-size: 15px;
}
"""


# ── HTML builders ─────────────────────────────────────────────────────────────

def build_trace_html(metrics: dict, query_history: list = None) -> str:
    steps_done = {s["step"] for s in metrics.get("steps", [])}
    all_steps  = [
        ("vector-search",        "① Vector"),
        ("metadata-filter",      "② Filter"),
        ("cross-encoder-rerank", "③ Rerank"),
        ("explain-books",        "④ Explain"),
    ]
    parts = []
    for step_id, label in all_steps:
        done   = step_id in steps_done
        color  = "#7ab87d" if done else "rgba(255,255,255,0.2)"
        bg     = "rgba(74,103,65,0.15)" if done else "transparent"
        border = "rgba(74,103,65,0.35)" if done else "rgba(255,255,255,0.07)"
        parts.append(
            f'<span style="display:inline-flex;align-items:center;gap:5px;'
            f'padding:4px 10px;border-radius:3px;background:{bg};'
            f'border:1px solid {border};color:{color};font-size:10px;'
            f'font-family:DM Mono,monospace;white-space:nowrap">'
            f'{"✓ " if done else ""}{label}</span>'
        )

    # Rewrite badge
    rewrites = ""
    if query_history and len(query_history) > 1:
        rewrites = (
            f'<span style="margin-left:12px;font-size:10px;font-family:DM Mono,monospace;'
            f'color:#c9a84c;border:1px solid rgba(201,168,76,0.3);'
            f'padding:3px 8px;border-radius:3px;background:rgba(201,168,76,0.06)">'
            f'↻ {len(query_history)-1} rewrite{"s" if len(query_history)>2 else ""}</span>'
        )

    sep   = '<span style="color:rgba(255,255,255,0.12);margin:0 3px;font-size:12px">›</span>'
    total = metrics.get("total_s", 0)
    timer = (
        f'<span style="margin-left:auto;font-family:DM Mono,monospace;'
        f'font-size:10px;color:rgba(255,255,255,0.25)">{total}s</span>'
    )
    label_html = (
        '<span style="font-family:DM Mono,monospace;font-size:9px;letter-spacing:1.5px;'
        'text-transform:uppercase;color:rgba(255,255,255,0.2);margin-right:10px;flex-shrink:0">'
        'Trace</span>'
    )
    return (
        f'<div style="background:#0a0805;padding:9px 40px;display:flex;align-items:center;'
        f'gap:4px;border-bottom:1px solid rgba(201,168,76,0.12);overflow-x:auto">'
        f'{label_html}{sep.join(parts)}{rewrites}{timer}</div>'
    )


def build_metrics_html(metrics: dict) -> str:
    top_score = metrics.get("top_score", 0.0)
    score_color = "#4a6741" if top_score >= 1.0 else ("#c9a84c" if top_score >= 0 else "#a84232")
    rows = [
        ("Candidates",  metrics.get("candidates",   0),  "#0e0c09"),
        ("After filter",metrics.get("after_filter", 0),  "#0e0c09"),
        ("After rerank",metrics.get("after_rerank", 0),  "#4a6741"),
        ("Top score",   f"{top_score:.3f}",               score_color),
        ("LLM calls",   metrics.get("llm_calls",    0),  "#0e0c09"),
        ("Time",        f'{metrics.get("total_s",0)}s',  "#0e0c09"),
    ]
    html = '<div style="font-size:12px;font-family:Outfit,sans-serif">'
    for label, val, color in rows:
        html += (
            f'<div style="display:flex;justify-content:space-between;align-items:center;'
            f'padding:6px 0;border-bottom:1px solid rgba(14,12,9,0.07)">'
            f'<span style="color:#6b6358;font-size:11px">{label}</span>'
            f'<span style="font-weight:500;color:{color};font-family:DM Mono,monospace;font-size:11px">{val}</span>'
            f'</div>'
        )
    html += "</div>"
    return html


def build_tools_html(tools_called: list) -> str:
    config = {
        "vector-search":        ("#2a6496", "Vector Search"),
        "metadata-filter":      ("#7a4a1a", "Metadata Filter"),
        "cross-encoder-rerank": ("#4a6741", "Cross-Encoder"),
        "explain-books":        ("#6a2a6a", "LLM Explain"),
    }
    chips = ""
    for t in tools_called:
        color, label = config.get(t, ("#555", t))
        chips += (
            f'<span style="display:inline-flex;align-items:center;gap:4px;'
            f'font-size:10px;font-family:DM Mono,monospace;padding:3px 8px;'
            f'border-radius:3px;margin:2px 3px 2px 0;border:1px solid {color}22;'
            f'color:{color};background:{color}0d">'
            f'<span style="width:5px;height:5px;border-radius:50%;background:{color};flex-shrink:0"></span>'
            f'{label}</span>'
        )
    
    # Fixed line - removed backslash from f-string expression
    if chips:
        return f'<div style="line-height:2">{chips}</div>'
    else:
        return '<div style="line-height:2"><span style="color:#6b6358;font-size:11px">None yet</span></div>'


def build_rewrites_html(query_history: list, reflections: list) -> str:
    if not query_history or len(query_history) == 1:
        return '<span style="color:#6b6358;font-size:11px;font-family:Outfit,sans-serif">No rewrites needed</span>'

    html = '<div style="font-size:11px;font-family:Outfit,sans-serif">'
    for i, q in enumerate(query_history):
        icon  = "○" if i == 0 else "↻"
        color = "#6b6358" if i == 0 else "#c9a84c"
        label = "Original" if i == 0 else f"Rewrite {i}"
        # Find reflection for this attempt
        ref   = next((r for r in reflections if r["attempt"] == i + 1), {})
        score = ref.get("top_score", None)
        score_str = f'<span style="font-family:DM Mono,monospace;color:{color}"> [{score:.2f}]</span>' if score is not None else ""
        html += (
            f'<div style="padding:5px 0;border-bottom:1px solid rgba(14,12,9,0.07)">'
            f'<div style="color:{color};font-weight:500">{icon} {label}{score_str}</div>'
            f'<div style="color:#6b6358;margin-top:2px;font-style:italic;line-height:1.4">'
            f'"{q[:60]}{"..." if len(q)>60 else ""}"</div>'
            f'</div>'
        )
    html += "</div>"
    return html


def build_detail_html(row: pd.Series, explanation: str) -> str:
    title   = row.get("title", "")
    authors = str(row.get("authors", ""))
    desc    = str(row.get("description", ""))
    score   = row.get("rerank_score", 0)
    cat     = row.get("simple_categories", "")
    rating  = row.get("average_rating", "")

    auth_parts = authors.split(";") if authors else []
    auth_str = auth_parts[0] if auth_parts else ""
    if len(auth_parts) == 2:
        auth_str = f"{auth_parts[0]} & {auth_parts[1]}"
    elif len(auth_parts) > 2:
        auth_str = f"{auth_parts[0]} et al."

    score_color = "#4a6741" if score >= 1.0 else ("#c9a84c" if score >= 0 else "#a84232")

    # Explanation block — only show if non-empty and not echoing
    explain_html = ""
    if explanation and len(explanation) > 10:
        explain_html = (
            f'<div style="margin:14px 0 0;padding:12px 14px;background:#f0ece3;'
            f'border-left:3px solid #c9a84c;border-radius:3px">'
            f'<div style="font-family:DM Mono,monospace;font-size:9px;letter-spacing:1.5px;'
            f'text-transform:uppercase;color:#8a6e2f;margin-bottom:6px">Why this book</div>'
            f'<div style="font-family:Libre Baskerville,serif;font-size:12px;font-style:italic;'
            f'color:#3a3028;line-height:1.65">{explanation}</div>'
            f'</div>'
        )

    meta_chips = ""
    if cat:
        meta_chips += f'<span style="font-size:10px;padding:2px 8px;border-radius:3px;background:#eee9df;color:#6b6358;border:1px solid rgba(14,12,9,0.1);font-family:Outfit,sans-serif">{cat}</span> '
    if rating:
        meta_chips += f'<span style="font-size:10px;padding:2px 8px;border-radius:3px;background:#f0ece3;color:#8a6e2f;border:1px solid rgba(201,168,76,0.2);font-family:DM Mono,monospace">★ {rating}</span> '

    score_badge = (
        f'<span style="font-family:DM Mono,monospace;font-size:10px;'
        f'padding:2px 8px;border-radius:3px;background:{score_color}15;'
        f'color:{score_color};border:1px solid {score_color}33">score {score:.3f}</span>'
    )

    desc_short = " ".join(desc.split()[:60]) + ("..." if len(desc.split()) > 60 else "")

    return f"""
    <div style="padding:20px 24px;font-family:Outfit,sans-serif">
        <div style="font-family:Libre Baskerville,serif;font-size:18px;font-weight:700;
             color:#0e0c09;line-height:1.3;margin-bottom:4px">{title}</div>
        <div style="font-size:12px;color:#6b6358;margin-bottom:10px;font-style:italic">
             {auth_str}</div>
        <div style="display:flex;flex-wrap:wrap;gap:5px;margin-bottom:12px;align-items:center">
             {meta_chips}{score_badge}
        </div>
        <div style="font-size:13px;color:#3a3028;line-height:1.7;border-top:1px solid rgba(14,12,9,0.08);
             padding-top:12px">{desc_short}</div>
        {explain_html}
    </div>
    """


def build_results_header(n: int, query: str) -> str:
    if n == 0:
        return (
            '<div style="font-family:Libre Baskerville,serif;font-size:18px;'
            'color:#a84232;margin-bottom:16px">No results found</div>'
            '<div style="font-size:12px;color:#6b6358">Try selecting "All" for category and tone, '
            'or use a broader description.</div>'
        )
    return (
        f'<div style="display:flex;align-items:baseline;gap:10px;margin-bottom:4px">'
        f'<span style="font-family:Libre Baskerville,serif;font-size:20px;font-weight:700;color:#0e0c09">'
        f'Recommendations</span>'
        f'<span style="font-family:DM Mono,monospace;font-size:11px;color:#6b6358">'
        f'{n} books</span>'
        f'</div>'
        f'<div style="font-size:12px;color:#6b6358;font-style:italic;margin-bottom:16px">'
        f'"{query[:80]}{"..." if len(query)>80 else ""}"</div>'
    )


# ── Main recommendation function ──────────────────────────────────────────────

def recommend(query: str, category: str, tone: str):
    if not query.strip():
        yield (
            build_trace_html({"steps": [], "total_s": 0}),
            [],
            "<span style='color:#6b6358;font-style:italic;font-size:12px'>Enter a query to begin.</span>",
            build_metrics_html({}),
            build_tools_html([]),
            build_rewrites_html([], []),
            "<div class='bm-empty'>Your recommendations will appear here.</div>",
            "",
        )
        return

    # ── Thinking state ────────────────────────────────────────────────────────
    yield (
        build_trace_html({"steps": [], "total_s": "..."}),
        [],
        "<span style='font-family:Libre Baskerville,serif;font-size:12px;font-style:italic;"
        "color:#6b6358'>Analysing your query...</span>",
        build_metrics_html({}),
        build_tools_html([]),
        build_rewrites_html([], []),
        "<div style='font-family:Libre Baskerville,serif;font-size:14px;font-style:italic;"
        "color:#6b6358;padding:40px 0;text-align:center'>Searching through 5,197 books...</div>",
        "",
    )

    # ── Run pipeline ──────────────────────────────────────────────────────────
    result        = run_agent(query, category, tone)
    final_df      = result["books"]
    explanations  = result["explanations"]
    metrics       = result["metrics"]
    reasoning     = result["reasoning"]
    query_history = result.get("query_history", [query])
    reflections   = result.get("reflections",   [])

    # ── Build gallery ─────────────────────────────────────────────────────────
    gallery = []
    for _, row in final_df.iterrows():
        title   = row.get("title",   "Unknown")
        authors = str(row.get("authors", ""))
        score   = row.get("rerank_score", 0)

        auth_parts = authors.split(";") if authors else []
        auth_str   = auth_parts[0] if auth_parts else ""
        if len(auth_parts) > 2:
            auth_str = f"{auth_parts[0]} et al."
        elif len(auth_parts) == 2:
            auth_str = f"{auth_parts[0]} & {auth_parts[1]}"

        explain   = explanations.get(title, "")
        # Show first 80 chars of explanation in caption if available
        exp_line  = f"\n✦ {explain[:80]}..." if explain and len(explain) > 10 else ""
        caption   = f"{title}\n{auth_str}{exp_line}"

        thumb = row.get("large_thumbnail", "cover-not-found.jpg")
        gallery.append((thumb, caption))

    # ── Detail panel for top book ─────────────────────────────────────────────
    detail_html = ""
    if not final_df.empty:
        top_row     = final_df.iloc[0]
        top_title   = top_row.get("title", "")
        top_explain = explanations.get(top_title, "")
        detail_html = build_detail_html(top_row, top_explain)

    # ── Sidebar ───────────────────────────────────────────────────────────────
    reasoning_short = reasoning[:220] + "..." if len(reasoning) > 220 else reasoning

    yield (
        build_trace_html(metrics, query_history),
        gallery,
        f'<span style="font-family:Libre Baskerville,serif;font-size:12px;font-style:italic;'
        f'color:#4a3a2a;line-height:1.7">{reasoning_short}</span>',
        build_metrics_html(metrics),
        build_tools_html(metrics.get("tools_called", [])),
        build_rewrites_html(query_history, reflections),
        build_results_header(len(final_df), query),
        detail_html,
    )


# ── Gradio layout ─────────────────────────────────────────────────────────────

with gr.Blocks(title="BookMind — Agentic RAG") as app:

    gr.HTML(f"""
    <div id="bm-header">
      <div>
        <span class="bm-logo">Book<em>Mind</em></span>
        <span class="bm-badge">Agentic RAG</span>
      </div>
      <div class="bm-status">
        <span class="bm-dot"></span>
        flan-t5-base · 5,197 books · ChromaDB
      </div>
    </div>
    <style>{CSS}</style>
    """)

    # ── Search row ────────────────────────────────────────────────────────────
    with gr.Row(elem_id="bm-search"):
        query_input = gr.Textbox(
            label       = "Describe the story you're looking for",
            placeholder = "e.g. A melancholic love story set during wartime with beautiful prose",
            lines       = 1,
            scale       = 6,
            elem_id     = "bm-query",
        )
        category_dd = gr.Dropdown(
            choices = CATEGORIES,
            value   = "All",
            label   = "Category",
            scale   = 1,
            elem_id = "bm-cat",
        )
        tone_dd = gr.Dropdown(
            choices = TONES,
            value   = "All",
            label   = "Tone",
            scale   = 1,
            elem_id = "bm-tone",
        )
        find_btn = gr.Button("Find Books →", variant="primary", scale=1, elem_id="bm-btn")

    # ── Agent trace bar ───────────────────────────────────────────────────────
    trace_bar = gr.HTML(
        value=(
            '<div style="background:#0a0805;padding:9px 40px;font-family:DM Mono,monospace;'
            'font-size:10px;color:rgba(255,255,255,0.2);border-bottom:1px solid '
            'rgba(201,168,76,0.12)">Agent trace will appear here after search...</div>'
        ),
        elem_id="bm-trace",
    )

    # ── Body ──────────────────────────────────────────────────────────────────
    with gr.Row():

        # Sidebar
        with gr.Column(scale=1, min_width=240, elem_id="bm-sidebar"):

            gr.HTML('<div class="bm-sidebar-label">Agent Reasoning</div>')
            reasoning_box = gr.HTML(
                value='<span style="font-family:Libre Baskerville,serif;font-size:12px;'
                      'font-style:italic;color:#6b6358">Run a search to see agent reasoning...</span>',
                elem_id="bm-reasoning",
            )

            gr.HTML('<div class="bm-sidebar-label" style="margin-top:18px">Run Metrics</div>')
            metrics_box = gr.HTML(
                value=build_metrics_html({}),
                elem_id="bm-metrics",
            )

            gr.HTML('<div class="bm-sidebar-label" style="margin-top:18px">Tools Called</div>')
            tools_box = gr.HTML(
                value=build_tools_html([]),
                elem_id="bm-tools",
            )

            gr.HTML('<div class="bm-sidebar-label" style="margin-top:18px">Query Rewrites</div>')
            rewrites_box = gr.HTML(
                value='<span style="color:#6b6358;font-size:11px">No rewrites yet</span>',
                elem_id="bm-rewrites",
            )

            gr.HTML(
                '<div style="margin-top:20px;padding:8px 10px;background:rgba(74,103,65,0.06);'
                'border:1px solid rgba(74,103,65,0.18);border-radius:4px;display:flex;'
                'align-items:center;gap:7px">'
                '<span style="width:6px;height:6px;border-radius:50%;background:#4a6741;'
                'box-shadow:0 0 5px #4a6741;flex-shrink:0"></span>'
                '<span style="font-family:DM Mono,monospace;font-size:10px;color:#4a6741">'
                'LangSmith tracing ON</span></div>'
            )

        # Main grid
        with gr.Column(scale=4, elem_id="bm-grid"):

            results_header = gr.HTML(
                value='<div style="font-family:Libre Baskerville,serif;font-size:20px;'
                      'font-weight:700;color:#0e0c09;margin-bottom:16px">Recommendations</div>',
            )

            book_gallery = gr.Gallery(
                label      = "",
                columns    = 4,
                rows       = 4,
                height     = "auto",
                elem_id    = "bm-gallery",
                show_label = False,
                object_fit = "cover",
            )

            # ── Top book detail panel ─────────────────────────────────────────
            gr.HTML(
                '<div style="font-family:DM Mono,monospace;font-size:9px;letter-spacing:2px;'
                'text-transform:uppercase;color:#6b6358;margin:24px 0 10px;'
                'padding-bottom:8px;border-bottom:1px solid rgba(14,12,9,0.1)">'
                'Top Match Detail</div>'
            )
            detail_box = gr.HTML(
                value="",
                elem_id="bm-detail",
            )

    # ── Wire up ───────────────────────────────────────────────────────────────
    outputs = [
        trace_bar,
        book_gallery,
        reasoning_box,
        metrics_box,
        tools_box,
        rewrites_box,
        results_header,
        detail_box,
    ]

    find_btn.click(
        fn      = recommend,
        inputs  = [query_input, category_dd, tone_dd],
        outputs = outputs,
    )
    query_input.submit(
        fn      = recommend,
        inputs  = [query_input, category_dd, tone_dd],
        outputs = outputs,
    )


if __name__ == "__main__":
    app.launch(
        server_name = "127.0.0.1",
        server_port = 7860,
        share       = False,
    )