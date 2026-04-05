
import pandas as pd
import gradio as gr

from rag_agent import initialize, run_agent

print("Starting BookMind Agentic RAG...")
books_df = initialize(
    csv_path  = "../dataset/books_with_emotions.csv",
    txt_path  = "../tagged_description.txt",
    llm_model = "google/flan-t5-base",
)

CATEGORIES = ["All"] + sorted(books_df["simple_categories"].dropna().unique().tolist())
TONES      = ["All", "Happy", "Surprising", "Angry", "Suspenseful", "Sad"]


CSS = """
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,400;0,500;0,600;1,300;1,400&family=Syne:wght@400;500;600;700&family=JetBrains+Mono:wght@300;400;500&display=swap');

:root {
    --void:       #f4f1ec;
    --deep:       #edeae3;
    --surface:    #e6e2d9;
    --raised:     #ffffff;
    --elevated:   #f9f7f3;
    --card:       #ffffff;
    --border:     rgba(60,50,35,0.10);
    --border-h:   rgba(60,50,35,0.20);
    --gold:       #c09248;
    --gold-bright:#d4a85a;
    --gold-dim:   #8a6a2e;
    --amber:      #c8643a;
    --teal:       #2a9d8f;
    --violet:     #7c5cbf;
    --sage:       #4a9e6d;
    --rose:       #c05050;
    --text-primary:   #1e1a14;
    --text-secondary: #4a4035;
    --text-muted:     #8a7d6a;
    --font-display: 'Cormorant Garamond', Georgia, serif;
    --font-ui:      'Syne', sans-serif;
    --font-mono:    'JetBrains Mono', monospace;
    --shadow-sm:  0 1px 4px rgba(0,0,0,0.08);
    --shadow-md:  0 4px 16px rgba(0,0,0,0.10);
    --shadow-lg:  0 10px 32px rgba(0,0,0,0.14);
    --shadow-card:0 2px 8px rgba(0,0,0,0.07);
}

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

.gradio-container {
    background: var(--void) !important;
    max-width: 100% !important;
    padding: 0 !important;
    margin: 0 !important;
    min-height: 100vh !important;
}
.gradio-container > .contain,
.gradio-container > .contain > div {
    padding: 0 !important;
    max-width: 100% !important;
    background: transparent !important;
}
footer, .built-with, .svelte-1ipelgc { display: none !important; }

.block, .form, .gap, .padded,
div[data-testid="block"],
div[data-testid="column"],
div[data-testid="row"] {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
    box-shadow: none !important;
}
#bm-search-zone > div,
#bm-search-zone .block,
#bm-search-zone .form,
#bm-search-zone .wrap,
#bm-search-zone > div > div {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
}
#bm-cat .wrap, #bm-tone .wrap,
#bm-cat .wrap-inner, #bm-tone .wrap-inner,
#bm-cat > div, #bm-tone > div,
#bm-cat > label, #bm-tone > label,
#bm-cat .block, #bm-tone .block {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
}

.gradio-container::before {
    content: '';
    position: fixed;
    inset: 0;
    background:
        radial-gradient(ellipse 60% 40% at 10% 5%,  rgba(192,146,72,0.05) 0%, transparent 55%),
        radial-gradient(ellipse 50% 35% at 90% 90%, rgba(42,157,143,0.03) 0%, transparent 55%);
    pointer-events: none;
    z-index: 0;
}

/* ── HEADER ──────────────────────────────────────────────────────────────── */
#bm-header {
    position: sticky; top: 0; z-index: 200;
    background: rgba(244,241,236,0.97);
    backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px);
    border-bottom: 1px solid var(--border);
    padding: 0 36px;
    height: 66px;
    display: flex; align-items: center; justify-content: space-between;
}
#bm-header::after {
    content: ''; position: absolute; bottom: 0; left: 0; right: 0; height: 1px;
    background: linear-gradient(90deg, transparent 0%, var(--gold) 40%, rgba(192,146,72,0.2) 70%, transparent 100%);
    opacity: 0.5;
}
.bm-logo-wrap { display: flex; align-items: center; gap: 12px; }

/* ── LARGER LOGO ────────────────────────────────────────────────────────── */
.bm-logo {
    font-family: var(--font-display) !important;
    font-size: 38px;
    font-weight: 600;
    letter-spacing: -0.5px;
    color: var(--text-primary);
    line-height: 1;
}
.bm-logo em { font-style: italic; color: var(--gold); font-weight: 400; }
.bm-tagline {
    font-family: var(--font-mono) !important;
    font-size: 8.5px; letter-spacing: 2px; text-transform: uppercase;
    color: var(--gold-dim); border: 1px solid rgba(192,146,72,0.30);
    padding: 3px 8px; border-radius: 3px; background: rgba(192,146,72,0.07);
    align-self: center; flex-shrink: 0;
}
.bm-nav-right { display: flex; align-items: center; gap: 18px; }
.bm-stat-pill {
    font-family: var(--font-mono) !important;
    font-size: 10px; color: var(--text-muted);
    display: flex; align-items: center; gap: 5px;
}
.bm-stat-pill span { color: var(--text-secondary); }
.bm-pulse {
    width: 6px; height: 6px; border-radius: 50%;
    background: var(--sage); animation: pulse 2.5s ease-in-out infinite;
}
@keyframes pulse {
    0%,100% { opacity:1; transform: scale(1); }
    50%      { opacity:0.6; transform: scale(0.85); }
}

/* Pipeline trace bar sticky offset matches header height */
#bm-trace-wrap [style*="top:54px"],
#bm-trace-wrap [style*="top:70px"],
#bm-trace-wrap div {
    top: 66px !important;
}

#bm-search-zone {
    background: var(--deep) !important;
    border-bottom: 1px solid var(--border) !important;
    padding: 6px 36px !important;
    position: relative;
}
#bm-search-zone::before {
    content: ''; position: absolute; bottom: 0; left: 0; right: 0; height: 1px;
    background: linear-gradient(90deg, transparent, rgba(192,146,72,0.18), transparent);
}
#bm-search-zone > div.gap,
#bm-search-zone > div {
    display: flex !important; align-items: flex-end !important;
    gap: 10px !important; flex-wrap: nowrap !important;
}
#bm-btn { align-self: flex-end !important; margin-bottom: 0 !important; }

#bm-query > label > span { display: none !important; }
#bm-query textarea,
#bm-query input[type="text"] {
    font-family: var(--font-display) !important; font-size: 15px !important;
    color: var(--text-primary) !important; background: var(--raised) !important;
    border: 1.5px solid var(--border-h) !important; border-radius: 8px !important;
    padding: 10px 14px !important; transition: all 0.2s ease !important;
    box-shadow: var(--shadow-sm) !important; caret-color: var(--gold) !important;
    min-height: 44px !important; resize: none !important;
}
#bm-query textarea:focus,
#bm-query input[type="text"]:focus {
    border-color: var(--gold) !important;
    box-shadow: 0 0 0 3px rgba(192,146,72,0.12) !important;
    outline: none !important; background: var(--elevated) !important;
}
#bm-query textarea::placeholder,
#bm-query input::placeholder {
    color: var(--text-muted) !important; font-style: italic !important; font-size: 13.5px !important;
}

#bm-cat > label > span, #bm-tone > label > span {
    font-family: var(--font-mono) !important; font-size: 9px !important;
    letter-spacing: 1.5px !important; text-transform: uppercase !important;
    color: var(--gold-dim) !important; margin-bottom: 4px !important;
    display: block !important; padding: 0 !important; background: transparent !important;
}
#bm-cat .wrap, #bm-tone .wrap {
    background: var(--raised) !important; border: 1.5px solid var(--border-h) !important;
    border-radius: 8px !important; min-height: 44px !important; padding: 0 12px !important;
    display: flex !important; align-items: center !important; cursor: pointer !important;
    transition: border-color 0.2s !important; box-shadow: var(--shadow-sm) !important;
}
#bm-cat .wrap:hover, #bm-tone .wrap:hover { border-color: var(--gold) !important; }
#bm-cat .wrap input, #bm-tone .wrap input,
#bm-cat input[type="text"], #bm-tone input[type="text"] {
    font-family: var(--font-ui) !important; font-size: 13px !important;
    color: var(--text-secondary) !important; background: transparent !important;
    border: none !important; padding: 0 !important; box-shadow: none !important;
}
#bm-cat .wrap svg, #bm-tone .wrap svg { color: var(--gold-dim) !important; }
#bm-cat .options, #bm-tone .options {
    background: var(--raised) !important;
    border: 1px solid rgba(192,146,72,0.25) !important;
    border-radius: 8px !important; box-shadow: var(--shadow-lg) !important; z-index: 9999 !important;
}
#bm-cat .item, #bm-tone .item {
    font-family: var(--font-ui) !important; font-size: 13px !important;
    color: var(--text-secondary) !important; padding: 7px 12px !important; background: transparent !important;
}
#bm-cat .item:hover, #bm-tone .item:hover,
#bm-cat .item.selected, #bm-tone .item.selected {
    background: rgba(192,146,72,0.09) !important; color: var(--gold-dim) !important;
}

#bm-btn {
    font-family: var(--font-ui) !important; font-size: 11.5px !important;
    font-weight: 700 !important; letter-spacing: 1.2px !important;
    text-transform: uppercase !important; color: #ffffff !important;
    background: linear-gradient(135deg, #c8a052 0%, #b07e32 100%) !important;
    border: none !important; border-radius: 8px !important;
    padding: 10px 22px !important; height: 44px !important;
    cursor: pointer !important; box-shadow: 0 3px 10px rgba(192,146,72,0.35) !important;
    transition: all 0.2s ease !important; white-space: nowrap !important;
}
#bm-btn:hover {
    background: linear-gradient(135deg, #d4a85a 0%, #c09248 100%) !important;
    box-shadow: 0 5px 18px rgba(192,146,72,0.45) !important; transform: translateY(-1px) !important;
}
#bm-btn:active { transform: translateY(0) !important; }

#bm-trace-wrap { background: var(--surface); border-bottom: 1px solid var(--border); }

#bm-body {
    display: flex !important; min-height: calc(100vh - 140px) !important;
    position: relative; z-index: 1; background: var(--void) !important;
}
#bm-body > div {
    display: flex !important; width: 100% !important;
    background: transparent !important; gap: 0 !important; align-items: stretch !important;
}

#bm-sidebar {
    width: 248px !important; flex-shrink: 0 !important;
    background: var(--deep) !important; border-right: 1px solid var(--border) !important;
    padding: 14px 14px !important; display: flex !important; flex-direction: column !important;
    gap: 0 !important; overflow-y: auto !important; overflow-x: hidden !important;
    position: relative !important; align-self: stretch !important; max-height: none !important;
}
/* Gradio injects a wrapping div inside the Column — make it flex column too */
#bm-sidebar > div {
    display: flex !important;
    flex-direction: column !important;
    gap: 0 !important;
    width: 100% !important;
    background: transparent !important;
}

/* ── SIDEBAR section heads — clear spacing so nothing overlaps ───────────── */
.bm-section-head {
    font-family: var(--font-mono) !important; font-size: 8px; letter-spacing: 2.2px;
    text-transform: uppercase; color: var(--gold-dim);
    padding: 10px 0 6px;
    margin-top: 10px;
    border-top: 1px solid var(--border);
    display: flex; align-items: center; gap: 8px;
    flex-shrink: 0;
}
.bm-section-head:first-child { border-top: none; padding-top: 0; margin-top: 0; }
.bm-section-head::after { content: ''; flex: 1; height: 1px; background: var(--border); }

/* Give each sidebar content block explicit bottom breathing room */
#bm-reasoning {
    font-family: var(--font-display) !important; font-size: 12.5px !important;
    font-style: italic !important; color: var(--text-secondary) !important;
    line-height: 1.75 !important; background: rgba(192,146,72,0.05) !important;
    border: 1px solid rgba(192,146,72,0.15) !important; border-radius: 6px !important;
    padding: 9px 11px !important;
    margin-bottom: 2px !important;
    display: block !important;
}
#bm-metrics, #bm-tools, #bm-rewrites {
    background: transparent !important; border: none !important; padding: 0 !important;
    margin-bottom: 2px !important;
    display: block !important;
}

/* Gradio wraps each HTML component in a .block div — strip decoration only,
   preserve natural height so section-heads' margin-top takes effect */
#bm-sidebar > div > .block,
#bm-sidebar > div > div > .block,
#bm-sidebar .block {
    display: block !important;
    padding: 0 !important;
    margin-top: 0 !important;
    margin-bottom: 0 !important;
    margin-left: 0 !important;
    margin-right: 0 !important;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    width: 100% !important;
    min-height: 0 !important;
    /* DO NOT set height:0 or overflow:hidden — would clip content */
}

/* (main-content layout defined below in the spacing section) */

/* ── GALLERY — kill ALL sources of the blank gap ────────────────────────── */
#bm-gallery {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
    margin: 0 !important;
    /* Gradio sets height inline; this wins only for non-inline — JS handles the rest */
    height: auto !important;
    min-height: 0 !important;
    max-height: none !important;
}
/* Every possible Gradio internal wrapper */
#bm-gallery > div,
#bm-gallery > div > div,
#bm-gallery .grid-container,
#bm-gallery [data-testid="gallery"],
#bm-gallery .overflow-y-auto,
#bm-gallery div[style*="overflow"],
#bm-gallery .grid-wrap,
#bm-gallery .grid,
#bm-gallery .wrap,
#bm-gallery .contain,
#bm-gallery .scroll-hide,
#bm-gallery .svelte-gallery,
#bm-gallery [class*="gallery"],
#bm-gallery [class*="scroll"] {
    height: auto !important;
    min-height: 0 !important;
    max-height: none !important;
    overflow: visible !important;
    padding-bottom: 0 !important;
    margin-bottom: 0 !important;
}
/* The sibling wrappers that Gradio injects around the Gallery component */
#bm-gallery ~ div,
#bm-gallery ~ .block,
#bm-gallery + div,
#bm-gallery + .block {
    margin-top: 0 !important;
    padding-top: 0 !important;
    height: auto !important;
    min-height: 0 !important;
}
#bm-gallery .thumbnail-item,
#bm-gallery .gallery-item {
    border-radius: 5px !important; overflow: hidden !important;
    background: var(--card) !important; border: 1px solid var(--border) !important;
    box-shadow: var(--shadow-card) !important;
    transition: all 0.22s cubic-bezier(0.34, 1.3, 0.64, 1) !important; cursor: pointer !important;
}
#bm-gallery .thumbnail-item:hover {
    transform: translateY(-3px) !important;
    box-shadow: 0 8px 20px rgba(0,0,0,0.12), 0 0 0 1px rgba(192,146,72,0.30) !important;
    border-color: rgba(192,146,72,0.30) !important;
}
#bm-gallery img {
    width: 100% !important; object-fit: cover !important;
    aspect-ratio: 2/3 !important; display: block !important;
    transition: transform 0.3s ease !important;
}
#bm-gallery .thumbnail-item:hover img { transform: scale(1.04) !important; }
#bm-gallery .caption-label {
    font-family: var(--font-ui) !important; font-size: 9.5px !important;
    padding: 5px 7px !important; color: var(--text-secondary) !important;
    background: var(--card) !important; line-height: 1.3 !important;
    border-top: 1px solid var(--border) !important;
}
#bm-gallery .thumbnail-item.selected,
#bm-gallery .gallery-item.selected {
    border-color: var(--gold) !important;
    box-shadow: 0 0 0 2px rgba(192,146,72,0.40), var(--shadow-card) !important;
}

/* Zero out only Gradio's invisible wrapper divs — not named components */
#bm-main-content {
    padding: 12px 22px !important;
    display: flex !important;
    flex-direction: column !important;
    gap: 0 !important;
    background: var(--void) !important;
}
#bm-main-content > div:not([id]),
#bm-main-content div.gap,
#bm-main-content div.form,
#bm-main-content div.padded,
#bm-main-content div.contain {
    gap: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
    background: transparent !important;
}

/* Results header */
#bm-results-hdr { margin-bottom: 6px !important; }

/* Gallery bottom — no extra space (JS handles internal height) */
#bm-gallery { margin-bottom: 0 !important; padding-bottom: 0 !important; }

/* ── DETAIL SECTION — 14px breathing room below gallery ─────────────── */
#bm-detail-wrap {
    margin-top: 14px !important;
    padding-top: 0 !important;
    display: block !important;
}
#bm-detail-wrap > .block,
#bm-detail-wrap > div { padding: 0 !important; margin: 0 !important; }

#bm-detail-label {
    font-family: var(--font-mono) !important; font-size: 8px; letter-spacing: 2.2px;
    text-transform: uppercase; color: var(--gold-dim);
    margin: 0 !important; padding-bottom: 6px !important; padding-top: 0 !important;
    border-bottom: 1px solid var(--border);
    display: block !important;
}
#bm-detail {
    margin-top: 8px !important;
    background: var(--raised) !important; border: 1px solid var(--border-h) !important;
    border-radius: 10px !important; padding: 0 !important;
    box-shadow: var(--shadow-md) !important; overflow: hidden !important; font-size: 13px !important;
}

.bm-welcome { text-align: center; padding: 56px 40px; font-family: var(--font-display); }
.bm-welcome-icon { font-size: 36px; margin-bottom: 10px; opacity: 0.25; }
.bm-welcome-title {
    font-size: 24px; font-weight: 400;
    color: rgba(30,26,20,0.28); letter-spacing: -0.3px; margin-bottom: 6px;
}
.bm-welcome-sub { font-size: 13px; font-style: italic; color: rgba(30,26,20,0.18); }

@keyframes shimmer {
    0%   { background-position: -200% center; }
    100% { background-position:  200% center; }
}
.bm-shimmer {
    background: linear-gradient(90deg,
        rgba(0,0,0,0.04) 0%, rgba(192,146,72,0.10) 50%, rgba(0,0,0,0.04) 100%);
    background-size: 200% auto; animation: shimmer 1.5s linear infinite; border-radius: 4px;
}

::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--surface); }
::-webkit-scrollbar-thumb { background: rgba(0,0,0,0.12); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: rgba(192,146,72,0.35); }

button:focus-visible, input:focus-visible, select:focus-visible {
    outline: 2px solid var(--gold) !important; outline-offset: 2px !important;
}
"""



def build_trace_html(metrics: dict, query_history: list = None) -> str:
    steps_done = {s["step"] for s in metrics.get("steps", [])}
    pipeline = [
        ("vector-search",        "Vector Search",   "#2a9d8f"),
        ("metadata-filter",      "Metadata Filter", "#c09248"),
        ("cross-encoder-rerank", "Re-Rank",         "#7c5cbf"),
        ("explain-books",        "LLM Explain",     "#c8643a"),
    ]
    parts = []
    for step_id, label, color in pipeline:
        done = step_id in steps_done
        if done:
            bg, border, txt_color, dot_bg, tick = f"{color}14", f"{color}38", color, color, "✓ "
        else:
            bg, border = "transparent", "rgba(60,50,35,0.10)"
            txt_color, dot_bg, tick = "rgba(74,64,53,0.30)", "rgba(60,50,35,0.12)", ""
        parts.append(
            f'<div style="display:inline-flex;align-items:center;gap:7px;'
            f'padding:5px 14px;border-radius:20px;background:{bg};'
            f'border:1px solid {border};transition:all 0.3s ease">'
            f'<span style="width:6px;height:6px;border-radius:50%;background:{dot_bg};flex-shrink:0"></span>'
            f'<span style="font-family:JetBrains Mono,monospace;font-size:10px;color:{txt_color};'
            f'letter-spacing:0.5px;white-space:nowrap">{tick}{label}</span></div>'
        )
    connector = '<span style="color:rgba(60,50,35,0.18);margin:0 2px;font-size:14px">→</span>'
    total    = metrics.get("total_s", 0)
    rewrites = len(query_history) - 1 if query_history and len(query_history) > 1 else 0
    extra_chips = ""
    if rewrites > 0:
        extra_chips = (
            f'<span style="margin-left:12px;font-family:JetBrains Mono,monospace;font-size:9px;'
            f'color:#8a6a2e;border:1px solid rgba(192,146,72,0.25);padding:3px 10px;'
            f'border-radius:20px;background:rgba(192,146,72,0.06)">'
            f'↻ {rewrites} rewrite{"s" if rewrites>1 else ""}</span>'
        )
    timer_html = ""
    if total:
        timer_html = (
            f'<span style="margin-left:auto;font-family:JetBrains Mono,monospace;font-size:10px;'
            f'color:rgba(74,64,53,0.35);padding:5px 12px;border-radius:20px;'
            f'border:1px solid rgba(60,50,35,0.10);background:rgba(0,0,0,0.02)">⏱ {total}s</span>'
        )
    label_badge = (
        '<span style="font-family:JetBrains Mono,monospace;font-size:8.5px;letter-spacing:2px;'
        'text-transform:uppercase;color:rgba(74,64,53,0.35);margin-right:14px;'
        'flex-shrink:0;border-right:1px solid rgba(60,50,35,0.10);padding-right:14px">Pipeline</span>'
    )
    return (
        f'<div style="background:#edeae3;padding:10px 36px;display:flex;align-items:center;'
        f'flex-wrap:wrap;gap:6px;border-bottom:1px solid rgba(60,50,35,0.10);min-height:44px;'
        f'position:sticky;top:66px;z-index:100">'
        f'{label_badge}{connector.join(parts)}{extra_chips}{timer_html}</div>'
    )


def build_metrics_html(metrics: dict) -> str:
    if not metrics:
        return (
            '<div style="padding:6px 0">'
            '<div class="bm-shimmer" style="height:13px;margin-bottom:7px;width:100%"></div>'
            '<div class="bm-shimmer" style="height:13px;margin-bottom:7px;width:80%"></div>'
            '<div class="bm-shimmer" style="height:13px;margin-bottom:7px;width:90%"></div>'
            '<div class="bm-shimmer" style="height:13px;margin-bottom:7px;width:70%"></div>'
            '</div>'
        )
    top_score   = metrics.get("top_score", 0.0)
    score_color = "#4a9e6d" if top_score >= 1.0 else ("#c09248" if top_score >= 0 else "#c05050")
    rows = [
        ("Candidates",   metrics.get("candidates",   "—"), "#2a9d8f"),
        ("After filter", metrics.get("after_filter", "—"), "#c09248"),
        ("After rerank", metrics.get("after_rerank", "—"), "#4a9e6d"),
        ("Top score",    f"{top_score:.3f}" if isinstance(top_score, float) else "—", score_color),
        ("LLM calls",    metrics.get("llm_calls",    "—"), "#7c5cbf"),
        ("Total time",   f'{metrics.get("total_s","—")}s',  "#c8643a"),
    ]
    html = '<div style="display:flex;flex-direction:column;gap:3px">'
    for label, val, color in rows:
        html += (
            f'<div style="display:flex;justify-content:space-between;align-items:center;'
            f'padding:6px 9px;border-radius:5px;background:rgba(0,0,0,0.03);'
            f'border:1px solid rgba(60,50,35,0.09);margin-bottom:1px">'
            f'<span style="font-family:Syne,sans-serif;font-size:11px;color:#4a4035">{label}</span>'
            f'<span style="font-family:JetBrains Mono,monospace;font-size:11px;font-weight:500;'
            f'color:{color}">{val}</span></div>'
        )
    return html + "</div>"


def build_tools_html(tools_called: list) -> str:
    config = {
        "vector-search":        ("#2a9d8f", "Vector Search",   "◈"),
        "metadata-filter":      ("#c09248", "Metadata Filter", "◈"),
        "cross-encoder-rerank": ("#7c5cbf", "Cross-Encoder",   "◈"),
        "explain-books":        ("#c8643a", "LLM Explain",     "◈"),
    }
    if not tools_called:
        return '<span style="font-family:Syne,sans-serif;font-size:11px;color:rgba(74,64,53,0.35)">Awaiting run…</span>'
    chips = '<div style="display:flex;flex-direction:column;gap:4px">'
    for t in tools_called:
        color, label, icon = config.get(t, ("#888", t, "◈"))
        chips += (
            f'<div style="display:flex;align-items:center;gap:8px;padding:6px 9px;'
            f'border-radius:5px;background:{color}0f;border:1px solid {color}30">'
            f'<span style="width:5px;height:5px;border-radius:50%;background:{color};flex-shrink:0"></span>'
            f'<span style="font-family:Syne,sans-serif;font-size:11px;color:{color};font-weight:500">{label}</span>'
            f'<span style="margin-left:auto;font-family:JetBrains Mono,monospace;font-size:9px;'
            f'color:{color}aa">done</span></div>'
        )
    return chips + '</div>'


def build_rewrites_html(query_history: list, reflections: list) -> str:
    if not query_history or len(query_history) == 1:
        return '<span style="font-family:Syne,sans-serif;font-size:11px;color:rgba(74,64,53,0.30)">No rewrites needed</span>'
    html = '<div style="display:flex;flex-direction:column;gap:5px">'
    for i, q in enumerate(query_history):
        is_orig = i == 0
        color   = "rgba(74,64,53,0.40)"   if is_orig else "#8a6a2e"
        bg      = "rgba(0,0,0,0.02)"      if is_orig else "rgba(192,146,72,0.06)"
        border  = "rgba(60,50,35,0.08)"   if is_orig else "rgba(192,146,72,0.20)"
        label   = "Original"              if is_orig else f"Rewrite {i}"
        icon    = "○"                     if is_orig else "↻"
        ref   = next((r for r in reflections if r.get("attempt") == i + 1), {})
        score = ref.get("top_score")
        score_tag = ""
        if score is not None:
            sc = "#4a9e6d" if score >= 1.0 else ("#c09248" if score >= 0 else "#c05050")
            score_tag = (
                f'<span style="font-family:JetBrains Mono,monospace;font-size:10px;'
                f'color:{sc};margin-left:auto">{score:.2f}</span>'
            )
        html += (
            f'<div style="padding:7px 9px;border-radius:5px;background:{bg};border:1px solid {border}">'
            f'<div style="display:flex;align-items:center;gap:6px;margin-bottom:3px">'
            f'<span style="font-family:JetBrains Mono,monospace;font-size:9px;color:{color};'
            f'letter-spacing:1px;text-transform:uppercase">{icon} {label}</span>{score_tag}</div>'
            f'<div style="font-family:Cormorant Garamond,serif;font-size:12px;font-style:italic;'
            f'color:rgba(74,64,53,0.55);line-height:1.5;margin-top:2px">'
            f'"{q[:65]}{"..." if len(q) > 65 else ""}"</div></div>'
        )
    return html + "</div>"


def build_detail_html(row: pd.Series, explanation: str) -> str:
    title   = row.get("title", "Unknown")
    authors = str(row.get("authors", ""))
    desc    = str(row.get("description", ""))
    score   = row.get("rerank_score", 0.0)
    cat     = row.get("simple_categories", "")
    rating  = row.get("average_rating", "")
    thumb   = row.get("large_thumbnail", "")
    auth_parts = [a.strip() for a in authors.split(";") if a.strip()]
    if not auth_parts:         auth_str = "Unknown Author"
    elif len(auth_parts) == 1: auth_str = auth_parts[0]
    elif len(auth_parts) == 2: auth_str = f"{auth_parts[0]} & {auth_parts[1]}"
    else:                      auth_str = f"{auth_parts[0]} et al."
    score_color = "#4a9e6d" if score >= 1.0 else ("#c09248" if score >= 0 else "#c05050")
    desc_words  = desc.split()
    desc_text   = " ".join(desc_words[:80]) + ("…" if len(desc_words) > 80 else "")
    cover_html = ""
    if thumb:
        cover_html = (
            f'<div style="width:120px;flex-shrink:0;border-radius:6px;overflow:hidden;'
            f'box-shadow:0 6px 24px rgba(0,0,0,0.14);border:1px solid rgba(60,50,35,0.10)">'
            f'<img src="{thumb}" style="width:100%;display:block;object-fit:cover;aspect-ratio:2/3" '
            f'onerror="this.style.display=\'none\'" /></div>'
        )
    explain_html = ""
    if explanation and len(explanation.strip()) > 10:
        explain_html = (
            f'<div style="margin-top:14px;padding:12px 14px;'
            f'background:linear-gradient(135deg,rgba(192,146,72,0.07) 0%,rgba(200,100,58,0.04) 100%);'
            f'border:1px solid rgba(192,146,72,0.20);border-radius:7px;border-left:3px solid #c09248">'
            f'<div style="font-family:JetBrains Mono,monospace;font-size:8px;letter-spacing:2px;'
            f'text-transform:uppercase;color:rgba(138,106,46,0.70);margin-bottom:7px">Why this book</div>'
            f'<div style="font-family:Cormorant Garamond,serif;font-size:14px;font-style:italic;'
            f'color:rgba(74,64,53,0.82);line-height:1.75">{explanation}</div></div>'
        )
    meta_row = ""
    if cat:
        meta_row += (
            f'<span style="font-family:Syne,sans-serif;font-size:9.5px;padding:2px 9px;'
            f'border-radius:20px;background:rgba(42,157,143,0.08);color:#2a9d8f;'
            f'border:1px solid rgba(42,157,143,0.18)">{cat}</span>'
        )
    if rating:
        meta_row += (
            f' <span style="font-family:JetBrains Mono,monospace;font-size:9.5px;padding:2px 9px;'
            f'border-radius:20px;background:rgba(192,146,72,0.08);color:#8a6a2e;'
            f'border:1px solid rgba(192,146,72,0.20)">★ {rating}</span>'
        )
    meta_row += (
        f' <span style="font-family:JetBrains Mono,monospace;font-size:9.5px;padding:2px 9px;'
        f'border-radius:20px;background:{score_color}12;color:{score_color};'
        f'border:1px solid {score_color}30">score {score:.3f}</span>'
    )
    return (
        f'<div style="display:flex;gap:20px;padding:18px 22px;'
        f'background:linear-gradient(135deg,#faf8f4 0%,#f4f1ec 100%)">'
        f'{cover_html}'
        f'<div style="flex:1;min-width:0">'
        f'<div style="font-family:Cormorant Garamond,serif;font-size:21px;font-weight:600;'
        f'color:#1e1a14;line-height:1.25;margin-bottom:4px;letter-spacing:-0.3px">{title}</div>'
        f'<div style="font-family:Syne,sans-serif;font-size:11.5px;color:rgba(74,64,53,0.60);'
        f'margin-bottom:9px;letter-spacing:0.3px">{auth_str}</div>'
        f'<div style="display:flex;flex-wrap:wrap;gap:5px;margin-bottom:11px;align-items:center">'
        f'{meta_row}</div>'
        f'<div style="font-family:Cormorant Garamond,serif;font-size:13.5px;'
        f'color:rgba(74,64,53,0.82);line-height:1.78;'
        f'border-top:1px solid rgba(60,50,35,0.08);padding-top:11px">{desc_text}</div>'
        f'{explain_html}</div></div>'
    )


def build_results_header(n: int, query: str) -> str:
    if n == 0:
        return (
            '<div style="font-family:Cormorant Garamond,serif;font-size:22px;font-weight:400;'
            'color:#c05050;margin-bottom:7px">No results found</div>'
            '<div style="font-family:Syne,sans-serif;font-size:12px;color:rgba(74,64,53,0.45);">'
            'Try broadening your filters or rewording your query.</div>'
        )
    return (
        f'<div style="display:flex;align-items:baseline;gap:12px;margin-bottom:5px">'
        f'<span style="font-family:Cormorant Garamond,serif;font-size:23px;font-weight:600;'
        f'color:#1e1a14;letter-spacing:-0.5px">Recommendations</span>'
        f'<span style="font-family:JetBrains Mono,monospace;font-size:10.5px;'
        f'color:rgba(74,64,53,0.35);padding:2px 9px;border:1px solid rgba(60,50,35,0.10);'
        f'border-radius:20px;background:rgba(0,0,0,0.02)">{n} books</span></div>'
        f'<div style="font-family:Cormorant Garamond,serif;font-size:13.5px;font-style:italic;'
        f'color:rgba(74,64,53,0.40);margin-bottom:8px;'
        f'border-left:2px solid rgba(192,146,72,0.35);padding-left:10px">'
        f'"{query[:90]}{"…" if len(query)>90 else ""}"</div>'
    )



def recommend(query: str, category: str, tone: str):
    empty_trace    = build_trace_html({"steps": [], "total_s": ""})
    empty_metrics  = build_metrics_html({})
    empty_tools    = build_tools_html([])
    empty_rewrites = build_rewrites_html([], [])

    if not query.strip():
        yield (
            empty_trace, [],
            '<span style="font-family:Cormorant Garamond,serif;font-size:13px;font-style:italic;'
            'color:rgba(74,64,53,0.30)">Enter a query to begin.</span>',
            empty_metrics, empty_tools, empty_rewrites,
            '<div class="bm-welcome"><div class="bm-welcome-icon">📚</div>'
            '<div class="bm-welcome-title">Discover your next book</div>'
            '<div class="bm-welcome-sub">Describe what you\'re looking for above</div></div>',
            "", None, None,
        )
        return

    skeletons = "".join([
        f'<div style="display:inline-block;width:calc(12.5% - 9px);margin:4px;border-radius:6px;'
        f'overflow:hidden;background:#ffffff;border:1px solid rgba(60,50,35,0.09)">'
        f'<div class="bm-shimmer" style="width:100%;height:130px"></div>'
        f'<div style="padding:7px 8px">'
        f'<div class="bm-shimmer" style="height:10px;margin-bottom:5px;border-radius:3px;width:85%"></div>'
        f'<div class="bm-shimmer" style="height:8px;border-radius:3px;width:55%"></div>'
        f'</div></div>'
        for _ in range(16)
    ])
    thinking_trace = (
        f'<div style="background:#edeae3;border-bottom:1px solid rgba(60,50,35,0.10);'
        f'padding:10px 36px;display:flex;align-items:center;gap:10px;'
        f'min-height:44px;position:sticky;top:66px;z-index:100">'
        f'<span style="font-family:JetBrains Mono,monospace;font-size:10px;'
        f'color:rgba(138,106,46,0.65);letter-spacing:1.5px">Initialising pipeline</span>'
        f'<span class="bm-shimmer" style="display:inline-block;width:110px;height:9px;'
        f'border-radius:10px;vertical-align:middle"></span></div>'
    )
    yield (
        thinking_trace, [],
        '<span style="font-family:Cormorant Garamond,serif;font-size:13px;font-style:italic;'
        'color:rgba(74,64,53,0.40)">Analysing your query…</span>',
        build_metrics_html({}), empty_tools, empty_rewrites,
        f'<div style="font-family:Cormorant Garamond,serif;font-size:15px;font-style:italic;'
        f'color:rgba(74,64,53,0.28);padding:28px 0 14px;letter-spacing:0.2px">'
        f'Searching through 5,197 books…</div><div style="margin-top:6px">{skeletons}</div>',
        "", None, None,
    )

    result        = run_agent(query, category, tone)
    final_df      = result["books"]
    explanations  = result["explanations"]
    metrics       = result["metrics"]
    reasoning     = result["reasoning"]
    query_history = result.get("query_history", [query])
    reflections   = result.get("reflections",   [])

    gallery = []
    for _, row in final_df.iterrows():
        title       = row.get("title", "Unknown")
        authors_raw = str(row.get("authors", ""))
        auth_parts  = [a.strip() for a in authors_raw.split(";") if a.strip()]
        if not auth_parts:         auth_str = ""
        elif len(auth_parts) == 1: auth_str = auth_parts[0]
        elif len(auth_parts) == 2: auth_str = f"{auth_parts[0]} & {auth_parts[1]}"
        else:                      auth_str = f"{auth_parts[0]} et al."
        explain  = explanations.get(title, "")
        exp_line = f"\n✦ {explain[:70]}…" if explain and len(explain) > 10 else ""
        caption  = f"{title}\n{auth_str}{exp_line}"
        thumb    = row.get("large_thumbnail", "cover-not-found.jpg")
        gallery.append((thumb, caption))

    detail_html = ""
    if not final_df.empty:
        top_row     = final_df.iloc[0]
        top_explain = explanations.get(top_row.get("title", ""), "")
        detail_html = build_detail_html(top_row, top_explain)

    reasoning_preview = reasoning[:260] + "…" if len(reasoning) > 260 else reasoning
    reasoning_html = (
        f'<div style="font-family:Cormorant Garamond,serif;font-size:12.5px;font-style:italic;'
        f'color:rgba(74,64,53,0.70);line-height:1.8;padding:9px 11px;'
        f'background:rgba(0,0,0,0.02);border-radius:5px;border-left:2px solid rgba(192,146,72,0.30)">'
        f'{reasoning_preview}</div>'
    )

    yield (
        build_trace_html(metrics, query_history),
        gallery,
        reasoning_html,
        build_metrics_html(metrics),
        build_tools_html(metrics.get("tools_called", [])),
        build_rewrites_html(query_history, reflections),
        build_results_header(len(final_df), query),
        detail_html,
        final_df, explanations,
    )



with gr.Blocks(title="BookMind — Agentic RAG", css=CSS) as app:

    state_df      = gr.State(value=None)
    state_explain = gr.State(value=None)

    gr.HTML("""
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    """)

    gr.HTML("""
    <div id="bm-header">
      <div class="bm-logo-wrap">
        <span class="bm-logo">Book<em>Mind</em></span>
        <span class="bm-tagline">Agentic RAG</span>
      </div>
      <div class="bm-nav-right">
        <div class="bm-stat-pill">
          <span style="color:rgba(60,50,35,0.35)">model</span>
          <span>flan-t5-base</span>
        </div>
        <div class="bm-stat-pill">
          <span style="color:rgba(60,50,35,0.35)">corpus</span>
          <span>5,197 books</span>
        </div>
        <div class="bm-stat-pill">
          <span style="color:rgba(60,50,35,0.35)">index</span>
          <span>ChromaDB</span>
        </div>
        <div class="bm-stat-pill">
          <div class="bm-pulse"></div>
          <span style="color:#4a9e6d">online</span>
        </div>
      </div>
    </div>
    """)

    with gr.Row(elem_id="bm-search-zone"):
        query_input = gr.Textbox(
            label="",
            placeholder="Describe the book you're looking for…  e.g. a melancholic wartime love story with exquisite prose",
            lines=1, scale=6, elem_id="bm-query",
        )
        category_dd = gr.Dropdown(choices=CATEGORIES, value="All", label="Category", scale=1, elem_id="bm-cat")
        tone_dd     = gr.Dropdown(choices=TONES, value="All", label="Emotional Tone", scale=1, elem_id="bm-tone")
        find_btn    = gr.Button("Find Books →", variant="primary", scale=1, elem_id="bm-btn")

    gr.HTML("""<script>
    // ── Light theme patch ────────────────────────────────────────────────────
    (function patchGradioStyles() {
        function applyLight() {
            var zone = document.getElementById('bm-search-zone');
            if (!zone) return;
            zone.querySelectorAll('.wrap, .wrap-inner, .block, .form, .gap').forEach(function(el) {
                el.style.setProperty('background','transparent','important');
                el.style.setProperty('border','none','important');
                el.style.setProperty('box-shadow','none','important');
            });
            ['bm-cat','bm-tone'].forEach(function(id) {
                var el = document.getElementById(id);
                if (!el) return;
                var wrap = el.querySelector('.wrap');
                if (wrap) {
                    wrap.style.setProperty('background','#ffffff','important');
                    wrap.style.setProperty('border','1.5px solid rgba(60,50,35,0.20)','important');
                    wrap.style.setProperty('border-radius','8px','important');
                    wrap.style.setProperty('min-height','44px','important');
                    wrap.style.setProperty('padding','0 12px','important');
                    wrap.style.setProperty('box-shadow','0 1px 4px rgba(0,0,0,0.08)','important');
                }
                var inp = el.querySelector('input[type="text"]');
                if (inp) {
                    inp.style.setProperty('background','transparent','important');
                    inp.style.setProperty('color','#4a4035','important');
                    inp.style.setProperty('border','none','important');
                }
                var ul = el.querySelector('.options, ul');
                if (ul) {
                    ul.style.setProperty('background','#ffffff','important');
                    ul.style.setProperty('border','1px solid rgba(192,146,72,0.25)','important');
                    ul.style.setProperty('border-radius','8px','important');
                    ul.style.setProperty('box-shadow','0 10px 32px rgba(0,0,0,0.14)','important');
                }
            });
        }
        applyLight();
        setTimeout(applyLight, 400);
        setTimeout(applyLight, 1000);
        document.addEventListener('click', function(){ setTimeout(applyLight, 60); });
    })();

    // ── Gallery gap fix (aggressive) ─────────────────────────────────────────
    // Gradio injects inline style="height: Npx" AND wraps the gallery in
    // extra divs that also carry fixed heights. CSS !important cannot beat
    // inline styles, so we use a MutationObserver to strip them after every
    // render, and also directly zero-out the gallery wrapper itself.
    (function fixGalleryGap() {
        function nukeHeights(root) {
            if (!root) return;

            // 1. Strip any pixel-height inline style from every descendant
            root.querySelectorAll('*').forEach(function(el) {
                var h = el.style.height;
                if (h && h !== 'auto' && h !== '100%' && h !== '0px') {
                    el.style.removeProperty('height');
                    el.style.setProperty('height', 'auto', 'important');
                    el.style.setProperty('min-height', '0', 'important');
                    el.style.setProperty('max-height', 'none', 'important');
                    el.style.setProperty('overflow-y', 'visible', 'important');
                    el.style.setProperty('overflow', 'visible', 'important');
                }
            });

            // 2. Also fix the root element itself
            root.style.setProperty('height', 'auto', 'important');
            root.style.setProperty('min-height', '0', 'important');
            root.style.setProperty('max-height', 'none', 'important');
            root.style.setProperty('overflow', 'visible', 'important');

            // 3. Fix any PARENT wrappers Gradio adds around #bm-gallery
            var parent = root.parentElement;
            var depth  = 0;
            while (parent && depth < 6) {
                var ph = parent.style.height;
                if (ph && ph !== 'auto' && ph !== '100%') {
                    parent.style.setProperty('height', 'auto', 'important');
                    parent.style.setProperty('min-height', '0', 'important');
                    parent.style.setProperty('max-height', 'none', 'important');
                    parent.style.setProperty('overflow', 'visible', 'important');
                }
                // Stop climbing once we hit a layout boundary
                if (parent.id === 'bm-main-content') break;
                parent = parent.parentElement;
                depth++;
            }
        }

        function init() {
            var gallery = document.getElementById('bm-gallery');
            if (!gallery) { setTimeout(init, 250); return; }

            nukeHeights(gallery);

            var obs = new MutationObserver(function() {
                clearTimeout(obs._t);
                obs._t = setTimeout(function() { nukeHeights(gallery); }, 30);
            });

            obs.observe(gallery, {
                childList:       true,
                subtree:         true,
                attributes:      true,
                attributeFilter: ['style', 'class']
            });

            // Also watch the parent in case Gradio re-wraps the whole component
            if (gallery.parentElement) {
                obs.observe(gallery.parentElement, {
                    childList:       true,
                    subtree:         false,
                    attributes:      true,
                    attributeFilter: ['style']
                });
            }
        }

        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', init);
        } else {
            setTimeout(init, 0);
        }
    })();
    </script>""")

    trace_bar = gr.HTML(
        value=(
            '<div style="background:#edeae3;padding:10px 36px;font-family:JetBrains Mono,monospace;'
            'font-size:10px;color:rgba(74,64,53,0.30);border-bottom:1px solid rgba(60,50,35,0.10);'
            'min-height:44px;display:flex;align-items:center;letter-spacing:1px">'
            'Pipeline trace will appear here after your first search…</div>'
        ),
        elem_id="bm-trace-wrap",
    )

    with gr.Row(elem_id="bm-body"):

        with gr.Column(scale=1, min_width=260, elem_id="bm-sidebar"):
            gr.HTML('<div class="bm-section-head">Agent Reasoning</div>')
            reasoning_box = gr.HTML(
                value=(
                    '<span style="font-family:Cormorant Garamond,serif;font-size:12.5px;'
                    'font-style:italic;color:rgba(74,64,53,0.28)">Run a search to see reasoning…</span>'
                ),
                elem_id="bm-reasoning",
            )
            gr.HTML('<div class="bm-section-head">Run Metrics</div>')
            metrics_box = gr.HTML(value=build_metrics_html({}), elem_id="bm-metrics")
            gr.HTML('<div class="bm-section-head">Tools Called</div>')
            tools_box = gr.HTML(value=build_tools_html([]), elem_id="bm-tools")
            gr.HTML('<div class="bm-section-head">Query Rewrites</div>')
            rewrites_box = gr.HTML(
                value='<span style="font-family:Syne,sans-serif;font-size:11px;color:rgba(74,64,53,0.28)">No rewrites yet</span>',
                elem_id="bm-rewrites",
            )
            gr.HTML(
                '<div style="margin-top:14px;padding:8px 10px;background:rgba(74,158,109,0.06);'
                'border:1px solid rgba(74,158,109,0.18);border-radius:6px;display:flex;align-items:center;gap:8px">'
                '<span style="width:6px;height:6px;border-radius:50%;background:#4a9e6d;'
                'flex-shrink:0;animation:pulse 2s ease-in-out infinite"></span>'
                '<span style="font-family:JetBrains Mono,monospace;font-size:9px;color:#4a9e6d;'
                'letter-spacing:0.8px">LangSmith tracing active</span></div>'
            )

        with gr.Column(scale=4, elem_id="bm-main-content"):
            results_header = gr.HTML(
                value=(
                    '<div class="bm-welcome"><div class="bm-welcome-icon">📚</div>'
                    '<div class="bm-welcome-title">Discover your next book</div>'
                    '<div class="bm-welcome-sub">Describe what you\'re in the mood for above</div></div>'
                ),
                elem_id="bm-results-hdr",
            )
            book_gallery = gr.Gallery(
                label="", columns=8, rows=2, height="auto",
                elem_id="bm-gallery", show_label=False,
                object_fit="cover", allow_preview=False,
            )
            detail_label = gr.HTML(
                value='<div id="bm-detail-label">Top Match Detail</div>',
                elem_id="bm-detail-wrap",
            )
            detail_box = gr.HTML(value="", elem_id="bm-detail")

    def on_gallery_select(evt: gr.SelectData, df, expl):
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            return '<div id="bm-detail-label">Top Match Detail</div>', ""
        idx = evt.index
        if idx < 0 or idx >= len(df):
            return '<div id="bm-detail-label">Top Match Detail</div>', ""
        row   = df.iloc[idx]
        title = row.get("title", "")
        exp   = expl.get(title, "") if isinstance(expl, dict) else ""
        label_text = "Top Match Detail" if idx == 0 else f"Selected Book — #{idx + 1}"
        return f'<div id="bm-detail-label">{label_text}</div>', build_detail_html(row, exp)

    outputs = [
        trace_bar, book_gallery, reasoning_box,
        metrics_box, tools_box, rewrites_box,
        results_header, detail_box,
        state_df, state_explain,
    ]
    find_btn.click(fn=recommend, inputs=[query_input, category_dd, tone_dd], outputs=outputs)
    query_input.submit(fn=recommend, inputs=[query_input, category_dd, tone_dd], outputs=outputs)
    book_gallery.select(fn=on_gallery_select, inputs=[state_df, state_explain], outputs=[detail_label, detail_box])


if __name__ == "__main__":
    app.launch(server_name="127.0.0.1", server_port=7860, share=False)