<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>BookMind — Contextual RAG Book Recommender</title>
<link href="https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,500;0,600;1,300;1,400&family=DM+Mono:wght@300;400;500&family=DM+Sans:wght@300;400;500&display=swap" rel="stylesheet">
<style>
  :root {
    --ink: #1a1209;
    --parchment: #f7f2e8;
    --cream: #fdf9f2;
    --gold: #b8860b;
    --gold-light: #d4a017;
    --rust: #8b3a0f;
    --sage: #4a5c3a;
    --muted: #6b5c42;
    --border: #d4c9a8;
    --shadow: rgba(26,18,9,0.12);
  }

  * { margin: 0; padding: 0; box-sizing: border-box; }

  body {
    background-color: var(--cream);
    color: var(--ink);
    font-family: 'DM Sans', sans-serif;
    font-weight: 300;
    line-height: 1.7;
    background-image:
      radial-gradient(ellipse at 20% 0%, rgba(184,134,11,0.06) 0%, transparent 60%),
      radial-gradient(ellipse at 80% 100%, rgba(139,58,15,0.05) 0%, transparent 60%);
  }

  /* ─── GRAIN OVERLAY ─── */
  body::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)' opacity='0.03'/%3E%3C/svg%3E");
    pointer-events: none;
    z-index: 0;
    opacity: 0.4;
  }

  .wrapper {
    max-width: 900px;
    margin: 0 auto;
    padding: 60px 40px 100px;
    position: relative;
    z-index: 1;
  }

  /* ─── HERO ─── */
  .hero {
    text-align: center;
    padding: 80px 0 60px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 70px;
    position: relative;
  }

  .hero::before {
    content: '❧';
    display: block;
    font-size: 2rem;
    color: var(--gold);
    margin-bottom: 24px;
    opacity: 0.7;
  }

  .hero h1 {
    font-family: 'Cormorant Garamond', serif;
    font-size: clamp(3rem, 7vw, 5.5rem);
    font-weight: 300;
    letter-spacing: -0.02em;
    line-height: 1.05;
    color: var(--ink);
  }

  .hero h1 span {
    color: var(--gold);
    font-style: italic;
  }

  .hero .tagline {
    margin-top: 20px;
    font-size: 1.1rem;
    color: var(--muted);
    font-weight: 300;
    letter-spacing: 0.02em;
    font-style: italic;
    font-family: 'Cormorant Garamond', serif;
  }

  .hero .live-link {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    margin-top: 32px;
    padding: 12px 28px;
    background: var(--ink);
    color: var(--parchment);
    text-decoration: none;
    font-size: 0.8rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    font-family: 'DM Mono', monospace;
    transition: all 0.25s ease;
    border: 1px solid var(--ink);
  }

  .hero .live-link:hover {
    background: var(--gold);
    border-color: var(--gold);
    color: var(--ink);
  }

  .hero .live-link::before {
    content: '▶';
    font-size: 0.65rem;
  }

  .badges {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    justify-content: center;
    margin-top: 28px;
  }

  .badge {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    padding: 4px 12px;
    border: 1px solid var(--border);
    color: var(--muted);
    letter-spacing: 0.05em;
    background: rgba(247,242,232,0.6);
  }

  /* ─── SECTIONS ─── */
  section {
    margin-bottom: 64px;
  }

  h2 {
    font-family: 'Cormorant Garamond', serif;
    font-size: 2rem;
    font-weight: 400;
    color: var(--ink);
    margin-bottom: 24px;
    padding-bottom: 12px;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    gap: 12px;
  }

  h2 .section-num {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    color: var(--gold);
    letter-spacing: 0.1em;
    font-weight: 400;
    opacity: 0.8;
    align-self: flex-end;
    margin-bottom: 4px;
  }

  h3 {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.25rem;
    font-weight: 500;
    color: var(--ink);
    margin: 28px 0 10px;
  }

  p {
    color: var(--muted);
    margin-bottom: 14px;
    font-size: 0.95rem;
  }

  /* ─── FEATURE GRID ─── */
  .feature-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
    gap: 1px;
    background: var(--border);
    border: 1px solid var(--border);
    margin-top: 8px;
  }

  .feature-card {
    background: var(--cream);
    padding: 24px;
    transition: background 0.2s;
  }

  .feature-card:hover {
    background: var(--parchment);
  }

  .feature-card .icon {
    font-size: 1.4rem;
    margin-bottom: 10px;
    display: block;
  }

  .feature-card strong {
    display: block;
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.05rem;
    font-weight: 600;
    color: var(--ink);
    margin-bottom: 6px;
  }

  .feature-card p {
    font-size: 0.85rem;
    margin: 0;
    line-height: 1.55;
  }

  /* ─── ARCHITECTURE ─── */
  .arch-box {
    background: var(--ink);
    color: #e8dfc8;
    font-family: 'DM Mono', monospace;
    font-size: 0.78rem;
    line-height: 1.8;
    padding: 36px 40px;
    overflow-x: auto;
    position: relative;
  }

  .arch-box::before {
    content: 'PIPELINE';
    position: absolute;
    top: 14px;
    right: 20px;
    font-size: 0.65rem;
    letter-spacing: 0.15em;
    color: var(--gold);
    opacity: 0.6;
  }

  .arch-box .gold { color: #d4a017; }
  .arch-box .muted { color: #8a7a60; }
  .arch-box .green { color: #7aad6a; }

  /* ─── TABLES ─── */
  table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.88rem;
    margin-top: 8px;
  }

  thead tr {
    background: var(--ink);
    color: var(--parchment);
  }

  thead th {
    padding: 12px 18px;
    text-align: left;
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.1em;
    font-weight: 400;
  }

  tbody tr {
    border-bottom: 1px solid var(--border);
    transition: background 0.15s;
  }

  tbody tr:hover {
    background: var(--parchment);
  }

  td {
    padding: 11px 18px;
    color: var(--muted);
    vertical-align: top;
  }

  td:first-child {
    font-family: 'DM Mono', monospace;
    font-size: 0.8rem;
    color: var(--rust);
  }

  /* ─── CODE BLOCKS ─── */
  pre, code {
    font-family: 'DM Mono', monospace;
  }

  pre {
    background: #f0ebe0;
    border: 1px solid var(--border);
    border-left: 3px solid var(--gold);
    padding: 20px 24px;
    overflow-x: auto;
    font-size: 0.8rem;
    line-height: 1.7;
    color: #3a2f1f;
    margin: 12px 0;
  }

  code {
    background: rgba(184,134,11,0.08);
    padding: 1px 6px;
    border-radius: 2px;
    font-size: 0.85em;
    color: var(--rust);
  }

  pre code {
    background: none;
    padding: 0;
    color: inherit;
    font-size: inherit;
  }

  /* ─── STAGE STEPS ─── */
  .stages {
    counter-reset: stage;
    display: flex;
    flex-direction: column;
    gap: 0;
  }

  .stage {
    display: flex;
    gap: 0;
    position: relative;
  }

  .stage-line {
    display: flex;
    flex-direction: column;
    align-items: center;
    width: 48px;
    flex-shrink: 0;
  }

  .stage-dot {
    width: 28px;
    height: 28px;
    border: 2px solid var(--gold);
    background: var(--cream);
    display: flex;
    align-items: center;
    justify-content: center;
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    color: var(--gold);
    flex-shrink: 0;
    z-index: 1;
  }

  .stage-connector {
    width: 1px;
    flex: 1;
    background: var(--border);
    min-height: 24px;
  }

  .stage-content {
    padding: 0 0 32px 20px;
    flex: 1;
  }

  .stage-content h3 {
    margin-top: 2px;
    font-size: 1.1rem;
  }

  /* ─── QUERY EXAMPLES ─── */
  .query-examples {
    display: flex;
    flex-direction: column;
    gap: 12px;
    margin-top: 8px;
  }

  .query-row {
    display: grid;
    grid-template-columns: 1fr auto auto;
    gap: 12px;
    align-items: center;
    padding: 14px 18px;
    border: 1px solid var(--border);
    background: var(--parchment);
    transition: border-color 0.2s;
  }

  .query-row:hover {
    border-color: var(--gold);
  }

  .query-text {
    font-family: 'Cormorant Garamond', serif;
    font-style: italic;
    font-size: 1rem;
    color: var(--ink);
  }

  .query-tag {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    padding: 3px 10px;
    border: 1px solid var(--border);
    color: var(--muted);
    white-space: nowrap;
    background: var(--cream);
  }

  /* ─── MODELS GRID ─── */
  .model-list {
    display: flex;
    flex-direction: column;
    gap: 1px;
    background: var(--border);
    border: 1px solid var(--border);
  }

  .model-row {
    display: grid;
    grid-template-columns: 1fr 1fr 80px;
    background: var(--cream);
    padding: 16px 20px;
    gap: 16px;
    align-items: center;
    transition: background 0.15s;
  }

  .model-row:hover {
    background: var(--parchment);
  }

  .model-row .model-name {
    font-family: 'DM Mono', monospace;
    font-size: 0.75rem;
    color: var(--rust);
  }

  .model-row .model-role {
    font-size: 0.87rem;
    color: var(--muted);
  }

  .model-row .model-src {
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    color: var(--gold);
    text-align: right;
    opacity: 0.8;
  }

  /* ─── SETUP STEPS ─── */
  .setup-steps {
    counter-reset: step;
    display: flex;
    flex-direction: column;
    gap: 16px;
  }

  .setup-step {
    display: flex;
    gap: 16px;
    align-items: flex-start;
  }

  .step-num {
    counter-increment: step;
    width: 28px;
    height: 28px;
    background: var(--ink);
    color: var(--parchment);
    display: flex;
    align-items: center;
    justify-content: center;
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    flex-shrink: 0;
    margin-top: 2px;
  }

  .step-num::after { content: counter(step); }

  .step-body {
    flex: 1;
  }

  .step-body strong {
    display: block;
    font-family: 'Cormorant Garamond', serif;
    font-size: 1rem;
    font-weight: 600;
    color: var(--ink);
    margin-bottom: 4px;
  }

  /* ─── LIMITATIONS ─── */
  .limitation-list {
    display: flex;
    flex-direction: column;
    gap: 12px;
  }

  .limitation {
    padding: 16px 20px 16px 44px;
    border: 1px solid var(--border);
    background: var(--parchment);
    font-size: 0.88rem;
    color: var(--muted);
    position: relative;
    line-height: 1.6;
  }

  .limitation::before {
    content: attr(data-icon);
    position: absolute;
    left: 14px;
    top: 15px;
    font-size: 1rem;
  }

  /* ─── FOOTER ─── */
  footer {
    text-align: center;
    padding: 48px 0 0;
    border-top: 1px solid var(--border);
    font-family: 'Cormorant Garamond', serif;
    font-size: 0.95rem;
    color: var(--muted);
    font-style: italic;
  }

  footer .ornament {
    display: block;
    font-size: 1.5rem;
    color: var(--gold);
    margin-bottom: 16px;
    opacity: 0.5;
  }

  footer a {
    color: var(--gold);
    text-decoration: none;
  }

  /* ─── ANIMATIONS ─── */
  @keyframes fadeUp {
    from { opacity: 0; transform: translateY(18px); }
    to { opacity: 1; transform: translateY(0); }
  }

  .hero { animation: fadeUp 0.7s ease both; }
  section { animation: fadeUp 0.6s ease both; }
  section:nth-child(2) { animation-delay: 0.05s; }
  section:nth-child(3) { animation-delay: 0.1s; }
  section:nth-child(4) { animation-delay: 0.15s; }

  /* ─── RESPONSIVE ─── */
  @media (max-width: 640px) {
    .wrapper { padding: 40px 20px 80px; }
    .query-row { grid-template-columns: 1fr; }
    .model-row { grid-template-columns: 1fr 1fr; }
    .model-row .model-src { display: none; }
    .arch-box { padding: 24px 20px; font-size: 0.7rem; }
  }
</style>
</head>
<body>
<div class="wrapper">

  <!-- ─── HERO ─── -->
  <header class="hero">
    <h1>Book<span>Mind</span></h1>
    <p class="tagline">An AI that understands what you're in the mood for, not just what you search for.</p>
    <a class="live-link" href="https://huggingface.co/spaces/nayanasisil2700/Contextual-RAG-Book-Recommender" target="_blank">
      Try the live demo
    </a>
    <div class="badges">
      <span class="badge">Python 3.10+</span>
      <span class="badge">LangChain 0.2+</span>
      <span class="badge">ChromaDB</span>
      <span class="badge">HuggingFace Transformers</span>
      <span class="badge">Gradio UI</span>
      <span class="badge">6,810 books</span>
    </div>
  </header>

  <!-- ─── OVERVIEW ─── -->
  <section>
    <h2><span class="section-num">01</span> Overview</h2>
    <p>BookMind is an end-to-end <strong>Agentic Retrieval-Augmented Generation (RAG)</strong> system for book discovery. It goes beyond simple keyword matching by combining semantic vector search, emotion-aware filtering, cross-encoder reranking, and LLM-powered explanations — all orchestrated by an agent that can reflect on its own results and rewrite its query if needed.</p>
    <p>The system was built on a corpus of <strong>6,810 books</strong> (cleaned to ~5,197 usable entries) sourced from Google Books metadata.</p>
  </section>

  <!-- ─── FEATURES ─── -->
  <section>
    <h2><span class="section-num">02</span> Features</h2>
    <div class="feature-grid">
      <div class="feature-card">
        <span class="icon">🔍</span>
        <strong>Semantic Search</strong>
        <p>Queries are embedded with <code>all-MiniLM-L6-v2</code> and matched against book descriptions in ChromaDB.</p>
      </div>
      <div class="feature-card">
        <span class="icon">🎭</span>
        <strong>Emotion-Aware Filtering</strong>
        <p>Books are tagged across 6 emotional dimensions (joy, sadness, anger, fear, surprise, love) using a fine-tuned BERT model.</p>
      </div>
      <div class="feature-card">
        <span class="icon">📚</span>
        <strong>Zero-Shot Classification</strong>
        <p>Books without clear categories are classified as Fiction or Nonfiction using <code>facebook/bart-large-mnli</code>.</p>
      </div>
      <div class="feature-card">
        <span class="icon">🎯</span>
        <strong>Cross-Encoder Reranking</strong>
        <p>Candidate books are reranked using <code>ms-marco-MiniLM-L-6-v2</code> for precision beyond initial retrieval.</p>
      </div>
      <div class="feature-card">
        <span class="icon">🔄</span>
        <strong>Self-Reflection and Query Rewriting</strong>
        <p>The agent evaluates result quality and rewrites the query (up to 2 times) if scores fall below threshold.</p>
      </div>
      <div class="feature-card">
        <span class="icon">✨</span>
        <strong>LLM Explanations</strong>
        <p><code>flan-t5-base</code> generates a personalised "why this book" explanation for each recommendation.</p>
      </div>
      <div class="feature-card">
        <span class="icon">📡</span>
        <strong>LangSmith Observability</strong>
        <p>Full pipeline tracing and run metrics for every agent invocation.</p>
      </div>
      <div class="feature-card">
        <span class="icon">🖼</span>
        <strong>Gradio Dashboard</strong>
        <p>A polished, book-themed UI with live pipeline trace, reasoning sidebar, and a cover image gallery.</p>
      </div>
    </div>
  </section>

  <!-- ─── ARCHITECTURE ─── -->
  <section>
    <h2><span class="section-num">03</span> Architecture</h2>
    <div class="arch-box">
<span class="muted">User Query</span>
    │
    ▼
<span class="gold">┌─────────────────────────────────────────────────────────┐</span>
<span class="gold">│</span>                    RAG Agent Loop                       <span class="gold">│</span>
<span class="gold">│</span>                                                         <span class="gold">│</span>
<span class="gold">│</span>  <span class="green">1. Vector Search</span>      (ChromaDB + MiniLM embeddings)   <span class="gold">│</span>
<span class="gold">│</span>         │                                               <span class="gold">│</span>
<span class="gold">│</span>  <span class="green">2. Metadata Filter</span>    (category + emotional tone)      <span class="gold">│</span>
<span class="gold">│</span>         │                                               <span class="gold">│</span>
<span class="gold">│</span>  <span class="green">3. Cross-Encoder Rerank</span>  (ms-marco-MiniLM)             <span class="gold">│</span>
<span class="gold">│</span>         │                                               <span class="gold">│</span>
<span class="gold">│</span>  <span class="green">4. Self-Reflection</span>    (score threshold check)          <span class="gold">│</span>
<span class="gold">│</span>         │                                               <span class="gold">│</span>
<span class="gold">│</span>    satisfied? ──yes──► LLM Explain ──► Return results   <span class="gold">│</span>
<span class="gold">│</span>         │                                               <span class="gold">│</span>
<span class="gold">│</span>        no ──► Query Rewrite (flan-t5) ──► Retry         <span class="gold">│</span>
<span class="gold">│</span>                      (max 2 retries)                    <span class="gold">│</span>
<span class="gold">└─────────────────────────────────────────────────────────┘</span>
    </div>
  </section>

  <!-- ─── DATA PIPELINE ─── -->
  <section>
    <h2><span class="section-num">04</span> Data Pipeline</h2>
    <p>The dataset goes through four sequential preprocessing stages, starting from 6,810 raw books.</p>

    <div class="stages">
      <div class="stage">
        <div class="stage-line">
          <div class="stage-dot">01</div>
          <div class="stage-connector"></div>
        </div>
        <div class="stage-content">
          <h3>EDA and Cleaning</h3>
          <p>Rows missing <code>description</code>, <code>num_pages</code>, <code>average_rating</code>, or <code>published_year</code> were dropped. Books with fewer than 25 words in their description were removed to eliminate uninformative stubs. Title and subtitle were merged, and <code>age_of_book</code> was computed. <strong>Final cleaned corpus: 5,197 books.</strong></p>
        </div>
      </div>

      <div class="stage">
        <div class="stage-line">
          <div class="stage-dot">02</div>
          <div class="stage-connector"></div>
        </div>
        <div class="stage-content">
          <h3>Category Simplification and Zero-Shot Classification</h3>
          <p>The raw dataset contained <strong>531 unique category strings</strong>. These were mapped to simplified labels (Fiction, Nonfiction, Children's Fiction, etc.). For the remaining ~1,454 books with unmapped categories, <code>facebook/bart-large-mnli</code> classified each as Fiction or Nonfiction from its description. Accuracy validated at <strong>77.8%</strong> on a held-out set.</p>
        </div>
      </div>

      <div class="stage">
        <div class="stage-line">
          <div class="stage-dot">03</div>
          <div class="stage-connector"></div>
        </div>
        <div class="stage-content">
          <h3>Emotion Tagging</h3>
          <p>Each book's description was split into sentences and classified by <code>bert-base-uncased-emotion</code> across 6 emotion labels. Per-sentence scores were averaged to produce a book-level emotion vector, with a <code>dominant_emotion</code> column added using <code>idxmax</code>. This enables tone-based filtering in the UI.</p>
        </div>
      </div>

      <div class="stage">
        <div class="stage-line">
          <div class="stage-dot">04</div>
        </div>
        <div class="stage-content">
          <h3>Vector Indexing</h3>
          <p><code>tagged_description</code> strings (ISBN prefix + description) were embedded with <code>all-MiniLM-L6-v2</code> and stored in ChromaDB. Semantic search is performed at query time against this index.</p>
        </div>
      </div>
    </div>
  </section>

  <!-- ─── MODELS ─── -->
  <section>
    <h2><span class="section-num">05</span> Models Used</h2>
    <div class="model-list">
      <div class="model-row" style="background:var(--ink); color:var(--parchment);">
        <span style="font-family:'DM Mono',monospace;font-size:0.7rem;letter-spacing:0.08em;opacity:0.6;">MODEL</span>
        <span style="font-family:'DM Mono',monospace;font-size:0.7rem;letter-spacing:0.08em;opacity:0.6;">ROLE</span>
        <span style="font-family:'DM Mono',monospace;font-size:0.7rem;letter-spacing:0.08em;opacity:0.6;text-align:right;">SOURCE</span>
      </div>
      <div class="model-row">
        <span class="model-name">all-MiniLM-L6-v2</span>
        <span class="model-role">Query and document embeddings</span>
        <span class="model-src">HF</span>
      </div>
      <div class="model-row">
        <span class="model-name">bert-base-uncased-emotion</span>
        <span class="model-role">Emotion classification (6 labels)</span>
        <span class="model-src">HF</span>
      </div>
      <div class="model-row">
        <span class="model-name">bart-large-mnli</span>
        <span class="model-role">Zero-shot Fiction/Nonfiction classification</span>
        <span class="model-src">HF</span>
      </div>
      <div class="model-row">
        <span class="model-name">ms-marco-MiniLM-L-6-v2</span>
        <span class="model-role">Cross-encoder candidate reranking</span>
        <span class="model-src">HF</span>
      </div>
      <div class="model-row">
        <span class="model-name">flan-t5-base</span>
        <span class="model-role">Query rewriting and book explanations</span>
        <span class="model-src">HF</span>
      </div>
    </div>
    <p style="margin-top:12px;font-size:0.83rem;">All models run locally on CPU — no GPU required, though GPU will be faster for the sentiment and classification stages.</p>
  </section>

  <!-- ─── SETUP ─── -->
  <section>
    <h2><span class="section-num">06</span> Setup and Installation</h2>

    <div class="setup-steps">
      <div class="setup-step">
        <div class="step-num"></div>
        <div class="step-body">
          <strong>Prerequisites</strong>
          <p>Python 3.10+, ~4GB disk space for model downloads (first run only)</p>
        </div>
      </div>
      <div class="setup-step">
        <div class="step-num"></div>
        <div class="step-body">
          <strong>Install dependencies</strong>
          <pre>pip install -r requirements.txt</pre>
        </div>
      </div>
      <div class="setup-step">
        <div class="step-num"></div>
        <div class="step-body">
          <strong>Configure LangSmith (optional)</strong>
          <p>Create a <code>.env</code> file in the project root:</p>
          <pre>LANGCHAIN_API_KEY=your_langsmith_api_key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=bookmind-rag
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com</pre>
          <p style="margin-top:8px;">If no API key is provided, tracing is silently disabled and the app runs normally.</p>
        </div>
      </div>
      <div class="setup-step">
        <div class="step-num"></div>
        <div class="step-body">
          <strong>Run the dashboard</strong>
          <pre>python gradio_dashboard.py</pre>
          <p style="margin-top:8px;">Available at <code>http://127.0.0.1:7860</code>. On first launch, models (~1.5GB total) will download automatically from HuggingFace.</p>
        </div>
      </div>
    </div>

    <h3>Standalone usage</h3>
    <pre>from rag_agent.rag_agent import initialize, run_agent

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
print(result["reasoning"])</pre>
  </section>

  <!-- ─── EXAMPLE QUERIES ─── -->
  <section>
    <h2><span class="section-num">07</span> Example Queries</h2>
    <div class="query-examples">
      <div class="query-row">
        <span class="query-text">"a melancholic wartime love story with exquisite prose"</span>
        <span class="query-tag">Fiction</span>
        <span class="query-tag">Sad</span>
      </div>
      <div class="query-row">
        <span class="query-text">"books to teach children about nature and animals"</span>
        <span class="query-tag">Children's Fiction</span>
        <span class="query-tag">Happy</span>
      </div>
      <div class="query-row">
        <span class="query-text">"a gripping thriller with a female detective"</span>
        <span class="query-tag">Fiction</span>
        <span class="query-tag">Suspenseful</span>
      </div>
      <div class="query-row">
        <span class="query-text">"philosophy of consciousness and the nature of self"</span>
        <span class="query-tag">Nonfiction</span>
        <span class="query-tag">All</span>
      </div>
      <div class="query-row">
        <span class="query-text">"a story about redemption and second chances"</span>
        <span class="query-tag">All</span>
        <span class="query-tag">All</span>
      </div>
    </div>
  </section>

  <!-- ─── LIMITATIONS ─── -->
  <section>
    <h2><span class="section-num">08</span> Limitations and Future Work</h2>
    <div class="limitation-list">
      <div class="limitation" data-icon="⚠">
        <strong style="font-family:'Cormorant Garamond',serif;font-size:1rem;font-weight:600;color:var(--ink);display:block;margin-bottom:4px;">Small LLM explanations</strong>
        <code>flan-t5-base</code> is a small model; explanations can sometimes be generic. Upgrading to a larger instruction-tuned model would improve explanation quality.
      </div>
      <div class="limitation" data-icon="📊">
        <strong style="font-family:'Cormorant Garamond',serif;font-size:1rem;font-weight:600;color:var(--ink);display:block;margin-bottom:4px;">Zero-shot classification accuracy</strong>
        Currently ~78%. A fine-tuned classifier on book descriptions would meaningfully improve category assignment.
      </div>
      <div class="limitation" data-icon="⏱">
        <strong style="font-family:'Cormorant Garamond',serif;font-size:1rem;font-weight:600;color:var(--ink);display:block;margin-bottom:4px;">Reranking latency</strong>
        On CPU, reranking 80 candidates averages 2–3 seconds. Batching or a lighter cross-encoder model would reduce this.
      </div>
      <div class="limitation" data-icon="🗺">
        <strong style="font-family:'Cormorant Garamond',serif;font-size:1rem;font-weight:600;color:var(--ink);display:block;margin-bottom:4px;">Future directions</strong>
        User preference memory, collaborative filtering signals, and multi-turn conversational recommendations.
      </div>
    </div>
  </section>

  <!-- ─── FOOTER ─── -->
  <footer>
    <span class="ornament">❦</span>
    Built with <a href="https://huggingface.co" target="_blank">HuggingFace</a> models and LangChain. MIT License.
    <br><br>
    <a href="https://huggingface.co/spaces/nayanasisil2700/Contextual-RAG-Book-Recommender" target="_blank" style="display:inline-block;margin-top:16px;font-family:'DM Mono',monospace;font-size:0.75rem;letter-spacing:0.1em;padding:10px 24px;border:1px solid var(--border);color:var(--muted);text-decoration:none;transition:all 0.2s;" onmouseover="this.style.borderColor='var(--gold)';this.style.color='var(--gold)'" onmouseout="this.style.borderColor='var(--border)';this.style.color='var(--muted)'">
      ▶ huggingface.co/spaces/nayanasisil2700/Contextual-RAG-Book-Recommender
    </a>
  </footer>

</div>
</body>
</html>
