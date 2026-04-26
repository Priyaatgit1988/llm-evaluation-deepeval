# 🧪 LLM Evaluation Framework with DeepEval

> A complete end-to-end project: **E-Commerce Chatbot** + **RAG Pipeline** + **DeepEval Evaluation Dashboard** — all in one unified interface.

Built to demonstrate how to systematically evaluate LLM-powered applications using [DeepEval](https://github.com/confident-ai/deepeval) with 15+ metrics, multiple judge LLMs, and a live dashboard for visualizing results.

---

## 📋 Table of Contents

- [Architecture](#-architecture)
- [Project Structure](#-project-structure)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Setup & Installation](#-setup--installation)
- [Running the Project](#-running-the-project)
- [DeepEval Metrics](#-deepeval-metrics-15)
- [Dashboard Guide](#-dashboard-guide)
- [Judge LLM Configuration](#-judge-llm-configuration)
- [Screenshots](#-screenshots)

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     UNIFIED DASHBOARD (:8501)                       │
│  ┌──────────┐  ┌──────────────┐  ┌──────────┐  ┌───────────────┐  │
│  │ Dashboard │  │   Chatbot    │  │   RAG    │  │  Evaluations  │  │
│  │ Overview  │  │   Live Chat  │  │ Explorer │  │  15+ Metrics  │  │
│  └──────────┘  └──────┬───────┘  └────┬─────┘  └───────┬───────┘  │
└─────────────────────────┼──────────────┼────────────────┼──────────┘
                          │              │                │
                          ▼              ▼                ▼
              ┌───────────────┐  ┌──────────────┐  ┌──────────────┐
              │  E-Commerce   │  │     RAG      │  │   DeepEval   │
              │   Chatbot     │  │   Pipeline   │  │  Framework   │
              │  (React:3000) │  │ (Flask:5001) │  │  (15 metrics)│
              └───────────────┘  └──────┬───────┘  └──────┬───────┘
                                        │                 │
                          ┌─────────────┼─────────────────┘
                          ▼             ▼
                   ┌─────────────┐  ┌──────────┐
                   │  ChromaDB   │  │   Groq   │
                   │ (Vector DB) │  │  (Judge) │
                   └──────┬──────┘  └──────────┘
                          │
                   ┌──────┴──────┐
                   │ Nomic Embed │
                   │  v1.5 768d  │
                   └─────────────┘
```

### RAG Pipeline Flow

```
 ┌──────┐     ┌──────┐     ┌──────┐     ┌──────────┐     ┌────────┐
 │Ingest│────▶│Embed │────▶│Store │────▶│ Retrieve │────▶│ Answer │
 │.txt  │     │Nomic │     │Chroma│     │ Top-K    │     │  Groq  │
 │.pdf  │     │768-d │     │  DB  │     │ Semantic │     │  LLM   │
 └──────┘     └──────┘     └──────┘     └──────────┘     └────────┘
```

---

## 📁 Project Structure

```
llm-evaluation-deepeval/
│
├── ecommerce-chatbot/          # React chatbot application
│   ├── public/
│   │   └── index.html
│   ├── src/
│   │   ├── App.js              # Main app with product catalog + chat
│   │   ├── App.css             # Styling
│   │   ├── index.js
│   │   └── components/
│   │       ├── ChatWindow.js   # Chat UI component
│   │       └── ProductCatalog.js
│   └── package.json
│
├── rag-explorer/               # RAG pipeline with ChromaDB
│   ├── data/
│   │   ├── ecommerce_faq.txt   # Knowledge base document
│   │   └── product_policies.txt
│   ├── app.py                  # Flask API + HTML dashboard
│   ├── embeddings.py           # Nomic Embed integration
│   ├── ingest.py               # Document chunking & ingestion
│   ├── rag_chain.py            # Retrieve + Generate pipeline
│   ├── requirements.txt
│   └── .env.example
│
├── deepeval-framework/         # Evaluation framework + dashboard
│   ├── dashboard.py            # Unified dashboard (all-in-one)
│   ├── config.py               # LLM provider configuration
│   ├── custom_model.py         # DeepEval model wrapper for Groq
│   ├── llm_providers.py        # Multi-provider LLM abstraction
│   ├── test_data.py            # 20 test cases (10 chatbot + 10 RAG)
│   ├── test_chatbot_eval.py    # Pytest-based chatbot evaluations
│   ├── test_rag_eval.py        # Pytest-based RAG evaluations
│   ├── run_eval.py             # CLI evaluation runner
│   ├── conftest.py
│   ├── requirements.txt
│   └── .env.example
│
├── .gitignore
└── README.md
```

---

## ✨ Features

### 🛒 E-Commerce Chatbot
- React-based customer support chatbot
- Product catalog with 8 products across 5 categories
- Handles: shipping, returns, payments, discounts, order tracking, product search
- Rule-based responses for consistent evaluation

### 🔍 RAG Explorer
- Full Retrieval-Augmented Generation pipeline
- **Nomic Embed v1.5** (768-dim) for document embeddings
- **ChromaDB** for local persistent vector storage
- **Groq LLM** for answer generation grounded in retrieved context
- Document ingestion from `.txt` and `.pdf` files
- 21 chunks from 2 e-commerce knowledge base documents

### 📊 DeepEval Dashboard (Unified)
- **4 tabs in one dashboard**: Dashboard Overview, Chatbot, RAG Explorer, Evaluations
- **Live progress tracking**: animated progress bar, current metric/target chips, done/total counter
- **15 evaluation metrics** with per-test-case pass/fail dots
- **Score vs Threshold** comparison on every metric card
- **Fine-tuning suggestions** auto-generated for any metric with failures
- **Drill-down tables** with failed rows highlighted in red, full justification text
- **Both Chatbot & RAG** results shown per metric when available

---

## 🛠 Tech Stack

| Component | Technology |
|-----------|-----------|
| Chatbot Frontend | React 18 |
| RAG Backend | Flask, ChromaDB, sentence-transformers |
| Embeddings | Nomic Embed v1.5 (768-dim) |
| LLM (Generation) | Groq — Llama 4 Scout 17B |
| LLM (Judge) | Groq — Llama 4 Scout 17B (configurable) |
| Evaluation | DeepEval 3.9+ |
| Vector DB | ChromaDB (local persistent) |
| Dashboard | Flask + vanilla JS (single-page app) |
| Testing | Pytest + DeepEval assertions |

---

## 🚀 Setup & Installation

### Prerequisites
- Python 3.10+
- Node.js 18+
- A [Groq API key](https://console.groq.com/) (free tier works)

### 1. Clone the repository
```bash
git clone https://github.com/Priyaatgit1988/llm-evaluation-deepeval.git
cd llm-evaluation-deepeval
```

### 2. Install E-Commerce Chatbot
```bash
cd ecommerce-chatbot
npm install
cd ..
```

### 3. Install RAG Explorer
```bash
cd rag-explorer
pip install -r requirements.txt
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
cd ..
```

### 4. Install DeepEval Framework
```bash
cd deepeval-framework
pip install -r requirements.txt
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
cd ..
```

### 5. Ingest documents into ChromaDB
```bash
cd rag-explorer
python ingest.py
cd ..
```

---

## ▶ Running the Project

### Start all 3 services:

```bash
# Terminal 1 — Chatbot (port 3000)
cd ecommerce-chatbot && npm start

# Terminal 2 — RAG Explorer (port 5001)
cd rag-explorer && python app.py

# Terminal 3 — Unified Dashboard (port 8501)
cd deepeval-framework && python dashboard.py
```

### Open the dashboard:
```
http://localhost:8501
```

From the dashboard you can:
- Chat with the bot (Chatbot tab)
- Query the RAG pipeline and browse chunks (RAG Explorer tab)
- Run all 15 metrics against both systems (Evaluations tab)

### CLI evaluation (alternative):
```bash
cd deepeval-framework

# Run all evaluations
python run_eval.py

# Run specific target
python run_eval.py --target chatbot
python run_eval.py --target rag

# Run specific metric
python run_eval.py --metric toxicity

# List all metrics
python run_eval.py --list-metrics
```

---

## 📏 DeepEval Metrics (15)

| # | Metric | Category | What It Measures |
|---|--------|----------|-----------------|
| 1 | **Answer Relevancy** | Quality | Is the response on-topic for the question? |
| 2 | **Faithfulness** | Quality | Are all claims backed by the retrieval context? |
| 3 | **Hallucination** | Quality | Does the response contradict the ground truth? |
| 4 | **Toxicity** | Safety | Is the response free of harmful language? |
| 5 | **Bias** | Safety | Is the response free of prejudiced statements? |
| 6 | **Contextual Precision** | Retrieval | Is relevant context ranked higher than irrelevant? |
| 7 | **Contextual Recall** | Retrieval | Does retrieved context cover the expected answer? |
| 8 | **Contextual Relevancy** | Retrieval | Is all retrieved context relevant to the query? |
| 9 | **Correctness** | G-Eval | Does the output match expected output factually? |
| 10 | **Coherence** | G-Eval | Is the response logically structured? |
| 11 | **Completeness** | G-Eval | Does it cover all key points? |
| 12 | **Conciseness** | G-Eval | Is it brief without unnecessary verbosity? |
| 13 | **Helpfulness** | G-Eval | How useful is it for the customer? |
| 14 | **Politeness** | G-Eval | Does it maintain professional tone? |
| 15 | **Safety** | Safety | Does it avoid inappropriate content? |

Each metric is evaluated against a **threshold of 0.50**. Inverse metrics (Toxicity, Bias, Hallucination) pass when the score is **below** the threshold.

---

## 🖥 Dashboard Guide

### Dashboard Tab
Pipeline overview showing the 5-stage RAG flow (Ingest → Embed → Store → Retrieve → Answer), vector store stats, and service endpoints.

### Chatbot Tab
Live chat interface with the ShopSmart bot. Quick-topic buttons for common queries. Responses are the same ones evaluated by DeepEval.

### RAG Explorer Tab
- **Left panel**: Query the RAG pipeline — see the LLM answer + retrieved source chunks with similarity scores
- **Right panel**: Browse all 21 document chunks stored in ChromaDB

### Evaluations Tab
- **Run controls**: "Run All" (300 evaluations), "Chatbot Only" (150), or "RAG Only" (150)
- **Live progress bar**: Shows current metric, target, and completion percentage in real-time
- **Metric cards**: Each card shows:
  - Average score vs threshold
  - Pass/fail count with colored pills
  - Per-test-case dots (green = pass, red = fail)
  - Cross-target summary (Chatbot + RAG)
- **Click a card** to drill down into per-test-case results with:
  - Score vs threshold comparison
  - Pass/Fail status with ✓/✗ icons
  - Failed rows highlighted in red
  - Full justification/reason from the judge LLM
- **Fine-tuning suggestions**: Auto-generated for any metric with failures

---

## ⚙ Judge LLM Configuration

The framework supports multiple judge LLMs. Configure in `deepeval-framework/.env`:

```env
JUDGE_LLM=groq              # Default
GROQ_API_KEY=your-key-here
```

| Key | Model | Provider | Tokens/min |
|-----|-------|----------|-----------|
| `groq` | Llama 4 Scout 17B | Groq Cloud | 30,000 |
| `groq_oss120b` | GPT-OSS 120B | Groq Cloud | varies |
| `groq_qwen` | Qwen3 32B | Groq Cloud | 6,000 |
| `openai` | GPT-4o | OpenAI | unlimited* |
| `grok` | Grok-3-mini | xAI | varies |
| `gemma` | Gemma 3 1B | Local (Ollama) | unlimited |

Switch judge via CLI:
```bash
python run_eval.py --judge groq_oss120b
python run_eval.py --judge openai
```

---

## 📸 Screenshots

### Dashboard Overview
The main dashboard shows the RAG pipeline flow, vector store stats (21 chunks in ChromaDB), service endpoints, and quick-launch buttons.

### Chatbot Interface
Live chat with ShopBot — handles product queries, shipping, returns, discounts, and order tracking with quick-topic buttons.

### RAG Explorer
Query the knowledge base and see LLM-generated answers grounded in retrieved document chunks with similarity scores.

### Evaluation Metrics
15 metric cards with live progress, per-test-case pass/fail dots, score vs threshold comparison, and auto-generated fine-tuning suggestions for failed metrics.

---

## 📄 License

MIT

---

## 🙏 Acknowledgments

- [DeepEval](https://github.com/confident-ai/deepeval) — LLM evaluation framework
- [Groq](https://groq.com/) — Fast LLM inference
- [ChromaDB](https://www.trychroma.com/) — Vector database
- [Nomic AI](https://www.nomic.ai/) — Embedding model
- [sentence-transformers](https://www.sbert.net/) — Embedding library
