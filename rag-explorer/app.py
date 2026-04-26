"""
Flask API server for RAG Explorer with HTML frontend.
Endpoints:
  GET  /                  — HTML UI
  POST /api/ingest        — Ingest documents
  POST /api/query         — RAG query
  GET  /api/chunks        — List all chunks
  GET  /api/health        — Health check
"""
import os
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

HTML_TEMPLATE = None  # loaded below


@app.route("/")
def index():
    return render_template_string(get_html())


@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "message": "RAG Explorer is running"})


@app.route("/api/ingest", methods=["POST"])
def ingest():
    try:
        from ingest import ingest_documents
        chunks = ingest_documents()
        return jsonify({
            "status": "success",
            "chunks_ingested": len(chunks),
            "message": f"Successfully ingested {len(chunks)} chunks"
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/query", methods=["POST"])
def query():
    data = request.get_json()
    if not data or "query" not in data:
        return jsonify({"error": "Missing 'query' field"}), 400

    try:
        from rag_chain import rag_query
        result = rag_query(data["query"])
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/chunks")
def list_chunks():
    try:
        import chromadb
        client = chromadb.PersistentClient(path="./chroma_db")
        collection = client.get_collection("ecommerce_docs")
        results = collection.get(include=["documents", "metadatas"])
        chunks = []
        for i in range(len(results["ids"])):
            chunks.append({
                "id": results["ids"][i],
                "text": results["documents"][i],
                "source": results["metadatas"][i].get("source", "unknown"),
                "chunk_index": results["metadatas"][i].get("chunk_index", 0),
            })
        return jsonify({"chunks": chunks, "total": len(chunks)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def get_html():
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>RAG Explorer - ShopSmart</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #0f172a; color: #e2e8f0; min-height: 100vh; }
.header { background: linear-gradient(135deg, #1e293b 0%, #334155 100%); padding: 20px 32px; border-bottom: 1px solid #334155; display: flex; align-items: center; justify-content: space-between; }
.header h1 { font-size: 1.5rem; color: #38bdf8; }
.header .subtitle { color: #94a3b8; font-size: 0.9rem; }
.header .status { display: flex; align-items: center; gap: 8px; }
.status-dot { width: 10px; height: 10px; border-radius: 50%; background: #22c55e; }
.container { max-width: 1400px; margin: 0 auto; padding: 24px; display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }
.panel { background: #1e293b; border-radius: 12px; border: 1px solid #334155; overflow: hidden; }
.panel-header { padding: 16px 20px; border-bottom: 1px solid #334155; font-weight: 600; display: flex; align-items: center; gap: 8px; }
.panel-body { padding: 20px; }
.full-width { grid-column: 1 / -1; }
.btn { padding: 10px 20px; border: none; border-radius: 8px; font-weight: 600; cursor: pointer; font-size: 0.9rem; transition: all 0.2s; }
.btn-primary { background: #3b82f6; color: white; }
.btn-primary:hover { background: #2563eb; }
.btn-success { background: #22c55e; color: white; }
.btn-success:hover { background: #16a34a; }
input[type="text"], textarea { width: 100%; padding: 10px 14px; border: 1px solid #475569; border-radius: 8px; background: #0f172a; color: #e2e8f0; font-size: 0.9rem; outline: none; }
input[type="text"]:focus, textarea:focus { border-color: #3b82f6; }
.query-form { display: flex; gap: 8px; }
.query-form input { flex: 1; }
.chunk-card { background: #0f172a; border: 1px solid #334155; border-radius: 8px; padding: 14px; margin-bottom: 10px; }
.chunk-card .source { color: #38bdf8; font-size: 0.8rem; font-weight: 600; margin-bottom: 6px; }
.chunk-card .text { color: #cbd5e1; font-size: 0.85rem; line-height: 1.6; }
.chunk-card .meta { color: #64748b; font-size: 0.75rem; margin-top: 8px; }
.answer-box { background: #0f172a; border: 1px solid #334155; border-radius: 8px; padding: 16px; margin-top: 12px; line-height: 1.7; }
.answer-box .label { color: #38bdf8; font-weight: 600; margin-bottom: 8px; }
.sources-list { margin-top: 12px; }
.source-item { background: #1e293b; border-left: 3px solid #3b82f6; padding: 10px 14px; margin-bottom: 8px; border-radius: 0 8px 8px 0; font-size: 0.85rem; }
.source-item .dist { color: #f59e0b; font-size: 0.75rem; }
.loading { color: #94a3b8; font-style: italic; }
.stats { display: flex; gap: 16px; margin-bottom: 16px; }
.stat { background: #0f172a; border: 1px solid #334155; border-radius: 8px; padding: 12px 16px; flex: 1; text-align: center; }
.stat .num { font-size: 1.5rem; font-weight: 700; color: #38bdf8; }
.stat .label { font-size: 0.8rem; color: #94a3b8; }
</style>
</head>
<body>
<div class="header">
  <div>
    <h1>🔍 RAG Explorer</h1>
    <div class="subtitle">ShopSmart Knowledge Base — Powered by Nomic Embed + ChromaDB + Grok</div>
  </div>
  <div class="status"><div class="status-dot"></div><span style="color:#94a3b8;font-size:0.85rem">Connected</span></div>
</div>
<div class="container">
  <!-- Ingest Panel -->
  <div class="panel">
    <div class="panel-header">📥 Document Ingestion</div>
    <div class="panel-body">
      <p style="color:#94a3b8;margin-bottom:12px;font-size:0.9rem">Ingest documents from the data/ folder into ChromaDB</p>
      <button class="btn btn-success" onclick="ingestDocs()">Ingest Documents</button>
      <div id="ingest-result" style="margin-top:12px"></div>
    </div>
  </div>
  <!-- Query Panel -->
  <div class="panel">
    <div class="panel-header">💬 RAG Query</div>
    <div class="panel-body">
      <div class="query-form">
        <input type="text" id="query-input" placeholder="Ask about shipping, returns, products..." onkeydown="if(event.key==='Enter')doQuery()">
        <button class="btn btn-primary" onclick="doQuery()">Ask</button>
      </div>
      <div id="query-result"></div>
    </div>
  </div>
  <!-- Chunks Panel -->
  <div class="panel full-width">
    <div class="panel-header">📄 Document Chunks <button class="btn btn-primary" style="margin-left:auto;padding:6px 14px;font-size:0.8rem" onclick="loadChunks()">Refresh</button></div>
    <div class="panel-body">
      <div id="stats" class="stats"></div>
      <div id="chunks-list" style="max-height:400px;overflow-y:auto"></div>
    </div>
  </div>
</div>
<script>
async function ingestDocs() {
  const el = document.getElementById('ingest-result');
  el.innerHTML = '<span class="loading">Ingesting documents...</span>';
  try {
    const res = await fetch('/api/ingest', { method: 'POST' });
    const data = await res.json();
    el.innerHTML = data.status === 'success'
      ? `<span style="color:#22c55e">✅ ${data.message}</span>`
      : `<span style="color:#ef4444">❌ ${data.message}</span>`;
    loadChunks();
  } catch(e) { el.innerHTML = `<span style="color:#ef4444">Error: ${e.message}</span>`; }
}
async function doQuery() {
  const q = document.getElementById('query-input').value.trim();
  if (!q) return;
  const el = document.getElementById('query-result');
  el.innerHTML = '<span class="loading">Searching...</span>';
  try {
    const res = await fetch('/api/query', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({query: q}) });
    const data = await res.json();
    let html = `<div class="answer-box"><div class="label">Answer</div>${data.answer}</div>`;
    if (data.sources && data.sources.length) {
      html += '<div class="sources-list"><div style="color:#94a3b8;font-size:0.85rem;margin-bottom:8px">Sources:</div>';
      data.sources.forEach(s => {
        html += `<div class="source-item"><strong>${s.source}</strong> <span class="dist">(distance: ${s.distance.toFixed(4)})</span><br>${s.text.substring(0,200)}...</div>`;
      });
      html += '</div>';
    }
    el.innerHTML = html;
  } catch(e) { el.innerHTML = `<span style="color:#ef4444">Error: ${e.message}</span>`; }
}
async function loadChunks() {
  try {
    const res = await fetch('/api/chunks');
    const data = await res.json();
    if (data.error) { document.getElementById('chunks-list').innerHTML = `<span style="color:#ef4444">${data.error}</span>`; return; }
    const sources = {};
    data.chunks.forEach(c => { sources[c.source] = (sources[c.source]||0)+1; });
    document.getElementById('stats').innerHTML = `<div class="stat"><div class="num">${data.total}</div><div class="label">Total Chunks</div></div>` +
      Object.entries(sources).map(([s,n]) => `<div class="stat"><div class="num">${n}</div><div class="label">${s}</div></div>`).join('');
    document.getElementById('chunks-list').innerHTML = data.chunks.map(c =>
      `<div class="chunk-card"><div class="source">📄 ${c.source}</div><div class="text">${c.text}</div><div class="meta">Chunk #${c.chunk_index} | ID: ${c.id}</div></div>`
    ).join('');
  } catch(e) { document.getElementById('chunks-list').innerHTML = '<span style="color:#94a3b8">Click "Ingest Documents" first, then refresh.</span>'; }
}
loadChunks();
</script>
</body>
</html>"""


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port, debug=True)
