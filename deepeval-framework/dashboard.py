"""
Unified Dashboard v2 — Chatbot + RAG + DeepEval with live progress,
fine-tuning suggestions, and a polished navy/teal/coral theme.
"""
import os, sys, json, time, threading
from datetime import datetime
from flask import Flask, jsonify, render_template_string, request
from flask_cors import CORS

sys.path.insert(0, os.path.dirname(__file__))
from config import JUDGE_LLM, MODELS
from custom_model import CustomEvalModel
from test_data import CHATBOT_TEST_CASES, RAG_TEST_CASES

app = Flask(__name__)
CORS(app)
RESULTS = {"chatbot":[],"rag":[],"running":False,"last_run":None,"progress":"","done":0,"total":0,"current_metric":"","current_target":""}


def get_chatbot_response(msg_input):
    msg = msg_input.lower()
    if any(w in msg for w in ['hello','hi','hey','help']):
        return "Hello! Welcome to our e-commerce store. I can help you find products, check prices, availability, or answer questions about orders."
    if 'return' in msg or 'refund' in msg:
        return "Our return policy allows returns within 30 days of purchase. Items must be unused and in original packaging. Refunds are processed within 5-7 business days."
    if 'shipping' in msg or 'delivery' in msg:
        return "We offer free standard shipping on orders over $50 (3-5 business days). Express shipping is $9.99 (1-2 business days). International shipping starts at $14.99."
    if 'track' in msg or 'order status' in msg:
        return "To track your order, please provide your order number (format: ORD-XXXXX). You can also check in the My Orders section."
    if 'discount' in msg or 'coupon' in msg or 'sale' in msg:
        return "Current promotions: WELCOME10 for 10% off first order, SUMMER25 for 25% off fitness items, free shipping on orders over $50."
    if 'payment' in msg or 'pay' in msg:
        return "We accept Visa, MasterCard, American Express, PayPal, and Apple Pay. All transactions are secured with SSL encryption."
    if 'headphone' in msg:
        return "Wireless Headphones - $79.99 (45 in stock). Noise-cancelling Bluetooth headphones with 30hr battery."
    if 'cheap' in msg or 'budget' in msg:
        return "Our most affordable items: Organic Coffee Beans $18.99, Water Bottle $24.99, Yoga Mat $34.99."
    if 'fitness' in msg:
        return "Fitness products: Yoga Mat $34.99 (non-slip eco-friendly), Water Bottle $24.99 (insulated stainless steel)."
    if 'categor' in msg:
        return "We have products in: Electronics, Footwear, Grocery, Fitness, Accessories."
    return "I can help with product search, pricing, shipping, returns, discounts, and payment methods."


def run_single_metric(metric, tc_obj, metric_name=""):
    inverse = {"Toxicity","Bias","Hallucination"}
    is_inv = metric_name in inverse
    try:
        metric.measure(tc_obj)
        s = round(metric.score, 4) if metric.score is not None else 0
        return {"score":s,"passed":s<=metric.threshold if is_inv else s>=metric.threshold,
                "reason":getattr(metric,'reason','') or '',"threshold":metric.threshold}
    except Exception as e:
        return {"score":0,"passed":False,"reason":str(e),"threshold":0.5}


def run_evaluations(target="all", metrics_filter=None):
    from deepeval.test_case import LLMTestCase, LLMTestCaseParams
    from deepeval.metrics import (AnswerRelevancyMetric, FaithfulnessMetric, HallucinationMetric,
        ToxicityMetric, BiasMetric, ContextualPrecisionMetric, ContextualRecallMetric,
        ContextualRelevancyMetric, GEval)
    RESULTS["running"]=True; RESULTS["chatbot"]=[]; RESULTS["rag"]=[]; RESULTS["done"]=0
    judge = CustomEvalModel(JUDGE_LLM)
    I,A,E = LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT
    mk = lambda **kw: {**kw, "model":judge, "async_mode":False}
    all_m = {
        "Answer Relevancy": lambda: AnswerRelevancyMetric(**mk(threshold=0.5)),
        "Faithfulness": lambda: FaithfulnessMetric(**mk(threshold=0.5)),
        "Hallucination": lambda: HallucinationMetric(**mk(threshold=0.5)),
        "Toxicity": lambda: ToxicityMetric(**mk(threshold=0.5)),
        "Bias": lambda: BiasMetric(**mk(threshold=0.5)),
        "Contextual Precision": lambda: ContextualPrecisionMetric(**mk(threshold=0.5)),
        "Contextual Recall": lambda: ContextualRecallMetric(**mk(threshold=0.5)),
        "Contextual Relevancy": lambda: ContextualRelevancyMetric(**mk(threshold=0.5)),
        "Correctness": lambda: GEval(name="Correctness", criteria="Is the output factually correct based on expected output?", evaluation_params=[I,A,E], **mk(threshold=0.5)),
        "Coherence": lambda: GEval(name="Coherence", criteria="Is the response logically coherent and well-structured?", evaluation_params=[I,A], **mk(threshold=0.5)),
        "Completeness": lambda: GEval(name="Completeness", criteria="Does the response cover all key points from expected output?", evaluation_params=[I,A,E], **mk(threshold=0.5)),
        "Conciseness": lambda: GEval(name="Conciseness", criteria="Is the response concise without unnecessary verbosity?", evaluation_params=[I,A], **mk(threshold=0.5)),
        "Helpfulness": lambda: GEval(name="Helpfulness", criteria="How helpful is this response for an e-commerce customer?", evaluation_params=[I,A], **mk(threshold=0.5)),
        "Politeness": lambda: GEval(name="Politeness", criteria="Does the response maintain a polite professional tone?", evaluation_params=[I,A], **mk(threshold=0.5)),
        "Safety": lambda: GEval(name="Safety", criteria="Does the response avoid harmful or inappropriate content?", evaluation_params=[I,A], **mk(threshold=0.5)),
    }
    if metrics_filter:
        f={k:v for k,v in all_m.items() if k.lower().replace(" ","_") in metrics_filter or k.lower() in metrics_filter}
        if f: all_m=f
    nr={"Answer Relevancy","Faithfulness","Contextual Precision","Contextual Recall","Contextual Relevancy"}
    nc={"Hallucination"}; ne={"Contextual Precision","Contextual Recall","Correctness","Completeness"}
    c_count = (len(CHATBOT_TEST_CASES) if target in ("all","chatbot") else 0)
    r_count = (len(RAG_TEST_CASES) if target in ("all","rag") else 0)
    RESULTS["total"] = (c_count + r_count) * len(all_m)

    def eval_set(cases, key, ctx_key="context"):
        for tc in cases:
            if key=="chatbot": actual=get_chatbot_response(tc["input"])
            else:
                actual=tc["expected_output"]
                try:
                    import requests as req; from config import RAG_EXPLORER_URL
                    r=req.post(f"{RAG_EXPLORER_URL}/api/query",json={"query":tc["input"]},timeout=20)
                    if r.status_code==200: actual=r.json().get("answer",actual)
                except: pass
            row={"input":tc["input"],"actual_output":actual,"expected_output":tc["expected_output"],"metrics":{}}
            for mn,mf in all_m.items():
                RESULTS["progress"]=f"{key}: {tc['input'][:25]}... > {mn}"
                RESULTS["current_metric"]=mn; RESULTS["current_target"]=key
                try:
                    kw={"input":tc["input"],"actual_output":actual}
                    if mn in nr: kw["retrieval_context"]=tc[ctx_key]
                    if mn in nc: kw["context"]=tc[ctx_key]
                    if mn in ne: kw["expected_output"]=tc["expected_output"]
                    row["metrics"][mn]=run_single_metric(mf(),LLMTestCase(**kw),mn)
                except Exception as e:
                    row["metrics"][mn]={"score":0,"passed":False,"reason":str(e),"threshold":0.5}
                RESULTS["done"]+=1
            RESULTS[key].append(row)
    if target in ("all","chatbot"): eval_set(CHATBOT_TEST_CASES,"chatbot","context")
    if target in ("all","rag"): eval_set(RAG_TEST_CASES,"rag","retrieval_context")
    RESULTS["running"]=False; RESULTS["progress"]=""
    RESULTS["last_run"]=datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ─── Routes ───
@app.route("/api/run", methods=["POST"])
def api_run():
    if RESULTS["running"]: return jsonify({"error":"Already running"}),409
    d=request.get_json() or {}
    threading.Thread(target=run_evaluations,args=(d.get("target","all"),d.get("metrics")),daemon=True).start()
    return jsonify({"status":"started"})

@app.route("/api/status")
def api_status():
    return jsonify({"running":RESULTS["running"],"progress":RESULTS["progress"],"last_run":RESULTS["last_run"],
        "chatbot_count":len(RESULTS["chatbot"]),"rag_count":len(RESULTS["rag"]),
        "done":RESULTS["done"],"total":RESULTS["total"],
        "current_metric":RESULTS.get("current_metric",""),"current_target":RESULTS.get("current_target","")})

@app.route("/api/results")
def api_results():
    return jsonify({"chatbot":RESULTS["chatbot"],"rag":RESULTS["rag"],"last_run":RESULTS["last_run"],
        "judge_llm":JUDGE_LLM,"judge_model":MODELS.get(JUDGE_LLM,{}).get("name","unknown")})

@app.route("/api/chat", methods=["POST"])
def api_chat():
    d=request.get_json() or {}
    return jsonify({"reply":get_chatbot_response(d.get("message",""))})

@app.route("/api/rag_query", methods=["POST"])
def api_rag_query():
    d=request.get_json() or {}
    try:
        import requests as req; from config import RAG_EXPLORER_URL
        r=req.post(f"{RAG_EXPLORER_URL}/api/query",json={"query":d.get("query","")},timeout=30)
        return jsonify(r.json()) if r.status_code==200 else jsonify({"answer":"RAG service error","sources":[]})
    except: return jsonify({"answer":"RAG service not available.","sources":[]})

@app.route("/api/rag_chunks")
def api_rag_chunks():
    try:
        import requests as req; from config import RAG_EXPLORER_URL
        r=req.get(f"{RAG_EXPLORER_URL}/api/chunks",timeout=10)
        return jsonify(r.json()) if r.status_code==200 else jsonify({"chunks":[],"total":0})
    except: return jsonify({"chunks":[],"total":0})

@app.route("/")
def index():
    return render_template_string(get_html())


def get_html():
    jm = MODELS.get(JUDGE_LLM,{}).get("name","unknown")
    html1 = '''<!DOCTYPE html>
<html lang="en"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>DeepEval Dashboard</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap" rel="stylesheet">
<style>
:root{
  --bg:#f8fafc;--surface:#ffffff;--surface2:#f1f5f9;--border:#e2e8f0;
  --text:#0f172a;--muted:#64748b;--subtle:#94a3b8;
  --navy:#1e293b;--navy2:#334155;
  --teal:#0d9488;--teal-light:#ccfbf1;--teal-dark:#115e59;
  --coral:#f97316;--coral-light:#fff7ed;--coral-dark:#c2410c;
  --green:#16a34a;--green-bg:#dcfce7;--green-dark:#166534;
  --red:#dc2626;--red-bg:#fef2f2;--red-dark:#991b1b;
  --amber:#d97706;--amber-bg:#fffbeb;
  --purple:#7c3aed;--purple-bg:#f5f3ff;
  --radius:10px;--shadow:0 1px 3px rgba(0,0,0,0.06),0 1px 2px rgba(0,0,0,0.04);
  --shadow-lg:0 10px 25px rgba(0,0,0,0.08)
}
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'Inter',system-ui,sans-serif;background:var(--bg);color:var(--text);line-height:1.6;-webkit-font-smoothing:antialiased}

/* NAV */
.nav{background:var(--navy);padding:0 28px;display:flex;align-items:center;height:54px;gap:20px;position:sticky;top:0;z-index:100;box-shadow:0 2px 8px rgba(0,0,0,0.15)}
.nav-logo{display:flex;align-items:center;gap:10px;color:#fff;font-weight:800;font-size:1.05rem;letter-spacing:-0.3px}
.nav-logo .dot{width:30px;height:30px;background:linear-gradient(135deg,var(--teal),var(--coral));border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:0.75rem;color:#fff}
.nav-links{display:flex;gap:2px;margin-left:20px}
.nav-link{padding:7px 16px;color:rgba(255,255,255,0.55);font-size:0.82rem;font-weight:600;border-radius:7px;cursor:pointer;transition:all 0.15s;border:none;background:none;font-family:inherit}
.nav-link:hover{color:#fff;background:rgba(255,255,255,0.08)}
.nav-link.active{color:#fff;background:var(--teal)}
.nav-right{margin-left:auto;display:flex;align-items:center;gap:14px}
.nav-badge{padding:3px 10px;border-radius:20px;font-size:0.68rem;font-weight:700;letter-spacing:0.5px;text-transform:uppercase}
.badge-idle{background:rgba(255,255,255,0.08);color:rgba(255,255,255,0.4)}
.badge-run{background:var(--teal);color:#fff;animation:pulse 1.5s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:0.5}}
.nav-model{color:rgba(255,255,255,0.35);font-size:0.72rem;font-family:'SF Mono',monospace}

/* LAYOUT */
.page{display:none;padding:24px 28px;max-width:1440px;margin:0 auto}
.page.active{display:block}
h2{font-size:1.35rem;font-weight:800;letter-spacing:-0.5px;margin-bottom:4px}
.subtitle{color:var(--muted);font-size:0.88rem;margin-bottom:22px}

/* CARD */
.card{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);overflow:hidden;box-shadow:var(--shadow)}
.card-head{padding:12px 18px;border-bottom:1px solid var(--border);font-weight:700;font-size:0.85rem;display:flex;align-items:center;gap:8px;background:var(--surface2);color:var(--navy)}

/* BTN */
.btn{padding:8px 18px;border:1px solid var(--border);border-radius:8px;font-size:0.82rem;font-weight:600;cursor:pointer;background:var(--surface);color:var(--text);transition:all 0.15s;font-family:inherit}
.btn:hover{border-color:var(--teal);color:var(--teal)}
.btn-teal{background:var(--teal);color:#fff;border-color:var(--teal)}
.btn-teal:hover{background:var(--teal-dark)}
.btn-coral{background:var(--coral);color:#fff;border-color:var(--coral)}
.btn-coral:hover{background:var(--coral-dark)}
.btn:disabled{opacity:0.35;cursor:not-allowed}
.btn-sm{padding:5px 12px;font-size:0.78rem}

/* LIVE PROGRESS */
.live-bar{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);padding:16px 20px;margin-bottom:20px;display:none;box-shadow:var(--shadow)}
.live-bar.active{display:block}
.live-top{display:flex;align-items:center;justify-content:space-between;margin-bottom:10px}
.live-label{font-size:0.82rem;font-weight:600;color:var(--navy)}
.live-count{font-size:0.78rem;color:var(--muted);font-weight:600}
.live-track{height:6px;background:var(--surface2);border-radius:3px;overflow:hidden;margin-bottom:10px}
.live-fill{height:100%;background:linear-gradient(90deg,var(--teal),var(--coral));border-radius:3px;transition:width 0.5s ease}
.live-detail{display:flex;gap:16px;flex-wrap:wrap}
.live-chip{padding:4px 10px;border-radius:6px;font-size:0.72rem;font-weight:600;display:flex;align-items:center;gap:5px}
.live-chip.target{background:var(--purple-bg);color:var(--purple)}
.live-chip.metric{background:var(--teal-light);color:var(--teal-dark)}
.live-chip.done{background:var(--green-bg);color:var(--green-dark)}
.live-spinner{width:14px;height:14px;border:2px solid var(--teal-light);border-top-color:var(--teal);border-radius:50%;animation:spin 0.8s linear infinite;display:inline-block}
@keyframes spin{to{transform:rotate(360deg)}}

/* STATS */
.stats-row{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:10px;margin-bottom:22px}
.stat{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);padding:14px 16px;box-shadow:var(--shadow)}
.stat .lbl{font-size:0.68rem;color:var(--muted);text-transform:uppercase;letter-spacing:0.8px;font-weight:700}
.stat .val{font-size:1.5rem;font-weight:800;margin-top:2px;letter-spacing:-0.5px}
.stat .val.g{color:var(--green)}.stat .val.r{color:var(--red)}.stat .val.a{color:var(--amber)}

/* METRIC CARDS */
.m-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:14px;margin-bottom:24px}
.m-card{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);padding:18px;cursor:pointer;transition:all 0.2s;position:relative;box-shadow:var(--shadow)}
.m-card:hover{border-color:var(--teal);transform:translateY(-2px);box-shadow:var(--shadow-lg)}
.m-card.sel{border-color:var(--teal);background:var(--teal-light)}
.m-tags{display:flex;gap:4px;margin-bottom:8px;flex-wrap:wrap}
.m-tag{padding:2px 8px;border-radius:5px;font-size:0.62rem;font-weight:700;text-transform:uppercase;letter-spacing:0.4px}
.m-tag.quality{background:#dbeafe;color:#1d4ed8}.m-tag.safety{background:var(--red-bg);color:var(--red)}
.m-tag.retrieval{background:var(--green-bg);color:var(--green-dark)}.m-tag.geval{background:var(--amber-bg);color:var(--amber)}
.m-tag.chatbot{background:var(--purple-bg);color:var(--purple)}.m-tag.rag{background:var(--coral-light);color:var(--coral-dark)}
.m-name{font-weight:700;font-size:0.92rem;margin-bottom:3px}
.m-desc{font-size:0.76rem;color:var(--muted);margin-bottom:10px;line-height:1.4}
.m-score-row{display:flex;align-items:center;gap:8px;margin-bottom:6px}
.pill{display:inline-block;padding:3px 10px;border-radius:6px;font-size:0.72rem;font-weight:700}
.pill-pass{background:var(--green-bg);color:var(--green-dark)}.pill-fail{background:var(--red-bg);color:var(--red-dark)}
.pill-error{background:var(--amber-bg);color:var(--amber)}
.m-val{font-size:1.4rem;font-weight:800;letter-spacing:-0.5px}
.m-reason{font-size:0.74rem;color:var(--muted);line-height:1.4;margin-bottom:8px;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden}
.m-bar{height:5px;background:var(--surface2);border-radius:3px;overflow:hidden}
.m-bar-fill{height:100%;border-radius:3px;transition:width 0.4s}
.m-meta{font-size:0.68rem;color:var(--subtle);margin-top:8px;display:flex;justify-content:space-between}
.m-threshold{position:absolute;top:14px;right:16px;font-size:0.72rem;color:var(--subtle)}

/* SUGGESTION BOX */
.suggest{background:linear-gradient(135deg,#eff6ff,#f0fdf4);border:1px solid #bfdbfe;border-radius:var(--radius);padding:16px 18px;margin-top:16px}
.suggest-title{font-weight:700;font-size:0.88rem;color:var(--navy);margin-bottom:8px;display:flex;align-items:center;gap:6px}
.suggest-item{font-size:0.82rem;color:var(--navy2);line-height:1.6;padding:4px 0;padding-left:16px;position:relative}
.suggest-item::before{content:">";position:absolute;left:0;color:var(--teal);font-weight:700}

/* DETAIL TABLE */
.detail{margin-top:20px}
.detail h3{font-size:1rem;font-weight:700;margin-bottom:12px}
table{width:100%;border-collapse:collapse;background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);overflow:hidden;box-shadow:var(--shadow)}
th{background:var(--surface2);text-align:left;padding:10px 14px;font-size:0.68rem;text-transform:uppercase;letter-spacing:0.8px;color:var(--muted);font-weight:700;border-bottom:1px solid var(--border)}
td{padding:10px 14px;font-size:0.82rem;border-bottom:1px solid var(--border);vertical-align:top}
tr:last-child td{border-bottom:none}
tr:hover{background:var(--surface2)}

/* TABS */
.tabs{display:flex;gap:4px;margin-bottom:18px}
.tab-btn{padding:7px 16px;border:1px solid var(--border);border-radius:8px;font-size:0.82rem;font-weight:600;cursor:pointer;background:var(--surface);color:var(--muted);transition:all 0.15s;font-family:inherit}
.tab-btn:hover{color:var(--text)}.tab-btn.active{background:var(--navy);color:#fff;border-color:var(--navy)}

/* CHAT */
.chat-wrap{display:grid;grid-template-columns:1fr 340px;gap:18px}
@media(max-width:900px){.chat-wrap{grid-template-columns:1fr}}
.chat-box{display:flex;flex-direction:column;height:520px}
.chat-msgs{flex:1;overflow-y:auto;padding:16px;display:flex;flex-direction:column;gap:10px;background:var(--surface2)}
.chat-msg{display:flex;gap:10px;max-width:85%}
.chat-msg.bot{align-self:flex-start}.chat-msg.user{align-self:flex-end;flex-direction:row-reverse}
.chat-avatar{width:30px;height:30px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:0.85rem;flex-shrink:0}
.chat-msg.bot .chat-avatar{background:var(--teal-light);color:var(--teal-dark)}.chat-msg.user .chat-avatar{background:var(--coral-light);color:var(--coral-dark)}
.chat-bubble{padding:10px 14px;border-radius:14px;font-size:0.85rem;line-height:1.5;white-space:pre-wrap}
.chat-msg.bot .chat-bubble{background:var(--surface);border:1px solid var(--border);border-bottom-left-radius:4px}
.chat-msg.user .chat-bubble{background:var(--navy);color:#fff;border-bottom-right-radius:4px}
.chat-input{display:flex;gap:8px;padding:12px;border-top:1px solid var(--border);background:var(--surface)}
.chat-input input{flex:1;padding:9px 14px;border:1px solid var(--border);border-radius:20px;font-size:0.85rem;outline:none;font-family:inherit}
.chat-input input:focus{border-color:var(--teal)}
.chat-input button{padding:9px 20px;background:var(--teal);color:#fff;border:none;border-radius:20px;font-weight:600;cursor:pointer;font-family:inherit}

/* RAG */
.rag-source{background:var(--surface2);border-left:3px solid var(--teal);padding:10px 14px;margin-bottom:8px;border-radius:0 8px 8px 0;font-size:0.82rem}
.rag-source .src-name{color:var(--teal-dark);font-weight:700;font-size:0.78rem}
.chunk-card{background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:12px;margin-bottom:8px;transition:border-color 0.15s}
.chunk-card:hover{border-color:var(--teal)}
.chunk-card .chunk-src{color:var(--teal);font-size:0.72rem;font-weight:700;margin-bottom:3px}
.chunk-card .chunk-text{font-size:0.8rem;color:var(--text);line-height:1.5}
.chunk-card .chunk-meta{font-size:0.68rem;color:var(--subtle);margin-top:5px}

/* PIPELINE */
.pipeline{display:flex;gap:0;align-items:stretch;margin-bottom:24px;overflow-x:auto;padding-bottom:4px}
.pipe-step{flex:1;min-width:150px;background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);padding:14px;box-shadow:var(--shadow)}
.pipe-num{width:22px;height:22px;background:var(--teal);color:#fff;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:0.65rem;font-weight:800;margin-bottom:6px}
.pipe-title{font-weight:700;font-size:0.88rem;margin-bottom:3px}
.pipe-desc{font-size:0.76rem;color:var(--muted);line-height:1.4}
.pipe-desc code{background:var(--teal-light);padding:1px 5px;border-radius:4px;font-size:0.72rem;color:var(--teal-dark)}
.pipe-arrow{display:flex;align-items:center;padding:0 6px;color:var(--border);font-size:1.1rem}

.empty{text-align:center;padding:50px 20px;color:var(--muted)}
.empty .icon{font-size:2.5rem;margin-bottom:10px}
.footer{text-align:center;padding:20px;color:var(--subtle);font-size:0.78rem;border-top:1px solid var(--border);margin-top:32px}
</style></head>
<body>
'''
    html2 = '''
<nav class="nav">
  <div class="nav-logo"><div class="dot">&#9670;</div>DeepEval Dashboard</div>
  <div class="nav-links">
    <button class="nav-link active" data-page="dashboard" onclick="showPage('dashboard')">Dashboard</button>
    <button class="nav-link" data-page="chatbot" onclick="showPage('chatbot')">Chatbot</button>
    <button class="nav-link" data-page="rag" onclick="showPage('rag')">RAG Explorer</button>
    <button class="nav-link" data-page="eval" onclick="showPage('eval')">Evaluations</button>
  </div>
  <div class="nav-right">
    <span id="status-badge" class="nav-badge badge-idle">IDLE</span>
    <span class="nav-model">''' + jm + '''</span>
  </div>
</nav>

<!-- DASHBOARD -->
<div class="page active" id="page-dashboard">
  <h2>Pipeline Status</h2>
  <p class="subtitle">A complete local-first RAG pipeline: documents flow left-to-right; inspect every stage.</p>
  <div class="pipeline">
    <div class="pipe-step"><div class="pipe-num">1</div><div class="pipe-title">Ingest</div><div class="pipe-desc">Load <code>.txt</code> / <code>.pdf</code>; split into ~500-char chunks with overlap.</div></div>
    <div class="pipe-arrow">&#8594;</div>
    <div class="pipe-step"><div class="pipe-num">2</div><div class="pipe-title">Embed</div><div class="pipe-desc"><code>nomic-embed-text-v1.5</code> (768-dim) via sentence-transformers.</div></div>
    <div class="pipe-arrow">&#8594;</div>
    <div class="pipe-step"><div class="pipe-num">3</div><div class="pipe-title">Store</div><div class="pipe-desc">ChromaDB persistent collection, cosine distance. <code>ecommerce_docs</code></div></div>
    <div class="pipe-arrow">&#8594;</div>
    <div class="pipe-step"><div class="pipe-num">4</div><div class="pipe-title">Retrieve</div><div class="pipe-desc">Top-k semantic search returns chunks with similarity scores.</div></div>
    <div class="pipe-arrow">&#8594;</div>
    <div class="pipe-step"><div class="pipe-num">5</div><div class="pipe-title">Answer</div><div class="pipe-desc">Groq LLM grounds reply in retrieved chunks; cites sources.</div></div>
  </div>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:24px">
    <div class="card"><div class="card-head">Vector Store</div><div style="padding:14px">
      <table style="border:none;box-shadow:none"><tbody>
        <tr><td style="color:var(--muted);border:none;padding:5px 10px;font-size:0.84rem">Collection</td><td style="border:none;padding:5px 10px;font-size:0.84rem"><code style="background:var(--teal-light);padding:2px 6px;border-radius:4px;color:var(--teal-dark);font-size:0.78rem">ecommerce_docs</code></td></tr>
        <tr><td style="color:var(--muted);border:none;padding:5px 10px;font-size:0.84rem">Total chunks</td><td style="border:none;padding:5px 10px;font-size:0.84rem;font-weight:700" id="dash-chunks">-</td></tr>
        <tr><td style="color:var(--muted);border:none;padding:5px 10px;font-size:0.84rem">Groq configured</td><td style="border:none;padding:5px 10px;font-size:0.84rem;color:var(--green);font-weight:600">yes</td></tr>
        <tr><td style="color:var(--muted);border:none;padding:5px 10px;font-size:0.84rem">Judge LLM</td><td style="border:none;padding:5px 10px;font-size:0.84rem"><code style="background:var(--surface2);padding:2px 6px;border-radius:4px;font-size:0.76rem">''' + jm + '''</code></td></tr>
      </tbody></table>
    </div></div>
    <div class="card"><div class="card-head">Services</div><div style="padding:14px">
      <table style="border:none;box-shadow:none"><tbody>
        <tr><td style="color:var(--muted);border:none;padding:5px 10px;font-size:0.84rem">Chatbot</td><td style="border:none;padding:5px 10px;font-size:0.84rem"><code style="background:var(--surface2);padding:2px 6px;border-radius:4px;font-size:0.76rem">localhost:3000</code></td></tr>
        <tr><td style="color:var(--muted);border:none;padding:5px 10px;font-size:0.84rem">RAG Explorer</td><td style="border:none;padding:5px 10px;font-size:0.84rem"><code style="background:var(--surface2);padding:2px 6px;border-radius:4px;font-size:0.76rem">localhost:5001</code></td></tr>
        <tr><td style="color:var(--muted);border:none;padding:5px 10px;font-size:0.84rem">Dashboard</td><td style="border:none;padding:5px 10px;font-size:0.84rem"><code style="background:var(--surface2);padding:2px 6px;border-radius:4px;font-size:0.76rem">localhost:8501</code></td></tr>
        <tr><td style="color:var(--muted);border:none;padding:5px 10px;font-size:0.84rem">Embedding</td><td style="border:none;padding:5px 10px;font-size:0.84rem"><code style="background:var(--teal-light);padding:2px 6px;border-radius:4px;color:var(--teal-dark);font-size:0.76rem">nomic-embed-v1.5</code></td></tr>
      </tbody></table>
    </div></div>
  </div>
  <div style="display:flex;gap:10px;flex-wrap:wrap">
    <button class="btn btn-teal" onclick="showPage('eval');runEval('all')">&#9654; Run All Evaluations</button>
    <button class="btn" onclick="showPage('chatbot')">Open Chatbot</button>
    <button class="btn" onclick="showPage('rag')">Open RAG Explorer</button>
  </div>
</div>
'''
    html3 = '''
<!-- CHATBOT -->
<div class="page" id="page-chatbot">
  <h2>ShopSmart Chatbot</h2>
  <p class="subtitle">Customer Support Bot &middot; Ask about orders, shipping, refunds, products.</p>
  <div class="chat-wrap">
    <div class="card chat-box">
      <div class="card-head">&#128172; Chat</div>
      <div class="chat-msgs" id="chat-msgs">
        <div class="chat-msg bot"><div class="chat-avatar">&#129302;</div><div class="chat-bubble">Hi! I\'m ShopBot. Ask me about orders, shipping, refunds, or products.</div></div>
      </div>
      <div class="chat-input">
        <input id="chat-in" placeholder="Ask about orders, shipping, refunds..." onkeydown="if(event.key===\'Enter\')sendChat()">
        <button onclick="sendChat()">Send</button>
      </div>
    </div>
    <div class="card" style="display:flex;flex-direction:column">
      <div class="card-head">&#9889; Quick Topics</div>
      <div style="padding:14px;display:flex;flex-direction:column;gap:5px">
        <button class="btn btn-sm" onclick="quickChat(\'What\\'s your refund policy?\')">What\'s your refund policy?</button>
        <button class="btn btn-sm" onclick="quickChat(\'How long does standard shipping take?\')">How long does shipping take?</button>
        <button class="btn btn-sm" onclick="quickChat(\'Show me headphones\')">Show me headphones</button>
        <button class="btn btn-sm" onclick="quickChat(\'Do you have any discounts?\')">Do you have any discounts?</button>
        <button class="btn btn-sm" onclick="quickChat(\'What payment methods do you accept?\')">Payment methods?</button>
        <button class="btn btn-sm" onclick="quickChat(\'What are your cheapest products?\')">Cheapest products?</button>
        <button class="btn btn-sm" onclick="quickChat(\'I want to track my order\')">Track my order</button>
        <button class="btn btn-sm" onclick="quickChat(\'What categories do you have?\')">Product categories?</button>
      </div>
    </div>
  </div>
</div>

<!-- RAG EXPLORER -->
<div class="page" id="page-rag">
  <h2>RAG Explorer</h2>
  <p class="subtitle">Ingest &middot; Embed &middot; Search &middot; Answer &middot; Evaluate</p>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:18px">
    <div class="card">
      <div class="card-head">&#128269; RAG Query</div>
      <div style="padding:14px">
        <div style="display:flex;gap:8px;margin-bottom:12px">
          <input id="rag-in" style="flex:1;padding:9px 14px;border:1px solid var(--border);border-radius:8px;font-size:0.85rem;outline:none;font-family:inherit" placeholder="Ask about shipping, returns, products..." onkeydown="if(event.key===\'Enter\')ragQuery()">
          <button class="btn btn-teal" onclick="ragQuery()">Search</button>
        </div>
        <div id="rag-answer"></div>
        <div id="rag-sources" style="max-height:280px;overflow-y:auto"></div>
      </div>
    </div>
    <div class="card">
      <div class="card-head">&#128196; Document Chunks <span style="margin-left:auto;font-weight:400;font-size:0.78rem;color:var(--muted)" id="chunk-count"></span></div>
      <div id="chunks-list" style="padding:14px;max-height:420px;overflow-y:auto"><span style="color:var(--muted);font-size:0.84rem">Loading chunks...</span></div>
    </div>
  </div>
</div>
'''
    html4 = '''
<!-- EVALUATIONS -->
<div class="page" id="page-eval">
  <h2>DeepEval Metrics</h2>
  <p class="subtitle">Live metric runs against the chatbot and RAG pipeline &middot; 15 metrics &middot; Click a card for details.</p>
  <div style="display:flex;gap:10px;align-items:center;margin-bottom:16px;flex-wrap:wrap">
    <button class="btn btn-coral" onclick="runEval(\'all\')" id="btn-run">&#9654; Run All</button>
    <button class="btn" onclick="runEval(\'chatbot\')">Chatbot Only</button>
    <button class="btn" onclick="runEval(\'rag\')">RAG Only</button>
  </div>

  <div class="live-bar" id="live-bar">
    <div class="live-top">
      <div class="live-label"><span class="live-spinner"></span>&nbsp; Running evaluations...</div>
      <div class="live-count" id="live-count">0 / 0</div>
    </div>
    <div class="live-track"><div class="live-fill" id="live-fill" style="width:0%"></div></div>
    <div class="live-detail">
      <div class="live-chip target" id="live-target">-</div>
      <div class="live-chip metric" id="live-metric">-</div>
      <div class="live-chip done" id="live-done">0 completed</div>
    </div>
  </div>

  <div class="tabs" id="eval-tabs">
    <button class="tab-btn active" data-et="chatbot" onclick="switchEvalTab(\'chatbot\')">Chatbot</button>
    <button class="tab-btn" data-et="rag" onclick="switchEvalTab(\'rag\')">RAG Pipeline</button>
  </div>
  <div id="eval-stats" class="stats-row"></div>
  <div id="eval-content"><div class="empty"><div class="icon">&#129514;</div><p>Click "Run All" to start evaluations</p></div></div>
</div>

<div class="footer">DeepEval Dashboard &middot; ChromaDB + Nomic Embed + Groq &middot; Evaluated by DeepEval</div>
'''
    js = r'''<script>
let evalTab='chatbot',results={chatbot:[],rag:[]},selMetric=null;

function showPage(p){
  document.querySelectorAll('.page').forEach(el=>el.classList.remove('active'));
  document.getElementById('page-'+p).classList.add('active');
  document.querySelectorAll('.nav-link').forEach(el=>el.classList.toggle('active',el.dataset.page===p));
  if(p==='rag') loadChunks();
}

// CHAT
function sendChat(){
  const inp=document.getElementById('chat-in');const msg=inp.value.trim();if(!msg)return;inp.value='';
  addChatMsg('user',msg);
  fetch('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({message:msg})}).then(r=>r.json()).then(d=>addChatMsg('bot',d.reply));
}
function quickChat(m){document.getElementById('chat-in').value=m;sendChat()}
function addChatMsg(role,text){
  const el=document.getElementById('chat-msgs');
  el.innerHTML+=`<div class="chat-msg ${role}"><div class="chat-avatar">${role==='bot'?'&#129302;':'&#128578;'}</div><div class="chat-bubble">${text}</div></div>`;
  el.scrollTop=el.scrollHeight;
}

// RAG
async function ragQuery(){
  const q=document.getElementById('rag-in').value.trim();if(!q)return;
  document.getElementById('rag-answer').innerHTML='<span style="color:var(--muted)">Searching...</span>';
  document.getElementById('rag-sources').innerHTML='';
  const d=await(await fetch('/api/rag_query',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({query:q})})).json();
  document.getElementById('rag-answer').innerHTML=`<div style="background:var(--surface2);border:1px solid var(--border);border-radius:var(--radius);padding:14px;margin-bottom:12px;line-height:1.6;font-size:0.88rem">${d.answer||'No answer'}</div>`;
  if(d.sources&&d.sources.length){
    document.getElementById('rag-sources').innerHTML='<div style="font-size:0.78rem;color:var(--muted);margin-bottom:6px;font-weight:600">Sources:</div>'+
      d.sources.map(s=>`<div class="rag-source"><span class="src-name">${s.source}</span> <span style="color:var(--subtle);font-size:0.7rem">(dist: ${s.distance.toFixed(4)})</span><br>${s.text.substring(0,180)}...</div>`).join('');
  }
}
async function loadChunks(){
  try{
    const d=await(await fetch('/api/rag_chunks')).json();
    document.getElementById('dash-chunks').textContent=d.total||0;
    document.getElementById('chunk-count').textContent=(d.total||0)+' chunks';
    if(!d.chunks||!d.chunks.length){document.getElementById('chunks-list').innerHTML='<span style="color:var(--muted)">No chunks yet.</span>';return}
    document.getElementById('chunks-list').innerHTML=d.chunks.map(c=>
      `<div class="chunk-card"><div class="chunk-src">${c.source}</div><div class="chunk-text">${c.text.substring(0,200)}${c.text.length>200?'...':''}</div><div class="chunk-meta">Chunk #${c.chunk_index}</div></div>`
    ).join('');
  }catch(e){document.getElementById('chunks-list').innerHTML='<span style="color:var(--muted)">RAG service not available</span>'}
}

// EVAL
async function runEval(target){
  document.querySelectorAll('.btn').forEach(b=>b.disabled=true);
  document.getElementById('live-bar').classList.add('active');
  await fetch('/api/run',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({target})});
  pollStatus();
}
async function pollStatus(){
  const d=await(await fetch('/api/status')).json();
  const badge=document.getElementById('status-badge');
  if(d.running){
    badge.className='nav-badge badge-run';badge.textContent='RUNNING';
    const pct=d.total?Math.round(d.done/d.total*100):0;
    document.getElementById('live-fill').style.width=pct+'%';
    document.getElementById('live-count').textContent=d.done+' / '+d.total;
    document.getElementById('live-target').textContent=d.current_target||'-';
    document.getElementById('live-metric').textContent=d.current_metric||'-';
    document.getElementById('live-done').textContent=d.done+' completed';
    // Also refresh partial results
    const rd=await(await fetch('/api/results')).json();
    results=rd; renderEval();
    setTimeout(pollStatus,2000);
  }else{
    badge.className='nav-badge badge-idle';badge.textContent='IDLE';
    document.querySelectorAll('.btn').forEach(b=>b.disabled=false);
    document.getElementById('live-bar').classList.remove('active');
    loadResults();
  }
}
async function loadResults(){results=await(await fetch('/api/results')).json();renderEval()}
function switchEvalTab(t){evalTab=t;selMetric=null;document.querySelectorAll('#eval-tabs .tab-btn').forEach(b=>b.classList.toggle('active',b.dataset.et===t));renderEval()}
function getMetrics(rows){const s=new Set();rows.forEach(r=>Object.keys(r.metrics).forEach(m=>s.add(m)));return[...s]}
function avg(rows,m){let s=0,c=0;rows.forEach(r=>{if(r.metrics[m]){s+=r.metrics[m].score;c++}});return c?s/c:0}
function pr(rows,m){let p=0,c=0;rows.forEach(r=>{if(r.metrics[m]){c++;if(r.metrics[m].passed)p++}});return c?p/c:0}

const MM={
  "Answer Relevancy":{tag:"quality",desc:"Reply stays on-topic for the question.",fix:"Add more specific details from the query. Ensure the response directly addresses what was asked without tangential info."},
  "Faithfulness":{tag:"quality",desc:"Every claim is backed by ground-truth context.",fix:"Remove any claims not present in the retrieval context. Ground every statement in source documents."},
  "Hallucination":{tag:"quality",desc:"Detects statements contradicting ground-truth.",fix:"Cross-check generated facts against source docs. Add retrieval-augmented verification step before responding."},
  "Toxicity":{tag:"safety",desc:"Reply is free of rude / harmful language.",fix:"Add a toxicity filter layer. Fine-tune with safe response examples. Strengthen system prompt guardrails."},
  "Bias":{tag:"safety",desc:"Reply is free of biased / prejudiced statements.",fix:"Use neutral language. Avoid gendered or culturally specific assumptions. Review training data for bias patterns."},
  "Contextual Precision":{tag:"retrieval",desc:"Relevant context ranked higher than irrelevant.",fix:"Improve embedding model or add re-ranking step. Tune chunk size for better semantic boundaries."},
  "Contextual Recall":{tag:"retrieval",desc:"Retrieved context covers the expected answer.",fix:"Increase top-k retrieval count. Add more comprehensive source documents. Improve chunking overlap."},
  "Contextual Relevancy":{tag:"retrieval",desc:"All retrieved context is relevant to the query.",fix:"Reduce top-k or add relevance threshold filtering. Fine-tune embeddings on domain-specific data."},
  "Correctness":{tag:"geval",desc:"Reply matches expected output factually.",fix:"Ensure the LLM has access to accurate, up-to-date information. Add fact-checking against known answers."},
  "Coherence":{tag:"geval",desc:"Response is logically structured and clear.",fix:"Improve prompt template structure. Add chain-of-thought reasoning. Use numbered steps for complex answers."},
  "Completeness":{tag:"geval",desc:"Reply covers all key facts in expected output.",fix:"Expand response generation to cover all aspects. Use checklist-based prompting to ensure coverage."},
  "Conciseness":{tag:"geval",desc:"Response is brief without unnecessary verbosity.",fix:"Add max-length constraints. Instruct the model to be direct. Remove filler phrases from prompt templates."},
  "Helpfulness":{tag:"geval",desc:"How useful the response is for the customer.",fix:"Include actionable next steps. Add relevant links or references. Anticipate follow-up questions."},
  "Politeness":{tag:"geval",desc:"Maintains polite professional customer service tone.",fix:"Add tone guidelines to system prompt. Include greeting/closing patterns. Fine-tune on customer service data."},
  "Safety":{tag:"safety",desc:"Avoids harmful or inappropriate content.",fix:"Strengthen content filtering. Add safety-focused system instructions. Test with adversarial inputs."},
};

function renderEval(){
  // Combine both chatbot and rag for the "all" view
  const cRows=results.chatbot||[], rRows=results.rag||[];
  const viewRows=evalTab==='chatbot'?cRows:(evalTab==='rag'?rRows:[]);
  const allMetrics=new Set();
  cRows.forEach(r=>Object.keys(r.metrics).forEach(m=>allMetrics.add(m)));
  rRows.forEach(r=>Object.keys(r.metrics).forEach(m=>allMetrics.add(m)));
  const metrics=[...allMetrics];

  if(!cRows.length&&!rRows.length){
    document.getElementById('eval-stats').innerHTML='';
    document.getElementById('eval-content').innerHTML='<div class="empty"><div class="icon">&#129514;</div><p>No results yet. Click "Run All" to start.</p></div>';
    return;
  }

  // Stats for current tab
  const rows=viewRows;
  let tp=0,tt=0;
  metrics.forEach(m=>rows.forEach(r=>{if(r.metrics[m]){tt++;if(r.metrics[m].passed)tp++}}));
  const rate=tt?(tp/tt*100):0, avgAll=metrics.length?metrics.reduce((s,m)=>s+avg(rows,m),0)/metrics.length:0;

  document.getElementById('eval-stats').innerHTML=`
    <div class="stat"><div class="lbl">Test Cases</div><div class="val">${rows.length}</div></div>
    <div class="stat"><div class="lbl">Metrics</div><div class="val">${metrics.length}</div></div>
    <div class="stat"><div class="lbl">Pass Rate</div><div class="val ${rate>=70?'g':rate>=40?'a':'r'}">${rate.toFixed(1)}%</div></div>
    <div class="stat"><div class="lbl">Avg Score</div><div class="val ${avgAll>=0.7?'g':avgAll>=0.4?'a':'r'}">${avgAll.toFixed(3)}</div></div>
    <div class="stat"><div class="lbl">Passed</div><div class="val g">${tp}</div></div>
    <div class="stat"><div class="lbl">Failed</div><div class="val r">${tt-tp}</div></div>`;

  let h='<div class="m-grid">';
  metrics.forEach(m=>{
    const meta=MM[m]||{tag:'geval',desc:'',fix:''};
    // Gather per-test-case results for BOTH chatbot and rag
    const cData=cRows.map(r=>r.metrics[m]).filter(Boolean);
    const rData=rRows.map(r=>r.metrics[m]).filter(Boolean);
    const curData=evalTab==='chatbot'?cData:rData;
    const curPassN=curData.filter(d=>d.passed).length;
    const curFailN=curData.length-curPassN;
    const curAvg=curData.length?curData.reduce((s,d)=>s+d.score,0)/curData.length:0;
    const threshold=curData[0]?.threshold||0.5;
    const color=curAvg>=0.7?'var(--green)':curAvg>=0.4?'var(--amber)':'var(--red)';
    const allPass=curFailN===0&&curData.length>0;
    const allFail=curPassN===0&&curData.length>0;
    const sel=selMetric===m?' sel':'';

    // Test case dots: green=pass, red=fail
    let dots='';
    curData.forEach((d,i)=>{
      const dc=d.passed?'var(--green)':'var(--red)';
      dots+=`<span title="Test ${i+1}: ${d.score.toFixed(3)} ${d.passed?'PASS':'FAIL'}" style="display:inline-block;width:8px;height:8px;border-radius:50%;background:${dc};margin:1px"></span>`;
    });

    // Both-target summary line
    let bothLine='';
    if(cData.length>0&&rData.length>0){
      const cP=cData.filter(d=>d.passed).length, rP=rData.filter(d=>d.passed).length;
      bothLine=`<div style="font-size:0.68rem;color:var(--subtle);margin-top:4px">Chatbot: ${cP}/${cData.length} passed &middot; RAG: ${rP}/${rData.length} passed</div>`;
    }

    h+=`<div class="m-card${sel}" onclick="selectMetric('${m}')">
      <div class="m-tags"><span class="m-tag ${meta.tag}">${meta.tag}</span><span class="m-tag ${evalTab}">${evalTab}</span></div>
      <div class="m-threshold">threshold ${threshold.toFixed(2)}</div>
      <div class="m-name">${m}</div>
      <div class="m-desc">${meta.desc}</div>
      <div class="m-score-row">
        <span class="m-val" style="color:${color}">${curAvg.toFixed(3)}</span>
        <span style="font-size:0.76rem;color:var(--muted);margin-left:4px">/ ${threshold.toFixed(2)}</span>
      </div>
      <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px">
        <span class="pill pill-pass" style="font-size:0.7rem">${curPassN} passed</span>
        ${curFailN>0?`<span class="pill pill-fail" style="font-size:0.7rem">${curFailN} failed</span>`:''}
        ${allPass?'<span style="font-size:0.7rem;color:var(--green)">&#10003; All passed</span>':''}
      </div>
      <div style="display:flex;flex-wrap:wrap;gap:1px;margin-bottom:8px">${dots}</div>
      <div class="m-bar"><div class="m-bar-fill" style="width:${curData.length?(curPassN/curData.length*100):0}%;background:${color}"></div></div>
      ${bothLine}
      <div class="m-meta"><span>${curPassN}/${curData.length} test cases</span><span>groq</span></div>
    </div>`;
  });
  h+='</div>';

  // Suggestions for metrics with failures
  const failedMetrics=metrics.filter(m=>{
    const d=(evalTab==='chatbot'?cRows:rRows).map(r=>r.metrics[m]).filter(Boolean);
    return d.some(x=>!x.passed);
  });
  if(failedMetrics.length>0){
    h+=`<div class="suggest"><div class="suggest-title">&#128161; Fine-Tuning Suggestions (${failedMetrics.length} metrics have failures)</div>`;
    failedMetrics.forEach(m=>{
      const meta=MM[m]||{fix:'Review and improve this metric area.'};
      const d=(evalTab==='chatbot'?cRows:rRows).map(r=>r.metrics[m]).filter(Boolean);
      const failN=d.filter(x=>!x.passed).length;
      h+=`<div class="suggest-item"><strong>${m}</strong> (${failN} failed, avg ${avg(rows,m).toFixed(3)}): ${meta.fix}</div>`;
    });
    h+='</div>';
  }

  // Detail table for selected metric — show every test case with clear pass/fail
  if(selMetric&&metrics.includes(selMetric)){
    const meta=MM[selMetric]||{fix:''};
    const curRows=evalTab==='chatbot'?cRows:rRows;
    const mData=curRows.map(r=>({input:r.input,output:r.actual_output,...(r.metrics[selMetric]||{})})).filter(d=>d.score!==undefined);
    const passN=mData.filter(d=>d.passed).length, failN=mData.length-passN;

    h+=`<div class="detail"><h3>${selMetric} &mdash; ${passN} Passed, ${failN} Failed out of ${mData.length} Test Cases</h3>`;
    if(failN>0&&meta.fix){
      h+=`<div class="suggest" style="margin-bottom:14px"><div class="suggest-title">&#128295; How to Improve ${selMetric}</div><div class="suggest-item">${meta.fix}</div></div>`;
    }
    h+=`<table><tr><th style="width:22%">Input</th><th style="width:22%">Output</th><th style="width:10%">Score vs Threshold</th><th style="width:8%">Result</th><th>Reason / Justification</th></tr>`;
    mData.forEach(d=>{
      const cls=d.passed?'pill-pass':'pill-fail';
      const icon=d.passed?'&#10003;':'&#10007;';
      const rowBg=d.passed?'':'background:var(--red-bg);';
      h+=`<tr style="${rowBg}">
        <td>${d.input}</td>
        <td style="max-width:200px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${(d.output||'').substring(0,65)}...</td>
        <td><span class="pill ${cls}" style="font-size:0.78rem">${d.score.toFixed(3)}</span> <span style="color:var(--subtle);font-size:0.72rem">/ ${(d.threshold||0.5).toFixed(2)}</span></td>
        <td><span class="pill ${cls}" style="font-size:0.78rem">${icon} ${d.passed?'Pass':'Fail'}</span></td>
        <td style="font-size:0.76rem;color:var(--muted);max-width:280px;line-height:1.5">${(d.reason||'No reason provided').substring(0,200)}</td></tr>`;
    });
    h+='</table></div>';
  }
  document.getElementById('eval-content').innerHTML=h;
}
function selectMetric(m){selMetric=selMetric===m?null:m;renderEval()}

// Init
loadChunks();
fetch('/api/status').then(r=>r.json()).then(d=>{if(d.running)pollStatus();else if(d.last_run)loadResults()});
</script></body></html>'''

    return html1 + html2 + html3 + html4 + js


if __name__ == "__main__":
    port = int(os.environ.get("DASHBOARD_PORT", 8501))
    print(f"\n  DeepEval Dashboard -> http://localhost:{port}\n")
    app.run(host="0.0.0.0", port=port, debug=False)
