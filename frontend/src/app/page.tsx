"use client";

import { useState, useRef } from "react";

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

interface Hit {
  answer: string;
  pages: number[];
  image_paths: string[];
  image_b64s: string[];
  doc_id: string;
  query_type: string;
  confidence: number;
  verified: boolean;
  latencies: Record<string, number>;
  trace_url?: string;
}

function pageImageSrc(result: Hit, index: number): string {
  // Use base64 from Qdrant (cloud mode) or URL from local server
  const b64 = result.image_b64s?.[index];
  if (b64) return `data:image/jpeg;base64,${b64}`;
  const path = result.image_paths?.[index] || "";
  const parts = path.replace(/\\/g, "/").split("data/pages/");
  return parts.length > 1 ? `${API}/pages/${parts[1]}` : "";
}

export default function Home() {
  const [docId, setDocId] = useState<string | null>(null);
  const [ingesting, setIngesting] = useState(false);
  const [ingestMsg, setIngestMsg] = useState("");
  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<Hit | null>(null);
  const [error, setError] = useState("");
  const [sampleQuestions, setSampleQuestions] = useState<string[]>([
    "What was total revenue for the quarter?",
    "What was net income?",
    "What was the gross margin percentage?",
    "What was earnings per share?",
    "What is the outlook for next quarter?",
  ]);
  const inputRef = useRef<HTMLInputElement>(null);

  async function handleUpload(f: File) {
    setIngesting(true);
    setIngestMsg("Indexing pages with ColQwen2.5…");
    setResult(null);
    setError("");
    const form = new FormData();
    form.append("file", f);
    try {
      const res = await fetch(`${API}/ingest`, { method: "POST", body: form });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Ingest failed");
      setDocId(data.doc_id);
      setIngestMsg(`✓ ${data.pages} pages indexed — ${data.doc_id}`);
      // Fetch document-specific questions
      try {
        const qRes = await fetch(`${API}/suggest-questions/${data.doc_id}`);
        const qData = await qRes.json();
        if (qData.questions?.length) setSampleQuestions(qData.questions);
      } catch { /* keep generic questions */ }
    } catch (e: any) {
      setError(e.message);
      setIngestMsg("");
    } finally {
      setIngesting(false);
    }
  }

  async function handleAsk(q?: string) {
    const q2 = q ?? question;
    if (!q2.trim()) return;
    if (!docId) { setError("Upload a document first."); return; }
    setQuestion(q2);
    setLoading(true);
    setResult(null);
    setError("");
    try {
      const res = await fetch(`${API}/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: q2, doc_id: docId }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Query failed");
      setResult(data);
    } catch (e: any) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  const totalLatency = result
    ? Object.values(result.latencies).reduce((a, b) => a + b, 0).toFixed(2)
    : null;

  return (
    <main className="min-h-screen bg-gray-950">
      {/* NAV */}
      <nav className="border-b border-gray-800 px-6 py-4 flex items-center gap-3">
        <svg className="w-6 h-6 text-teal-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <circle cx="12" cy="12" r="3" strokeWidth="2"/>
          <path strokeWidth="2" d="M12 2v2M12 20v2M2 12h2M20 12h2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M4.93 19.07l1.41-1.41M17.66 6.34l1.41-1.41"/>
        </svg>
        <span className="font-semibold text-lg tracking-tight">Apertura</span>
        <span className="ml-auto text-xs text-gray-500">Visual Document RAG</span>
        <a href="https://github.com/PrajothSenthilKumar/apertura" target="_blank"
           className="ml-4 text-xs text-gray-400 hover:text-white border border-gray-700 rounded px-3 py-1">
          GitHub
        </a>
      </nav>

      {/* HERO */}
      <section className="text-center px-6 py-16 border-b border-gray-800">
        <span className="inline-block text-xs font-medium px-3 py-1 rounded-full border border-teal-700 text-teal-400 mb-5">
          ColQwen2.5 · Qdrant · Claude Vision · LangGraph
        </span>
        <h1 className="text-4xl font-semibold leading-tight max-w-2xl mx-auto mb-4">
          Your documents hide answers in{" "}
          <span className="text-teal-400">charts and tables</span>.<br />
          Apertura reads them.
        </h1>
        <p className="text-gray-400 max-w-xl mx-auto text-base leading-relaxed mb-6">
          Upload a financial filing. Apertura embeds every page as an image, retrieves visually, 
          and answers with a cited source page completely bypassing OCR and text extraction
        </p>
        <div className="flex justify-center gap-6 text-sm">
          <span className="text-teal-400 font-semibold">96.7% accuracy</span>
          <span className="text-gray-600">|</span>
          <span className="text-gray-300">+17.8% over text-RAG baseline</span>
          <span className="text-gray-600">|</span>
          <span className="text-gray-300">30-question golden set</span>
        </div>
      </section>

      {/* APP */}
      <section className="max-w-4xl mx-auto px-6 py-12 grid gap-8">

        {/* Upload */}
        <div
          onClick={() => inputRef.current?.click()}
          onDragOver={e => e.preventDefault()}
          onDrop={e => { e.preventDefault(); const f = e.dataTransfer.files[0]; if (f) handleUpload(f); }}
          className="border-2 border-dashed border-gray-700 hover:border-teal-600 rounded-xl p-10 text-center cursor-pointer transition-colors"
        >
          <input ref={inputRef} type="file" accept=".pdf" className="hidden"
            onChange={e => { const f = e.target.files?.[0]; if (f) handleUpload(f); }} />
          {ingesting ? (
            <p className="text-teal-400 animate-pulse">{ingestMsg}</p>
          ) : ingestMsg ? (
            <p className="text-teal-400 font-medium">{ingestMsg}</p>
          ) : (
            <>
              <p className="text-gray-300 font-medium mb-1">Drop a PDF here or click to upload</p>
              <p className="text-gray-500 text-sm">Financial filings, technical manuals, research papers</p>
            </>
          )}
        </div>

        {/* Sample questions */}
        {docId && (
          <div>
            <p className="text-xs text-gray-500 mb-3">Try a question</p>
            <div className="flex flex-wrap gap-2">
              {sampleQuestions.map(q => (
                <button key={q} onClick={() => handleAsk(q)}
                  className="text-xs px-3 py-2 rounded-lg border border-gray-700 hover:border-teal-600 hover:text-teal-400 transition-colors text-gray-300">
                  {q}
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Ask box */}
        <div className="flex gap-3">
          <input
            value={question}
            onChange={e => setQuestion(e.target.value)}
            onKeyDown={e => e.key === "Enter" && handleAsk()}
            placeholder={docId ? "Ask anything about the document…" : "Upload a document first"}
            disabled={!docId || loading}
            className="flex-1 bg-gray-900 border border-gray-700 rounded-lg px-4 py-3 text-sm focus:outline-none focus:border-teal-600 disabled:opacity-40"
          />
          <button onClick={() => handleAsk()} disabled={!docId || loading || !question.trim()}
            className="px-5 py-3 rounded-lg text-sm font-medium bg-teal-700 hover:bg-teal-600 disabled:opacity-40 transition-colors">
            {loading ? "…" : "Ask"}
          </button>
        </div>

        {/* Error */}
        {error && (
          <div className="bg-red-950 border border-red-800 rounded-lg px-4 py-3 text-sm text-red-300">{error}</div>
        )}

        {/* Answer */}
        {result && (
          <div className="bg-gray-900 border border-gray-700 rounded-xl p-6 space-y-5">
            <div className="flex items-center gap-3 flex-wrap">
              <span className="text-xs px-2 py-1 rounded bg-teal-900 text-teal-300 font-medium">Answer</span>
              <span className="text-xs px-2 py-1 rounded border border-gray-700 text-gray-400 capitalize">{result.query_type}</span>
              {result.verified ? (
                <span className="text-xs px-2 py-1 rounded bg-green-950 border border-green-800 text-green-400">
                  ✓ Verified · {(result.confidence * 100).toFixed(0)}% confidence
                </span>
              ) : (
                <span className="text-xs px-2 py-1 rounded bg-yellow-950 border border-yellow-800 text-yellow-400">
                  ⚠ Low confidence · {(result.confidence * 100).toFixed(0)}%
                </span>
              )}
              <span className="ml-auto text-xs text-gray-500">{totalLatency}s total</span>
            </div>

            <p className="text-gray-100 leading-relaxed text-base whitespace-pre-wrap">
              {result.answer.replace(/\*\*/g, "").replace(/^#{1,3} /gm, "")}
            </p>

            <div className="flex gap-4 flex-wrap">
              {Object.entries(result.latencies).map(([node, ms]) => (
                <div key={node} className="text-center">
                  <p className="text-xs text-teal-400 capitalize">{node}</p>
                  <p className="text-xs text-gray-400">{ms}s</p>
                </div>
              ))}
            </div>

            <div className="pt-2 border-t border-gray-800">
              <p className="text-xs text-gray-500 mb-3">
                Retrieved pages · Sources: {result.pages.join(", ")}
              </p>
              <div className="flex gap-3 flex-wrap">
                {result.pages.map((p, i) => {
                  const src = pageImageSrc(result, i);
                  return (
                    <div key={i} className="text-center">
                      {src ? (
                        <img src={src} alt={`Page ${p}`}
                          className="w-24 h-32 object-cover rounded border-2 border-teal-700" />
                      ) : (
                        <div className="w-24 h-32 bg-gray-800 rounded border-2 border-teal-700 flex items-center justify-center text-xs text-teal-400">
                          p.{p}
                        </div>
                      )}
                      <p className="text-xs text-gray-500 mt-1">page {p}</p>
                    </div>
                  );
                })}
              </div>
            </div>

            {result.trace_url && (
              <div className="pt-2 border-t border-gray-800">
                <a href={result.trace_url} target="_blank" className="text-xs text-teal-400 hover:underline">
                  View trace in Langfuse →
                </a>
              </div>
            )}
          </div>
        )}

        {/* How it works */}
        <div className="border-t border-gray-800 pt-10 grid grid-cols-1 md:grid-cols-3 gap-6 text-center">
          {[
            { step: "01", title: "Upload", body: "Every PDF page is rendered to an image — no OCR, no text extraction." },
            { step: "02", title: "Visual Retrieval", body: "ColQwen2.5 embeds each page. Qdrant finds the most relevant pages by layout, not words." },
            { step: "03", title: "Grounded Answer", body: "A LangGraph pipeline routes, retrieves, reranks, answers, and verifies — Claude reads the page." },
          ].map(({ step, title, body }) => (
            <div key={step} className="p-5 bg-gray-900 rounded-xl border border-gray-800">
              <p className="text-teal-400 text-sm font-semibold mb-2">{step}</p>
              <p className="font-medium mb-2">{title}</p>
              <p className="text-gray-400 text-sm leading-relaxed">{body}</p>
            </div>
          ))}
        </div>
      </section>

      <footer className="border-t border-gray-800 px-6 py-5 text-center text-xs text-gray-600">
        Apertura · Visual RAG for complex documents · © 2026
      </footer>
    </main>
  );
}
