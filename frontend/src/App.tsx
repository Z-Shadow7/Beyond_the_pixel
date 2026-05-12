import { useState, useRef, useCallback } from "react";

const API = "http://127.0.0.1:8000";

type ResultData = {
  success: boolean;
  prediction: string;
  deepfake_score: number | string;
};

type UploadState = "idle" | "loading" | "done" | "error";

function ScanLines() {
  return (
    <div style={{
      position: "absolute", inset: 0, pointerEvents: "none",
      background: "repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(0,255,170,0.015) 2px, rgba(0,255,170,0.015) 4px)",
      borderRadius: "inherit", zIndex: 0,
    }} />
  );
}

function GlitchText({ children }: { children: string }) {
  return (
    <span style={{ position: "relative", display: "inline-block" }}>
      <span style={{ position: "relative", zIndex: 1 }}>{children}</span>
      <span aria-hidden style={{
        position: "absolute", top: 0, left: "2px",
        color: "#ff2d55", opacity: 0.6, zIndex: 0,
        animation: "glitch1 3.5s infinite",
        clipPath: "inset(30% 0 50% 0)",
      }}>{children}</span>
      <span aria-hidden style={{
        position: "absolute", top: 0, left: "-2px",
        color: "#00ffaa", opacity: 0.5, zIndex: 0,
        animation: "glitch2 3.5s infinite",
        clipPath: "inset(60% 0 20% 0)",
      }}>{children}</span>
    </span>
  );
}

function ScoreBar({ score }: { score: number }) {
  const pct = Math.min(100, Math.max(0, Math.round(score * 100)));
  const color = pct > 70 ? "#ff2d55" : pct > 40 ? "#f59e0b" : "#00ffaa";
  const label = pct > 70 ? "HIGH RISK" : pct > 40 ? "SUSPICIOUS" : "AUTHENTIC";
  return (
    <div style={{ marginTop: 16 }}>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6 }}>
        <span style={{ fontSize: 11, letterSpacing: "0.15em", color: "#6b7280", fontFamily: "'Space Mono', monospace" }}>DEEPFAKE SCORE</span>
        <span style={{ fontSize: 11, letterSpacing: "0.1em", color, fontFamily: "'Space Mono', monospace", fontWeight: 700 }}>{label}</span>
      </div>
      <div style={{ height: 6, background: "rgba(255,255,255,0.06)", borderRadius: 3, overflow: "hidden" }}>
        <div style={{
          height: "100%", width: `${pct}%`, borderRadius: 3,
          background: `linear-gradient(90deg, ${color}88, ${color})`,
          boxShadow: `0 0 12px ${color}66`,
          transition: "width 1.2s cubic-bezier(0.22,1,0.36,1)",
        }} />
      </div>
      <div style={{ display: "flex", justifyContent: "space-between", marginTop: 6 }}>
        <span style={{ fontSize: 10, color: "#374151", fontFamily: "monospace" }}>0%</span>
        <span style={{ fontSize: 12, color, fontFamily: "'Space Mono', monospace", fontWeight: 700 }}>
          {pct}%
        </span>
        <span style={{ fontSize: 10, color: "#374151", fontFamily: "monospace" }}>100%</span>
      </div>
    </div>
  );
}

function UploadZone({
  accept, label, icon, file, onFile,
}: {
  accept: string; label: string; icon: string; file: File | null; onFile: (f: File) => void;
}) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragging(false);
    const f = e.dataTransfer.files[0];
    if (f) onFile(f);
  }, [onFile]);

  return (
    <div
      onClick={() => inputRef.current?.click()}
      onDragOver={e => { e.preventDefault(); setDragging(true); }}
      onDragLeave={() => setDragging(false)}
      onDrop={handleDrop}
      style={{
        border: `1px dashed ${dragging ? "#00ffaa" : file ? "#00ffaa44" : "#1f2937"}`,
        borderRadius: 12, padding: "28px 20px", textAlign: "center",
        cursor: "pointer", transition: "all 0.2s",
        background: dragging ? "rgba(0,255,170,0.04)" : file ? "rgba(0,255,170,0.02)" : "transparent",
        position: "relative", overflow: "hidden",
      }}
    >
      <input
        ref={inputRef} type="file" accept={accept} style={{ display: "none" }}
        onChange={e => { const f = e.target.files?.[0]; if (f) onFile(f); }}
      />
      <div style={{ fontSize: 28, marginBottom: 8 }}>{icon}</div>
      {file ? (
        <div>
          <p style={{ color: "#00ffaa", fontSize: 13, margin: 0, fontFamily: "'Space Mono', monospace" }}>{file.name}</p>
          <p style={{ color: "#4b5563", fontSize: 11, margin: "4px 0 0", fontFamily: "monospace" }}>
            {(file.size / 1024 / 1024).toFixed(2)} MB
          </p>
        </div>
      ) : (
        <div>
          <p style={{ color: "#374151", fontSize: 13, margin: 0 }}>Drop {label} or click to browse</p>
          <p style={{ color: "#1f2937", fontSize: 11, margin: "4px 0 0", fontFamily: "monospace" }}>accepts {accept}</p>
        </div>
      )}
    </div>
  );
}

function ResultCard({ result }: { result: ResultData }) {
  const score = typeof result.deepfake_score === "string"
    ? parseFloat(result.deepfake_score) : result.deepfake_score;
  const isFake = result.prediction?.toLowerCase().includes("fake") || score > 0.5;

  return (
    <div style={{
      marginTop: 20, padding: "20px",
      border: `1px solid ${isFake ? "#ff2d5522" : "#00ffaa22"}`,
      borderRadius: 12, background: isFake ? "rgba(255,45,85,0.04)" : "rgba(0,255,170,0.03)",
      position: "relative", overflow: "hidden",
    }}>
      <ScanLines />
      <div style={{ position: "relative", zIndex: 1 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 4 }}>
          <div style={{
            width: 8, height: 8, borderRadius: "50%",
            background: isFake ? "#ff2d55" : "#00ffaa",
            boxShadow: `0 0 8px ${isFake ? "#ff2d55" : "#00ffaa"}`,
            animation: "pulse 2s infinite",
          }} />
          <span style={{
            fontFamily: "'Space Mono', monospace", fontSize: 13, letterSpacing: "0.1em",
            color: isFake ? "#ff2d55" : "#00ffaa",
            fontWeight: 700,
          }}>
            {result.prediction?.toUpperCase() || "UNKNOWN"}
          </span>
        </div>
        <ScoreBar score={score} />
      </div>
    </div>
  );
}

function DetectorPanel({
  type, icon, label, accept,
}: {
  type: "image" | "video"; icon: string; label: string; accept: string;
}) {
  const [file, setFile] = useState<File | null>(null);
  const [result, setResult] = useState<ResultData | null>(null);
  const [state, setState] = useState<UploadState>("idle");
  const [scanLine, setScanLine] = useState(false);

  const detect = async () => {
    if (!file) return;
    setState("loading");
    setResult(null);
    setScanLine(true);

    try {
      const formData = new FormData();
      formData.append("file", file);
      const endpoint = type === "image" ? "/detect-image" : "/detect-video";
      const res = await fetch(`${API}${endpoint}`, { method: "POST", body: formData });
      const data = await res.json();
      setResult(data.result);
      setState("done");
    } catch {
      setState("error");
    } finally {
      setScanLine(false);
    }
  };

  return (
    <div style={{
      background: "rgba(255,255,255,0.02)",
      border: "1px solid #111827",
      borderRadius: 16,
      padding: 28,
      position: "relative",
      overflow: "hidden",
    }}>
      <ScanLines />

      {scanLine && (
        <div style={{
          position: "absolute", left: 0, right: 0, height: 2,
          background: "linear-gradient(90deg, transparent, #00ffaa, transparent)",
          boxShadow: "0 0 20px #00ffaa",
          zIndex: 10,
          animation: "scanDown 1.5s linear infinite",
        }} />
      )}

      <div style={{ position: "relative", zIndex: 1 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 20 }}>
          <div style={{
            width: 36, height: 36, borderRadius: 10,
            background: "rgba(0,255,170,0.08)",
            border: "1px solid rgba(0,255,170,0.15)",
            display: "flex", alignItems: "center", justifyContent: "center",
            fontSize: 16,
          }}>
            {icon}
          </div>
          <div>
            <h2 style={{
              margin: 0, fontSize: 14, fontWeight: 600,
              fontFamily: "'Space Mono', monospace",
              color: "#e5e7eb", letterSpacing: "0.05em",
            }}>
              {label.toUpperCase()}
            </h2>
            <p style={{ margin: 0, fontSize: 11, color: "#374151", fontFamily: "monospace" }}>
              forensic analysis module
            </p>
          </div>
        </div>

        <UploadZone accept={accept} label={label} icon={icon} file={file} onFile={setFile} />

        <button
          onClick={detect}
          disabled={!file || state === "loading"}
          style={{
            width: "100%", marginTop: 14,
            padding: "13px 0",
            background: file && state !== "loading"
              ? "linear-gradient(135deg, rgba(0,255,170,0.12), rgba(0,255,170,0.06))"
              : "rgba(255,255,255,0.03)",
            border: `1px solid ${file && state !== "loading" ? "rgba(0,255,170,0.3)" : "#111827"}`,
            borderRadius: 10,
            color: file && state !== "loading" ? "#00ffaa" : "#1f2937",
            fontFamily: "'Space Mono', monospace",
            fontSize: 12, letterSpacing: "0.15em", fontWeight: 700,
            cursor: file && state !== "loading" ? "pointer" : "not-allowed",
            transition: "all 0.2s",
            position: "relative", overflow: "hidden",
          }}
        >
          {state === "loading" ? (
            <span style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: 8 }}>
              <span style={{ animation: "spin 1s linear infinite", display: "inline-block" }}>◌</span>
              ANALYZING...
            </span>
          ) : "RUN ANALYSIS"}
        </button>

        {state === "error" && (
          <div style={{
            marginTop: 16, padding: 14, borderRadius: 10,
            background: "rgba(255,45,85,0.06)",
            border: "1px solid rgba(255,45,85,0.2)",
            color: "#ff2d55", fontSize: 12,
            fontFamily: "'Space Mono', monospace", letterSpacing: "0.05em",
          }}>
            ✕ CONNECTION FAILED — ensure the API is running on localhost:8000
          </div>
        )}

        {result?.success && <ResultCard result={result} />}
      </div>
    </div>
  );
}

export default function App() {
  return (
    <>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');

        *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

        body {
          background: #030712;
          color: #e5e7eb;
          font-family: 'Syne', sans-serif;
          min-height: 100vh;
          overflow-x: hidden;
        }

        body::before {
          content: '';
          position: fixed; inset: 0;
          background:
            radial-gradient(ellipse 80% 50% at 20% 0%, rgba(0,255,170,0.04) 0%, transparent 60%),
            radial-gradient(ellipse 60% 40% at 80% 100%, rgba(255,45,85,0.03) 0%, transparent 60%);
          pointer-events: none; z-index: 0;
        }

        @keyframes glitch1 {
          0%, 85%, 100% { clip-path: inset(30% 0 50% 0); transform: translate(0); }
          86% { clip-path: inset(10% 0 70% 0); transform: translate(-3px); }
          88% { clip-path: inset(50% 0 20% 0); transform: translate(3px); }
          90% { clip-path: inset(30% 0 50% 0); transform: translate(0); }
        }
        @keyframes glitch2 {
          0%, 87%, 100% { clip-path: inset(60% 0 20% 0); transform: translate(0); }
          88% { clip-path: inset(75% 0 5% 0); transform: translate(2px); }
          90% { clip-path: inset(55% 0 30% 0); transform: translate(-2px); }
          92% { clip-path: inset(60% 0 20% 0); transform: translate(0); }
        }
        @keyframes pulse {
          0%, 100% { opacity: 1; transform: scale(1); }
          50% { opacity: 0.6; transform: scale(0.85); }
        }
        @keyframes scanDown {
          0% { top: 0; }
          100% { top: 100%; }
        }
        @keyframes spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
        @keyframes fadeUp {
          from { opacity: 0; transform: translateY(24px); }
          to { opacity: 1; transform: translateY(0); }
        }
        @keyframes flicker {
          0%, 100% { opacity: 1; }
          92% { opacity: 1; }
          93% { opacity: 0.8; }
          94% { opacity: 1; }
          96% { opacity: 0.85; }
          97% { opacity: 1; }
        }

        .app-root {
          position: relative; z-index: 1;
          max-width: 860px;
          margin: 0 auto;
          padding: 60px 24px 80px;
          animation: fadeUp 0.7s ease both;
        }

        .grid {
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: 20px;
        }

        @media (max-width: 640px) {
          .grid { grid-template-columns: 1fr; }
        }

        .header-tag {
          font-family: 'Space Mono', monospace;
          font-size: 10px;
          letter-spacing: 0.25em;
          color: #00ffaa;
          background: rgba(0,255,170,0.08);
          border: 1px solid rgba(0,255,170,0.15);
          border-radius: 4px;
          padding: 4px 10px;
          display: inline-block;
          margin-bottom: 20px;
          animation: flicker 6s infinite;
        }
      `}</style>

      <div className="app-root">
        <header style={{ marginBottom: 52, textAlign: "center" }}>
          <div className="header-tag">◈ FORENSIC AI SYSTEM v2.0</div>
          <h1 style={{
            fontSize: "clamp(36px, 6vw, 62px)",
            fontWeight: 800,
            fontFamily: "'Syne', sans-serif",
            letterSpacing: "-0.02em",
            lineHeight: 1.05,
            color: "#f9fafb",
            marginBottom: 16,
          }}>
            <GlitchText>Beyond the Pixel</GlitchText>
          </h1>
          <p style={{
            color: "#4b5563",
            fontSize: 14,
            fontFamily: "'Space Mono', monospace",
            letterSpacing: "0.05em",
            maxWidth: 420,
            margin: "0 auto",
            lineHeight: 1.7,
          }}>
            Neural deepfake detection — image &amp; video analysis<br />
            powered by forensic pattern recognition
          </p>

          <div style={{
            display: "flex", justifyContent: "center", gap: 24, marginTop: 28,
          }}>
            {["NEURAL NET", "FRAME ANALYSIS", "ARTIFACT SCAN"].map(tag => (
              <div key={tag} style={{
                fontSize: 10, fontFamily: "'Space Mono', monospace",
                color: "#1f2937", letterSpacing: "0.15em",
                borderBottom: "1px solid #111827", paddingBottom: 2,
              }}>
                {tag}
              </div>
            ))}
          </div>
        </header>

        <div className="grid">
          <DetectorPanel type="image" icon="◈" label="image detection" accept="image/*" />
          <DetectorPanel type="video" icon="▶" label="video detection" accept="video/*" />
        </div>

        <footer style={{
          marginTop: 48, textAlign: "center",
          fontFamily: "'Space Mono', monospace",
          fontSize: 10, letterSpacing: "0.15em", color: "#111827",
        }}>
          BEYOND THE PIXEL — DEEPFAKE FORENSICS SUITE
        </footer>
      </div>
    </>
  );
}