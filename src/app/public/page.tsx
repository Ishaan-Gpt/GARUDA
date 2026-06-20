"use client";

import React, { useEffect, useRef, useState, useCallback } from "react";

type Tab = "record" | "analyze" | "status";
type RecordState = "idle" | "recording" | "paused";

interface MLCard {
  id: string;
  timestamp: string;
  annotatedFrame: string;
  vehicles: number;
  persons: number;
  plates: string[];
  violations: { type: string; confidence: number }[];
  severity: string;
  violationId?: string;
}

interface Submission {
  violation_id: string;
  violation_type: string;
  plate_text: string;
  location: string;
  timestamp: string;
  status: string;
  frame?: string;
}

export default function PublicPage() {
  const [tab, setTab] = useState<Tab>("record");
  const [recordState, setRecordState] = useState<RecordState>("idle");
  const [mlCards, setMlCards] = useState<MLCard[]>([]);
  const [submissions, setSubmissions] = useState<Submission[]>([]);
  const [newAnalyzeCount, setNewAnalyzeCount] = useState(0);

  // Camera refs
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const socketRef = useRef<WebSocket | null>(null);
  const intervalRef = useRef<any>(null);
  const streamRef = useRef<MediaStream | null>(null);

  // Camera state
  const [cameraReady, setCameraReady] = useState(false);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [cameraLabel, setCameraLabel] = useState("Tap Record to start");
  const [devices, setDevices] = useState<MediaDeviceInfo[]>([]);
  const [selectedDeviceId, setSelectedDeviceId] = useState("");
  const [location, setLocation] = useState("Public Edge Capture");

  // Submit state
  const [submitting, setSubmitting] = useState<string | null>(null);
  const [submitResult, setSubmitResult] = useState<{ id: string; ok: boolean } | null>(null);

  const apiBase = () => {
    if (typeof window === "undefined") return "http://localhost:8000";
    const proto = window.location.protocol === "https:" ? "https" : "http";
    return `${proto}://${window.location.hostname}:8000`;
  };

  const wsBase = () => {
    if (typeof window === "undefined") return "ws://localhost:8000";
    const proto = window.location.protocol === "https:" ? "wss" : "ws";
    return `${proto}://${window.location.hostname}:8000`;
  };

  // Load submissions from localStorage on mount
  useEffect(() => {
    loadSubmissions();
  }, []);

  const loadSubmissions = async () => {
    try {
      const stored = localStorage.getItem("garuda_public_submissions");
      if (stored) {
        const list: Submission[] = JSON.parse(stored);
        // Refresh statuses from backend
        const refreshed = await Promise.all(
          list.map(async (item) => {
            try {
              const res = await fetch(`${apiBase()}/api/v1/violations/${item.violation_id}`);
              if (res.ok) {
                const d = await res.json();
                return { ...item, status: d.status ?? item.status };
              }
            } catch {}
            return item;
          })
        );
        setSubmissions(refreshed);
        localStorage.setItem("garuda_public_submissions", JSON.stringify(refreshed));
      }
    } catch {}
  };

  // Initialize camera — only preview, do NOT start sending frames
  const startCamera = useCallback(async (deviceId?: string) => {
    setCameraError(null);
    setCameraReady(false);
    setCameraLabel("Initializing camera...");

    // Stop previous stream
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }

    try {
      const constraints: MediaStreamConstraints = {
        video: deviceId
          ? { deviceId: { exact: deviceId }, width: { ideal: 1280 }, height: { ideal: 720 } }
          : { facingMode: { ideal: "environment" }, width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false,
      };

      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      streamRef.current = stream;

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
        const track = stream.getVideoTracks()[0];
        setCameraLabel(track.label || "Camera active");
        setCameraReady(true);

        // Enumerate devices after permission granted
        const all = await navigator.mediaDevices.enumerateDevices();
        const vids = all.filter((d) => d.kind === "videoinput");
        setDevices(vids);
        if (!deviceId && vids.length > 0) {
          // Try to auto-select back camera
          const back = vids.find(
            (d) =>
              d.label.toLowerCase().includes("back") ||
              d.label.toLowerCase().includes("rear") ||
              d.label.toLowerCase().includes("environment")
          );
          if (back && back.deviceId !== track.getSettings().deviceId) {
            setSelectedDeviceId(back.deviceId);
          }
        }
      }
    } catch (err: any) {
      // Fallback: try environment facing without exact device
      try {
        const fallback = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: "environment" },
          audio: false,
        });
        streamRef.current = fallback;
        if (videoRef.current) {
          videoRef.current.srcObject = fallback;
          await videoRef.current.play();
          setCameraLabel("Camera (fallback)");
          setCameraReady(true);
        }
      } catch (fbErr: any) {
        setCameraError(fbErr.message || "Camera access denied");
        setCameraLabel("Camera unavailable");
      }
    }
  }, []);

  // Start recording → open WS → begin frame loop
  const startRecording = useCallback(() => {
    if (!cameraReady) return;

    const ws = new WebSocket(`${wsBase()}/ws/patrol`);
    socketRef.current = ws;

    ws.onopen = () => {
      // 3fps — lightweight enough for phone
      intervalRef.current = setInterval(() => {
        const canvas = canvasRef.current;
        const video = videoRef.current;
        if (!canvas || !video || !socketRef.current || socketRef.current.readyState !== WebSocket.OPEN) return;
        const ctx = canvas.getContext("2d");
        if (!ctx) return;
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        const dataUrl = canvas.toDataURL("image/jpeg", 0.5);
        socketRef.current.send(JSON.stringify({ frame: dataUrl, camera_id: "PUBLIC-PORTAL", location }));
      }, 333);
    };

    ws.onmessage = (e) => {
      try {
        const data = JSON.parse(e.data);
        if (data.frame) {
          const card: MLCard = {
            id: `ML-${Date.now()}`,
            timestamp: new Date().toLocaleTimeString(),
            annotatedFrame: data.frame,
            vehicles: data.detections?.vehicles ?? 0,
            persons: data.detections?.persons ?? 0,
            plates: data.violation?.plate ? [data.violation.plate] : [],
            violations: data.violation
              ? [{ type: data.violation.type, confidence: data.violation.confidence }]
              : [],
            severity: data.violation ? "high" : "none",
            violationId: data.violation?.violation_id,
          };
          // Only add card if there's something detected or violation
          if (data.violation || data.detections?.vehicles > 0 || data.detections?.persons > 0) {
            setMlCards((prev) => [card, ...prev].slice(0, 30));
            if (tab !== "analyze") setNewAnalyzeCount((n) => n + 1);
          }
        }
      } catch {}
    };

    ws.onerror = () => ws.close();
    ws.onclose = () => {
      if (recordState === "recording") {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    };

    setRecordState("recording");
  }, [cameraReady, location, tab, recordState]);

  const pauseRecording = useCallback(() => {
    clearInterval(intervalRef.current);
    intervalRef.current = null;
    setRecordState("paused");
  }, []);

  const resumeRecording = useCallback(() => {
    if (!socketRef.current || socketRef.current.readyState !== WebSocket.OPEN) {
      startRecording();
      return;
    }
    intervalRef.current = setInterval(() => {
      const canvas = canvasRef.current;
      const video = videoRef.current;
      if (!canvas || !video || !socketRef.current || socketRef.current.readyState !== WebSocket.OPEN) return;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      socketRef.current.send(JSON.stringify({ frame: canvas.toDataURL("image/jpeg", 0.5), camera_id: "PUBLIC-PORTAL", location }));
    }, 333);
    setRecordState("recording");
  }, [startRecording, location]);

  const stopStream = useCallback(() => {
    clearInterval(intervalRef.current);
    intervalRef.current = null;
    if (socketRef.current) {
      socketRef.current.close();
      socketRef.current = null;
    }
    setRecordState("idle");
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopStream();
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((t) => t.stop());
      }
    };
  }, [stopStream]);

  // Re-init camera when device changes
  useEffect(() => {
    if (selectedDeviceId) startCamera(selectedDeviceId);
  }, [selectedDeviceId]);

  const handleRecordButton = () => {
    if (!cameraReady) {
      startCamera();
      return;
    }
    if (recordState === "idle") startRecording();
    else if (recordState === "recording") pauseRecording();
    else if (recordState === "paused") resumeRecording();
  };

  const submitCard = async (card: MLCard) => {
    setSubmitting(card.id);
    const plate = card.plates[0] || "UNKNOWN";
    const vtype = card.violations[0]?.type || "Unclassified";
    const vid = card.violationId || `VIO-PUB-${Date.now().toString().slice(-6)}`;

    try {
      const res = await fetch(`${apiBase()}/api/v1/violations/public-report`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          violation_id: vid,
          violation_type: vtype,
          plate_text: plate,
          location,
          severity: card.severity === "high" ? "high" : "medium",
          frame_b64: card.annotatedFrame,
        }),
      });
      if (res.ok) {
        const newSub: Submission = {
          violation_id: vid,
          violation_type: vtype,
          plate_text: plate,
          location,
          timestamp: new Date().toLocaleString(),
          status: "pending",
          frame: card.annotatedFrame,
        };
        const updated = [newSub, ...submissions];
        setSubmissions(updated);
        localStorage.setItem("garuda_public_submissions", JSON.stringify(updated));
        setSubmitResult({ id: card.id, ok: true });
      } else {
        setSubmitResult({ id: card.id, ok: false });
      }
    } catch {
      setSubmitResult({ id: card.id, ok: false });
    } finally {
      setSubmitting(null);
      setTimeout(() => setSubmitResult(null), 3000);
    }
  };

  const recordBtnLabel = () => {
    if (!cameraReady) return { emoji: "📷", text: "Start Camera" };
    if (recordState === "idle") return { emoji: "⏺", text: "Record" };
    if (recordState === "recording") return { emoji: "⏸", text: "Pause" };
    return { emoji: "▶", text: "Resume" };
  };

  const btn = recordBtnLabel();

  return (
    <div style={{ width: "100vw", minHeight: "100vh", backgroundColor: "#060a12", color: "#fff", fontFamily: "system-ui, -apple-system, sans-serif", overflowX: "hidden" }}>
      <style dangerouslySetInnerHTML={{ __html: `
        * { box-sizing: border-box; }
        body { margin: 0; }
        .tab-btn { background: none; border: none; cursor: pointer; padding: 10px 0; font-size: 12px; font-weight: 700; letter-spacing: 0.5px; flex: 1; transition: all 0.15s; }
        .tab-btn.active { color: #facc15; border-bottom: 2px solid #facc15; }
        .tab-btn.inactive { color: #64748b; border-bottom: 2px solid transparent; }
        .record-btn { border: none; cursor: pointer; border-radius: 50px; font-weight: 800; font-size: 15px; letter-spacing: 0.5px; transition: all 0.15s; }
        .record-btn:active { transform: scale(0.95); }
        .card { background: #0f1623; border: 1px solid rgba(255,255,255,0.07); border-radius: 12px; overflow: hidden; margin-bottom: 12px; }
        .badge { display: inline-flex; align-items: center; gap: 4px; padding: 3px 8px; border-radius: 20px; font-size: 10px; font-weight: 700; }
        .badge-red { background: rgba(239,68,68,0.15); color: #f87171; border: 1px solid rgba(239,68,68,0.3); }
        .badge-yellow { background: rgba(250,204,21,0.15); color: #facc15; border: 1px solid rgba(250,204,21,0.3); }
        .badge-green { background: rgba(34,197,94,0.15); color: #4ade80; border: 1px solid rgba(34,197,94,0.3); }
        .badge-gray { background: rgba(100,116,139,0.15); color: #94a3b8; border: 1px solid rgba(100,116,139,0.3); }
        .submit-btn { background: #facc15; color: #000; border: none; border-radius: 6px; padding: 8px 14px; font-size: 11px; font-weight: 800; cursor: pointer; white-space: nowrap; }
        .submit-btn:disabled { opacity: 0.5; cursor: not-allowed; }
        .submit-btn:active { transform: scale(0.97); }
        @keyframes pulse-rec { 0%,100% { opacity: 1; } 50% { opacity: 0.3; } }
        .rec-dot { animation: pulse-rec 1s infinite; display: inline-block; width: 8px; height: 8px; background: #ef4444; border-radius: 50%; }
        @media (max-width: 480px) {
          .cam-container { max-width: 100% !important; }
        }
      `}} />

      {/* Hidden canvas for frame capture */}
      <canvas ref={canvasRef} width={640} height={360} style={{ display: "none" }} />

      {/* ── TOP HEADER ── */}
      <div style={{ padding: "14px 16px 0", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <div>
          <div style={{ fontSize: 11, fontWeight: 800, color: "#facc15", letterSpacing: 2 }}>GARUDA</div>
          <div style={{ fontSize: 9, color: "#475569", letterSpacing: 1 }}>PUBLIC REPORTER</div>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
          {recordState === "recording" && (
            <span style={{ display: "flex", alignItems: "center", gap: 5, fontSize: 10, color: "#ef4444", fontWeight: 700 }}>
              <span className="rec-dot" /> LIVE
            </span>
          )}
          {cameraReady && recordState === "idle" && (
            <span className="badge badge-green" style={{ fontSize: 9 }}>● READY</span>
          )}
        </div>
      </div>

      {/* ── TAB BAR ── */}
      <div style={{ display: "flex", borderBottom: "1px solid rgba(255,255,255,0.06)", margin: "12px 0 0" }}>
        <button className={`tab-btn ${tab === "record" ? "active" : "inactive"}`} onClick={() => setTab("record")}>
          📹 RECORD
        </button>
        <button
          className={`tab-btn ${tab === "analyze" ? "active" : "inactive"}`}
          onClick={() => { setTab("analyze"); setNewAnalyzeCount(0); }}
        >
          🔍 ANALYZE{newAnalyzeCount > 0 ? ` (${newAnalyzeCount})` : ""}
        </button>
        <button className={`tab-btn ${tab === "status" ? "active" : "inactive"}`} onClick={() => { setTab("status"); loadSubmissions(); }}>
          📋 STATUS{submissions.length > 0 ? ` (${submissions.length})` : ""}
        </button>
      </div>

      {/* ══════════════════════════════════ TAB: RECORD ══════════════════════════════════ */}
      {tab === "record" && (
        <div style={{ padding: "16px 12px", display: "flex", flexDirection: "column", gap: 14 }}>

          {/* Camera Preview — natural aspect ratio, not fullscreen */}
          <div
            className="cam-container"
            style={{
              width: "100%",
              maxWidth: 500,
              margin: "0 auto",
              aspectRatio: "16/9",
              backgroundColor: "#000",
              borderRadius: 12,
              overflow: "hidden",
              position: "relative",
              border: recordState === "recording"
                ? "2px solid #ef4444"
                : cameraReady
                ? "2px solid rgba(250,204,21,0.3)"
                : "2px solid rgba(255,255,255,0.08)",
            }}
          >
            <video
              ref={videoRef}
              playsInline
              muted
              autoPlay={false}
              style={{ width: "100%", height: "100%", objectFit: "contain", display: "block", background: "#000" }}
            />

            {!cameraReady && !cameraError && (
              <div style={{ position: "absolute", inset: 0, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", gap: 8, color: "#475569" }}>
                <span style={{ fontSize: 36 }}>📷</span>
                <span style={{ fontSize: 11, fontWeight: 600 }}>Tap "Record" to activate camera</span>
              </div>
            )}

            {cameraError && (
              <div style={{ position: "absolute", inset: 0, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", gap: 8, padding: 16, textAlign: "center" }}>
                <span style={{ fontSize: 28 }}>⚠️</span>
                <span style={{ fontSize: 11, color: "#f87171", fontWeight: 600 }}>Camera Error</span>
                <span style={{ fontSize: 10, color: "#64748b" }}>{cameraError}</span>
                <button onClick={() => startCamera()} style={{ marginTop: 8, background: "#facc15", color: "#000", border: "none", borderRadius: 6, padding: "6px 14px", fontSize: 11, fontWeight: 700, cursor: "pointer" }}>
                  Retry
                </button>
              </div>
            )}

            {/* Camera label overlay */}
            {cameraReady && (
              <div style={{ position: "absolute", bottom: 0, left: 0, right: 0, background: "linear-gradient(to top, rgba(0,0,0,0.7), transparent)", padding: "12px 10px 6px", fontSize: 9, color: "#94a3b8" }}>
                {cameraLabel}
              </div>
            )}
          </div>

          {/* Location input */}
          <div style={{ maxWidth: 500, margin: "0 auto", width: "100%" }}>
            <label style={{ fontSize: 9, color: "#64748b", fontWeight: 700, letterSpacing: 1 }}>LOCATION TAG</label>
            <input
              value={location}
              onChange={(e) => setLocation(e.target.value)}
              placeholder="e.g. MG Road Junction"
              style={{ display: "block", width: "100%", marginTop: 4, padding: "8px 10px", background: "#0f1623", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 6, color: "#fff", fontSize: 12, outline: "none" }}
            />
          </div>

          {/* Camera switcher */}
          {devices.length > 1 && (
            <div style={{ maxWidth: 500, margin: "0 auto", width: "100%" }}>
              <label style={{ fontSize: 9, color: "#64748b", fontWeight: 700, letterSpacing: 1 }}>CAMERA</label>
              <select
                value={selectedDeviceId}
                onChange={(e) => setSelectedDeviceId(e.target.value)}
                style={{ display: "block", width: "100%", marginTop: 4, padding: "8px 10px", background: "#0f1623", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 6, color: "#fff", fontSize: 11, outline: "none" }}
              >
                {devices.map((d, i) => (
                  <option key={d.deviceId} value={d.deviceId} style={{ color: "#000" }}>
                    {d.label || `Camera ${i + 1}`}
                  </option>
                ))}
              </select>
            </div>
          )}

          {/* Record / Pause / Resume button */}
          <div style={{ display: "flex", gap: 10, maxWidth: 500, margin: "0 auto", width: "100%" }}>
            <button
              className="record-btn"
              onClick={handleRecordButton}
              style={{
                flex: 1,
                padding: "14px",
                background: recordState === "recording"
                  ? "#ef4444"
                  : recordState === "paused"
                  ? "#f59e0b"
                  : "#facc15",
                color: "#000",
              }}
            >
              {btn.emoji} {btn.text}
            </button>

            {(recordState === "recording" || recordState === "paused") && (
              <button
                className="record-btn"
                onClick={stopStream}
                style={{ padding: "14px 20px", background: "rgba(255,255,255,0.08)", color: "#fff", border: "1px solid rgba(255,255,255,0.12)" }}
              >
                ■ Stop
              </button>
            )}
          </div>

          {/* Info box */}
          <div style={{ maxWidth: 500, margin: "0 auto", width: "100%", background: "rgba(250,204,21,0.05)", border: "1px solid rgba(250,204,21,0.15)", borderRadius: 8, padding: "10px 12px" }}>
            <div style={{ fontSize: 10, color: "#94a3b8", lineHeight: 1.6 }}>
              ① Tap <strong style={{ color: "#facc15" }}>Record</strong> to start live ML analysis.<br />
              ② ML detects violations, faces &amp; plates in real time.<br />
              ③ Go to <strong style={{ color: "#facc15" }}>Analyze</strong> tab to review &amp; submit evidence.
            </div>
          </div>
        </div>
      )}

      {/* ══════════════════════════════════ TAB: ANALYZE ══════════════════════════════════ */}
      {tab === "analyze" && (
        <div style={{ padding: "16px 12px", display: "flex", flexDirection: "column", gap: 12 }}>

          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <div>
              <div style={{ fontSize: 12, fontWeight: 800, color: "#facc15" }}>ML EVIDENCE CARDS</div>
              <div style={{ fontSize: 9, color: "#475569", marginTop: 2 }}>{mlCards.length} frames analyzed</div>
            </div>
            {mlCards.length > 0 && (
              <button onClick={() => setMlCards([])} style={{ background: "none", border: "1px solid rgba(255,255,255,0.1)", color: "#64748b", borderRadius: 6, padding: "5px 10px", fontSize: 10, cursor: "pointer" }}>
                Clear
              </button>
            )}
          </div>

          {mlCards.length === 0 ? (
            <div style={{ textAlign: "center", padding: "60px 20px", color: "#334155" }}>
              <div style={{ fontSize: 40, marginBottom: 10 }}>🔍</div>
              <div style={{ fontSize: 12, fontWeight: 700, color: "#475569" }}>No ML results yet</div>
              <div style={{ fontSize: 10, color: "#334155", marginTop: 4 }}>Start recording to begin live analysis</div>
              <button onClick={() => setTab("record")} style={{ marginTop: 14, background: "#facc15", color: "#000", border: "none", borderRadius: 8, padding: "10px 20px", fontSize: 12, fontWeight: 800, cursor: "pointer" }}>
                Go to Record
              </button>
            </div>
          ) : (
            mlCards.map((card) => (
              <div key={card.id} className="card">
                {/* Annotated frame */}
                <div style={{ width: "100%", aspectRatio: "16/9", background: "#000", overflow: "hidden" }}>
                  <img src={card.annotatedFrame} alt="ML annotated" style={{ width: "100%", height: "100%", objectFit: "contain", display: "block" }} />
                </div>

                {/* Card body */}
                <div style={{ padding: "12px" }}>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 10 }}>
                    <div style={{ fontSize: 10, color: "#475569" }}>{card.timestamp}</div>
                    <div style={{ display: "flex", gap: 4, flexWrap: "wrap", justifyContent: "flex-end" }}>
                      {card.violations.length > 0 && (
                        <span className="badge badge-red">⚠ {card.violations[0].type}</span>
                      )}
                      {card.violations.length === 0 && card.vehicles > 0 && (
                        <span className="badge badge-green">✓ No Violation</span>
                      )}
                    </div>
                  </div>

                  {/* Detection grid */}
                  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 8, marginBottom: 12 }}>
                    <div style={{ background: "rgba(255,255,255,0.04)", borderRadius: 6, padding: "8px 6px", textAlign: "center" }}>
                      <div style={{ fontSize: 18, fontWeight: 800, color: "#facc15" }}>{card.vehicles}</div>
                      <div style={{ fontSize: 9, color: "#475569", marginTop: 2 }}>VEHICLES</div>
                    </div>
                    <div style={{ background: "rgba(255,255,255,0.04)", borderRadius: 6, padding: "8px 6px", textAlign: "center" }}>
                      <div style={{ fontSize: 18, fontWeight: 800, color: "#60a5fa" }}>{card.persons}</div>
                      <div style={{ fontSize: 9, color: "#475569", marginTop: 2 }}>PERSONS</div>
                    </div>
                    <div style={{ background: "rgba(255,255,255,0.04)", borderRadius: 6, padding: "8px 6px", textAlign: "center" }}>
                      <div style={{ fontSize: 18, fontWeight: 800, color: card.plates.length > 0 ? "#4ade80" : "#334155" }}>
                        {card.plates.length > 0 ? "✓" : "—"}
                      </div>
                      <div style={{ fontSize: 9, color: "#475569", marginTop: 2 }}>PLATE</div>
                    </div>
                  </div>

                  {/* Plate & violation details */}
                  {card.plates.length > 0 && (
                    <div style={{ background: "#0a0f1a", borderRadius: 6, padding: "6px 10px", marginBottom: 8, fontFamily: "monospace", fontSize: 14, fontWeight: 800, color: "#facc15", letterSpacing: 2 }}>
                      {card.plates[0]}
                    </div>
                  )}

                  {card.violations.map((v, i) => (
                    <div key={i} style={{ display: "flex", justifyContent: "space-between", alignItems: "center", fontSize: 11, color: "#e2e8f0", marginBottom: 4 }}>
                      <span>🚨 {v.type}</span>
                      <span style={{ color: "#94a3b8", fontSize: 10 }}>{v.confidence.toFixed(1)}% conf</span>
                    </div>
                  ))}

                  {/* Submit button */}
                  {card.violations.length > 0 && (
                    <div style={{ marginTop: 10, display: "flex", gap: 8, alignItems: "center" }}>
                      <button
                        className="submit-btn"
                        disabled={submitting === card.id}
                        onClick={() => submitCard(card)}
                        style={{ flex: 1 }}
                      >
                        {submitting === card.id ? "Submitting..." : "📤 Dispatch as Evidence"}
                      </button>
                      {submitResult?.id === card.id && (
                        <span style={{ fontSize: 10, color: submitResult.ok ? "#4ade80" : "#f87171", fontWeight: 700 }}>
                          {submitResult.ok ? "✓ Sent" : "✗ Failed"}
                        </span>
                      )}
                    </div>
                  )}
                </div>
              </div>
            ))
          )}
        </div>
      )}

      {/* ══════════════════════════════════ TAB: STATUS ══════════════════════════════════ */}
      {tab === "status" && (
        <div style={{ padding: "16px 12px", display: "flex", flexDirection: "column", gap: 12 }}>

          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <div>
              <div style={{ fontSize: 12, fontWeight: 800, color: "#facc15" }}>SUBMITTED EVIDENCE</div>
              <div style={{ fontSize: 9, color: "#475569", marginTop: 2 }}>{submissions.length} reports filed</div>
            </div>
            <button onClick={loadSubmissions} style={{ background: "none", border: "1px solid rgba(255,255,255,0.1)", color: "#64748b", borderRadius: 6, padding: "5px 10px", fontSize: 10, cursor: "pointer" }}>
              ↻ Refresh
            </button>
          </div>

          {submissions.length === 0 ? (
            <div style={{ textAlign: "center", padding: "60px 20px", color: "#334155" }}>
              <div style={{ fontSize: 40, marginBottom: 10 }}>📋</div>
              <div style={{ fontSize: 12, fontWeight: 700, color: "#475569" }}>No submissions yet</div>
              <div style={{ fontSize: 10, color: "#334155", marginTop: 4 }}>Submit violations from the Analyze tab</div>
            </div>
          ) : (
            submissions.map((s, i) => {
              const statusColor =
                s.status === "confirmed" || s.status === "auto_challan"
                  ? "#4ade80"
                  : s.status === "rejected"
                  ? "#f87171"
                  : "#facc15";
              const statusLabel =
                s.status === "confirmed" || s.status === "auto_challan"
                  ? "Verified ✓"
                  : s.status === "rejected"
                  ? "Rejected ✗"
                  : "Pending Review";

              return (
                <div key={i} className="card" style={{ padding: 12 }}>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
                    <div style={{ flex: 1 }}>
                      <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 4 }}>
                        <span style={{ fontFamily: "monospace", fontWeight: 800, fontSize: 13, color: "#f3f4f6" }}>
                          {s.plate_text || "NO PLATE"}
                        </span>
                        <span className="badge badge-yellow" style={{ fontSize: 9 }}>{s.violation_type}</span>
                      </div>
                      <div style={{ fontSize: 9, color: "#475569" }}>
                        {s.violation_id} · {s.location}
                      </div>
                      <div style={{ fontSize: 9, color: "#334155", marginTop: 2 }}>{s.timestamp}</div>
                    </div>
                    <div style={{ textAlign: "right" }}>
                      <div style={{ fontSize: 10, fontWeight: 700, color: statusColor }}>{statusLabel}</div>
                    </div>
                  </div>
                </div>
              );
            })
          )}
        </div>
      )}

      {/* Bottom padding for safe area */}
      <div style={{ height: 32 }} />
    </div>
  );
}
