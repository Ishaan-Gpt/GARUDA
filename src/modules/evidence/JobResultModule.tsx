"use client";

import React, { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { usePlatform } from "@/context/PlatformContext";
import { ChevronLeftIcon, DownloadIcon } from "@/components/Icons";
import { SEVERITY_COLOR, VIOLATION_SEVERITY_BY_TYPE } from "@/lib/violations";
import {
  fetchJobResult,
  aggregateRecords,
  evidenceFileUrl,
  JobResult,
  PipelineRecord,
  StageAggregate,
} from "@/lib/evidence";

const TIER_LABEL: Record<number, string> = {
  1: "TIER 1 — AUTO CHALLAN",
  2: "TIER 2 — HUMAN REVIEW",
  3: "TIER 3 — LOGGED / DISCARDED",
};

const NOISE_PLATE_TEXTS = ["UNCLEAR", "PLATE-UNREAD", ""];

function RecordCard({ record, index }: { record: PipelineRecord; index: number }) {
  const [activeTab, setActiveTab] = useState<"annotated" | "demo" | "raw">("annotated");
  const paths = {
    annotated: record.evidence?.annotated_image,
    demo: record.evidence?.demo_image,
    raw: record.evidence?.raw_frame,
  };
  const path = paths[activeTab];
  const compliant = (record.violations || []).length === 0;
  const readablePlates = (record.all_plates_detected || []).filter(
    (p) => p.confidence > 0 && !NOISE_PLATE_TEXTS.includes(p.plate_text)
  );

  return (
    <div className="card" style={{ transform: "none" }}>
      <div className="card-title">
        <span>
          ITEM #{index + 1} — <span className="mono">{record.violation_id}</span>
        </span>
        <span className={`badge ${record.tier === 1 ? "approved" : record.tier === 2 ? "review" : "disabled"}`}>
          {TIER_LABEL[record.tier] || `TIER ${record.tier}`}
        </span>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 280px", gap: "10px" }}>
        <div className="evidence-pane" style={{ minHeight: "0" }}>
          <div className="evidence-stages">
            <div className={`stage-tab ${activeTab === "annotated" ? "active" : ""}`} onClick={() => setActiveTab("annotated")}>
              ANNOTATED
            </div>
            <div className={`stage-tab ${activeTab === "demo" ? "active" : ""}`} onClick={() => setActiveTab("demo")}>
              DEMO
            </div>
            <div className={`stage-tab ${activeTab === "raw" ? "active" : ""}`} onClick={() => setActiveTab("raw")}>
              RAW
            </div>
          </div>
          <div className="stage-viewer" style={{ minHeight: "220px" }}>
            {path ? (
              <img
                src={evidenceFileUrl(path)}
                alt={`${activeTab} evidence for ${record.violation_id}`}
                style={{ maxWidth: "100%", maxHeight: "100%", objectFit: "contain", display: "block" }}
              />
            ) : (
              <div style={{ color: "#94a3b8", fontSize: "12px", textAlign: "center" }}>
                No {activeTab} frame stored.
              </div>
            )}
          </div>
        </div>

        <div style={{ display: "flex", flexDirection: "column", gap: "8px", fontSize: "11px" }}>
          <div style={{ display: "flex", justifyContent: "space-between" }}>
            <span style={{ color: "var(--text-muted)" }}>Vehicle Class:</span>
            <span style={{ fontWeight: "600" }}>{record.vehicle?.vehicle_class || "—"}</span>
          </div>
          <div style={{ display: "flex", justifyContent: "space-between" }}>
            <span style={{ color: "var(--text-muted)" }}>Plate:</span>
            <span className="mono" style={{ fontWeight: "700" }}>
              {NOISE_PLATE_TEXTS.includes(record.vehicle?.license_plate || "") ? "Not Readable" : record.vehicle.license_plate}
            </span>
          </div>
          {!NOISE_PLATE_TEXTS.includes(record.vehicle?.license_plate || "") && (
            <div style={{ display: "flex", justifyContent: "space-between" }}>
              <span style={{ color: "var(--text-muted)" }}>Plate Confidence:</span>
              <span className="mono">{Math.round((record.vehicle?.plate_confidence || 0) * 100)}%</span>
            </div>
          )}
          <div style={{ display: "flex", justifyContent: "space-between" }}>
            <span style={{ color: "var(--text-muted)" }}>Timestamp:</span>
            <span className="mono" style={{ fontSize: "10px" }}>{new Date(record.timestamp).toLocaleString()}</span>
          </div>

          <div style={{ borderTop: "1px solid var(--border-color)", paddingTop: "6px" }}>
            <div style={{ fontWeight: "700", fontSize: "10px", textTransform: "uppercase", color: "var(--text-muted)", marginBottom: "4px" }}>
              Violations {compliant && <span style={{ color: "var(--success)" }}>— None (Compliant)</span>}
            </div>
            {record.violations.map((v, i) => (
              <div key={i} style={{ display: "flex", flexDirection: "column", marginBottom: "6px" }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
                  <span style={{ color: SEVERITY_COLOR[v.severity] || "var(--text-secondary)", fontWeight: "600" }}>{v.type}</span>
                  <span className="mono">{Math.round(v.confidence * 100)}%</span>
                </div>
                {v.plate_text && v.plate_text !== "UNCLEAR" && (
                  <span style={{ fontSize: "9px", color: "var(--success)", fontWeight: "600", marginTop: "1px" }}>
                    Associated Plate: {v.plate_text}
                  </span>
                )}
              </div>
            ))}
          </div>

          {readablePlates.length > 0 && (
            <div style={{ borderTop: "1px solid var(--border-color)", paddingTop: "6px" }}>
              <div style={{ fontWeight: "700", fontSize: "10px", textTransform: "uppercase", color: "var(--text-muted)", marginBottom: "4px" }}>
                All Plates In Frame ({readablePlates.length})
              </div>
              <div style={{ display: "flex", gap: "4px", flexWrap: "wrap" }}>
                {readablePlates.map((p, i) => (
                  <span key={i} className="mono" style={{ fontSize: "10px", border: "1px solid var(--border-color)", borderRadius: "4px", padding: "2px 5px" }}>
                    {p.plate_text} ({Math.round(p.confidence * 100)}%)
                  </span>
                ))}
              </div>
            </div>
          )}

          {record.driver_state?.alerts?.length > 0 && (
            <div style={{ borderTop: "1px solid var(--border-color)", paddingTop: "6px" }}>
              <div style={{ fontWeight: "700", fontSize: "10px", textTransform: "uppercase", color: "var(--text-muted)", marginBottom: "4px" }}>
                Driver State Alerts
              </div>
              {record.driver_state.alerts.map((a, i) => (
                <div key={i} style={{ display: "flex", justifyContent: "space-between" }}>
                  <span>{a.alert_type.replace(/_/g, " ")}</span>
                  <span className="mono">{Math.round(a.confidence * 100)}%</span>
                </div>
              ))}
            </div>
          )}

          <div style={{ borderTop: "1px solid var(--border-color)", paddingTop: "6px", fontSize: "10px", color: "var(--text-muted)" }}>
            {record.processing?.model} · {record.processing?.inference_time_ms}ms · {record.processing?.vehicles_detected}V/{record.processing?.persons_detected}P
            {record.processing?.camera_calibrated && <span style={{ color: "var(--text-accent)" }}> · CALIBRATED</span>}
          </div>
        </div>
      </div>
    </div>
  );
}

function StageCard({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="card" style={{ transform: "none", marginBottom: "8px" }}>
      <div className="card-title" style={{ marginBottom: "6px", paddingBottom: "4px" }}>
        <span>{title}</span>
      </div>
      <div style={{ fontSize: "11px", display: "flex", flexDirection: "column", gap: "4px" }}>{children}</div>
    </div>
  );
}

function Row({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div style={{ display: "flex", justifyContent: "space-between" }}>
      <span style={{ color: "var(--text-muted)" }}>{label}</span>
      <span style={{ fontWeight: "600" }}>{value}</span>
    </div>
  );
}

function PipelineBreakdown({ agg, job }: { agg: StageAggregate; job: JobResult["job"] }) {
  return (
    <>
      <StageCard title="1. PREPROCESSOR">
        <Row label="Source Type" value={job.source_type} />
        <Row label="Items Processed" value={agg.itemCount} />
        <Row label="Job Duration" value={`${job.duration}s`} />
      </StageCard>

      <StageCard title="2. DETECTOR">
        <Row label="Model(s) Used" value={agg.modelsUsed.join(", ") || "—"} />
        <Row label="Total Vehicles Detected" value={agg.totalVehicles} />
        <Row label="Total Persons Detected" value={agg.totalPersons} />
        <Row label="Avg Inference Time" value={`${agg.itemCount ? Math.round(agg.totalInferenceMs / agg.itemCount) : 0}ms`} />
      </StageCard>

      <StageCard title="3. TRACKER">
        <Row label="Track-Based Detections Fired" value={agg.tracked ? "YES" : "NO (static fallback only)"} />
        <Row label="Calibrated Items" value={`${agg.calibratedCount} / ${agg.itemCount}`} />
      </StageCard>

      <StageCard title="4. VIOLATION CLASSIFIER (9 Checks)">
        {agg.violationsByType.length === 0 ? (
          <span style={{ color: "var(--success)" }}>No violations fired across any item.</span>
        ) : (
          agg.violationsByType.map((vt) => (
            <div key={vt.type} style={{ marginBottom: "4px" }}>
              <div style={{ display: "flex", justifyContent: "space-between" }}>
                <span style={{ fontWeight: "700", color: SEVERITY_COLOR[VIOLATION_SEVERITY_BY_TYPE[vt.type]] }}>{vt.type}</span>
                <span className="mono">{vt.count}x</span>
              </div>
              <div style={{ fontSize: "10px", color: "var(--text-muted)" }}>
                {Object.entries(vt.methods).map(([m, c]) => `${m} (${c})`).join(", ")}
              </div>
            </div>
          ))
        )}
      </StageCard>

      <StageCard title="5. DRIVER STATE">
        {Object.keys(agg.driverAlertsByType).length === 0 ? (
          <span style={{ color: "var(--text-muted)" }}>No driver-state alerts.</span>
        ) : (
          Object.entries(agg.driverAlertsByType).map(([type, count]) => (
            <Row key={type} label={type.replace(/_/g, " ")} value={`${count}x`} />
          ))
        )}
      </StageCard>

      <StageCard title="6. OCR (PLATE RECOGNITION)">
        <Row label="OCR Engine(s)" value={agg.ocrEngine || "—"} />
        <Row label="Plates Read" value={agg.platesRead.length} />
        {agg.platesRead.slice(0, 8).map((p, i) => (
          <div key={i} style={{ display: "flex", justifyContent: "space-between", fontSize: "10px" }}>
            <span className="mono">{p.text}</span>
            <span>{Math.round(p.confidence * 100)}% {p.valid ? "✓" : "✕"}</span>
          </div>
        ))}
      </StageCard>

      <StageCard title="7. CONFIDENCE ROUTER">
        <Row label="Tier 1 (Auto-Challan)" value={agg.tierCounts.tier1} />
        <Row label="Tier 2 (Human Review)" value={agg.tierCounts.tier2} />
        <Row label="Tier 3 (Logged/Discarded)" value={agg.tierCounts.tier3} />
      </StageCard>

      <StageCard title="8. EVIDENCE PACKAGER">
        <Row label="Evidence Folder" value={<span className="mono" style={{ fontSize: "10px" }}>{agg.evidenceFolder || "—"}</span>} />
        <Row label="Files Generated" value={agg.fileCount} />
      </StageCard>
    </>
  );
}

export default function JobResultModule({ jobId }: { jobId: string }) {
  const { token } = usePlatform();
  const router = useRouter();
  const [result, setResult] = useState<JobResult | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    fetchJobResult(jobId, token)
      .then((r) => { if (!cancelled) setResult(r); })
      .catch((e) => { if (!cancelled) setError(e instanceof Error ? e.message : "Failed to load result."); })
      .finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, [jobId, token]);

  const handleDownload = () => {
    if (!result) return;
    const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(result, null, 2));
    const a = document.createElement("a");
    a.setAttribute("href", dataStr);
    a.setAttribute("download", `GARUDA_RESULT_${jobId}.json`);
    document.body.appendChild(a);
    a.click();
    a.remove();
  };

  if (loading) {
    return <div style={{ padding: "60px", textAlign: "center", color: "var(--text-muted)" }}>Loading clubbed result…</div>;
  }
  if (error || !result) {
    return <div style={{ padding: "60px", textAlign: "center", color: "var(--danger)" }}>{error || "Job result not found."}</div>;
  }

  const agg = aggregateRecords(result.records);
  const totalViolations = agg.violationsByType.reduce((sum, v) => sum + v.count, 0);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "12px" }}>

      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
        <div>
          <button className="btn btn-secondary btn-sm" onClick={() => router.push("/evidence")} style={{ marginBottom: "6px" }}>
            <ChevronLeftIcon size={12} /> BACK
          </button>
          <h1 style={{ fontSize: "20px", fontWeight: "700", letterSpacing: "-0.5px" }}>{result.job.name}</h1>
          <p style={{ fontSize: "11px", color: "var(--text-muted)", textTransform: "uppercase", marginTop: "2px" }}>
            <span className="mono">{result.job.id}</span> · {result.job.source_type} · Clubbed Result
          </p>
        </div>
        <button className="btn btn-secondary btn-sm" onClick={handleDownload}>
          <DownloadIcon size={12} /> DOWNLOAD FULL RESULT
        </button>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "repeat(5, 1fr)", gap: "10px" }}>
        <div className="metric-card">
          <div className="metric-title">Items Processed</div>
          <div className="metric-value">{agg.itemCount}</div>
        </div>
        <div className="metric-card">
          <div className="metric-title">Total Violations</div>
          <div className="metric-value">{totalViolations}</div>
        </div>
        <div className="metric-card">
          <div className="metric-title">Auto-Challan (T1)</div>
          <div className="metric-value">{agg.tierCounts.tier1}</div>
        </div>
        <div className="metric-card">
          <div className="metric-title">Human Review (T2)</div>
          <div className="metric-value">{agg.tierCounts.tier2}</div>
        </div>
        <div className="metric-card">
          <div className="metric-title">Total Inference Time</div>
          <div className="metric-value">{agg.totalInferenceMs}ms</div>
        </div>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "320px 1fr", gap: "12px", alignItems: "start" }}>
        <div>
          <PipelineBreakdown agg={agg} job={result.job} />
        </div>

        <div style={{ display: "flex", flexDirection: "column", gap: "10px" }}>
          {result.records.length === 0 ? (
            <div className="card" style={{ textAlign: "center", color: "var(--text-muted)", padding: "40px" }}>
              No records in this job's result summary.
            </div>
          ) : (
            result.records.map((r, i) => <RecordCard key={r.violation_id || i} record={r} index={i} />)
          )}
        </div>
      </div>
    </div>
  );
}
