"use client";

import React, { useState, useEffect, useMemo, useCallback } from "react";
import { useSearchParams } from "next/navigation";
import { usePlatform, Violation } from "@/context/PlatformContext";
import { CloseIcon, DownloadIcon } from "@/components/Icons";
import { STATUS_LABELS, STATUS_BADGE_CLASS, SEVERITY_COLOR } from "@/lib/violations";
import { getApiBase, getMediaBase } from "@/lib/evidence";

interface ReviewLogEntry {
  id: number;
  timestamp: string;
  actor: string;
  action: string;
  target: string;
  details: string;
}

export default function ReviewModule() {
  const { violations, role, reviewViolation, reviewViolationItem, fetchViolationDetail, deleteViolation, batchDeleteViolations, sendChallanSms } = usePlatform();
  const searchParams = useSearchParams();

  const [selectedId, setSelectedId] = useState<string>("");
  const [activeTab, setActiveTab] = useState<"annotated" | "raw">("annotated");
  const [correctedPlate, setCorrectedPlate] = useState("");
  const [showRejectModal, setShowRejectModal] = useState(false);
  const [rejectionReason, setRejectionReason] = useState("");
  const [reviewHistory, setReviewHistory] = useState<ReviewLogEntry[]>([]);
  const [detailLoading, setDetailLoading] = useState(false);

  const mediaBase = getMediaBase();

  const filteredViolations = useMemo(() => {
    const seenIds = new Set<string>();
    const seenPlates = new Set<string>();
    return violations.filter(v => {
      const plateText = v.plateNumber ? v.plateNumber.trim() : "";
      if (!plateText) return false; // No plate detected/cropped
      
      const ocrFailed = plateText.toUpperCase().includes("UNREAD") || plateText === "";
      const lowAccuracy = v.plateConfidence < 60;
      if (!ocrFailed && !lowAccuracy) return false;
      
      if (seenIds.has(v.id)) return false;
      seenIds.add(v.id);
      
      // Deduplicate by plate number if it was read
      if (!ocrFailed) {
        const plateUpper = plateText.toUpperCase();
        if (seenPlates.has(plateUpper)) return false;
        seenPlates.add(plateUpper);
      }
      return true;
    });
  }, [violations]);

  const pendingQueue = useMemo(() => {
    return filteredViolations.filter(v => v.status === "pending");
  }, [filteredViolations]);

  // Sync selectedId with URL parameter if present
  useEffect(() => {
    const urlId = searchParams?.get("id");
    if (urlId && filteredViolations.some(v => v.id === urlId)) {
      setSelectedId(urlId);
    } else if (filteredViolations.length > 0 && !selectedId) {
      const pending = filteredViolations.find(v => v.status === "pending");
      setSelectedId(pending ? pending.id : filteredViolations[0].id);
    }
  }, [searchParams, filteredViolations, selectedId]);

  const selectedViolation = useMemo(() => {
    return filteredViolations.find(v => v.id === selectedId) || filteredViolations[0] || null;
  }, [filteredViolations, selectedId]);

  // Pull the full evidence record (real OCR detections, driver alerts,
  // processing info) for the active violation — the synced list only carries
  // summary fields.
  useEffect(() => {
    if (!selectedId) return;
    setDetailLoading(true);
    fetchViolationDetail(selectedId).finally(() => setDetailLoading(false));
  }, [selectedId, fetchViolationDetail]);

  // Real review/audit history for this violation
  useEffect(() => {
    if (!selectedId) return;
    fetch(`${getApiBase()}/reviews`)
      .then(res => res.ok ? res.json() : [])
      .then((data: ReviewLogEntry[]) => setReviewHistory(data.filter(r => r.target === selectedId)))
      .catch(() => setReviewHistory([]));
  }, [selectedId]);

  useEffect(() => {
    if (selectedViolation) {
      setCorrectedPlate(selectedViolation.plateNumber);
      setActiveTab("annotated");
    }
  }, [selectedViolation?.id]);

  const advanceToNext = useCallback(() => {
    const next = pendingQueue.find(v => v.id !== selectedViolation?.id);
    if (next) setSelectedId(next.id);
  }, [pendingQueue, selectedViolation]);

  const handleDeleteCurrent = async () => {
    if (!selectedViolation) return;
    if (!window.confirm(`Delete active violation event ${selectedViolation.id}?`)) return;
    try {
      const activeId = selectedViolation.id;
      // Find the next one in violations to inspect
      const remaining = filteredViolations.filter(v => v.id !== activeId);
      await deleteViolation(activeId);
      if (remaining.length > 0) {
        const pending = remaining.find(v => v.status === "pending");
        setSelectedId(pending ? pending.id : remaining[0].id);
      } else {
        setSelectedId("");
      }
    } catch (err) {
      alert("Delete failed: " + (err instanceof Error ? err.message : String(err)));
    }
  };

  const handleDeleteAllPending = async () => {
    if (pendingQueue.length === 0) return;
    if (!window.confirm(`Delete all ${pendingQueue.length} pending violations in the queue?`)) return;
    try {
      const pendingIds = pendingQueue.map(v => v.id);
      const remaining = filteredViolations.filter(v => !pendingIds.includes(v.id));
      await batchDeleteViolations(pendingIds);
      if (remaining.length > 0) {
        setSelectedId(remaining[0].id);
      } else {
        setSelectedId("");
      }
    } catch (err) {
      alert("Batch delete failed: " + (err instanceof Error ? err.message : String(err)));
    }
  };

  const handleApprove = () => {
    if (!selectedViolation) return;
    const plateChanged = correctedPlate.toUpperCase() !== selectedViolation.plateNumber.toUpperCase();
    reviewViolation(
      selectedViolation.id,
      "Approved",
      `Officer ${role}`,
      plateChanged ? `Plate OCR corrected to ${correctedPlate}` : undefined,
      plateChanged ? correctedPlate : undefined,
    );
    advanceToNext();
  };

  // One image can carry several distinct violations (e.g. helmet + triple
  // riding on the same frame) — each is approved/rejected independently,
  // without leaving this page. Only advance to the next pending case once
  // every item in this record has a decision.
  const handleItemDecision = (itemIndex: number, action: "Approved" | "Rejected") => {
    if (!selectedViolation) return;
    reviewViolationItem(selectedViolation.id, itemIndex, action, `Officer ${role}`);
    const items = selectedViolation.violationItems || [];
    const stillPending = items.some((it, i) => i !== itemIndex && it.reviewStatus === "pending");
    if (!stillPending) advanceToNext();
  };

  const handleEscalate = () => {
    if (!selectedViolation) return;
    reviewViolation(
      selectedViolation.id,
      "Escalated",
      `Officer ${role}`,
      "Flagged for supervisor audit: verification required",
    );
  };

  const handleRejectSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!rejectionReason || !selectedViolation) return;
    reviewViolation(selectedViolation.id, "Rejected", `Officer ${role}`, rejectionReason);
    setShowRejectModal(false);
    setRejectionReason("");
    advanceToNext();
  };

  const handleDownloadPack = () => {
    if (!selectedViolation) return;
    const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(selectedViolation, null, 2));
    const downloadAnchor = document.createElement('a');
    downloadAnchor.setAttribute("href", dataStr);
    downloadAnchor.setAttribute("download", `GARUDA_EVIDENCE_${selectedViolation.id}.json`);
    document.body.appendChild(downloadAnchor);
    downloadAnchor.click();
    downloadAnchor.remove();
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "12px", height: "100%" }}>

      {/* Top Header */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
        <div>
          <h1 style={{ fontSize: "20px", fontWeight: "700", letterSpacing: "-0.5px" }}>HUMAN VERIFICATION DESK</h1>
          <p style={{ fontSize: "11px", color: "var(--text-muted)", textTransform: "uppercase", marginTop: "2px" }}>
            Audit AI-flagged violations against real evidence, override OCR readings, and issue or void citations
          </p>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
          <div style={{ fontSize: "12px", fontWeight: "600", color: "var(--text-accent)" }}>
            Pending In Queue: <span className="mono" style={{ padding: "1px 6px", background: "var(--border-accent)", borderRadius: "4px" }}>{pendingQueue.length}</span>
          </div>
          <button
            className="btn btn-danger btn-sm"
            onClick={handleDeleteCurrent}
            disabled={!selectedViolation}
            title="Delete current violation case from system"
          >
            ✕ CLEAR CURRENT
          </button>
          <button
            className="btn btn-danger btn-sm"
            onClick={handleDeleteAllPending}
            disabled={pendingQueue.length === 0}
            title="Delete all pending violations in the queue"
          >
            ✕ BATCH CLEAR PENDING
          </button>
        </div>
      </div>

      <div className="review-desk">

        {/* Left Column: Real Evidence Viewer */}
        <section className="evidence-pane">
          {selectedViolation ? (
            <>
              <div className="evidence-header">
                <div>
                  <span style={{ fontSize: "11px", color: "var(--text-muted)" }}>ACTIVE AUDIT:</span>
                  <span className="mono" style={{ fontWeight: "700", marginLeft: "4px" }}>{selectedViolation.id}</span>
                </div>
                <button className="btn btn-secondary btn-sm" onClick={handleDownloadPack} title="Export evidence record as JSON">
                  <DownloadIcon size={12} /> DOWNLOAD EVIDENCE RECORD
                </button>
              </div>

              <div className="evidence-stages">
                <div className={`stage-tab ${activeTab === "annotated" ? "active" : ""}`} onClick={() => setActiveTab("annotated")}>
                  ANNOTATED EVIDENCE
                </div>
                <div className={`stage-tab ${activeTab === "raw" ? "active" : ""}`} onClick={() => setActiveTab("raw")}>
                  RAW FRAME
                </div>
              </div>

              <div className="stage-viewer">
                {(() => {
                  const path = activeTab === "annotated" ? selectedViolation.annotatedImg : selectedViolation.rawImg;
                  if (!path) {
                    return (
                      <div style={{ color: "#94a3b8", fontSize: "12px", textAlign: "center" }}>
                        No {activeTab === "annotated" ? "annotated" : "raw"} frame stored for this event.
                      </div>
                    );
                  }
                  return (
                    <img
                      src={`${mediaBase}${path}`}
                      alt={`${activeTab} evidence for ${selectedViolation.id}`}
                      style={{ maxWidth: "100%", maxHeight: "100%", objectFit: "contain", display: "block" }}
                    />
                  );
                })()}
              </div>

              {/* Real per-plate OCR detections from this frame */}
              {selectedViolation.allPlatesDetected && selectedViolation.allPlatesDetected.length > 0 && (
                <div style={{ padding: "10px 12px", borderTop: "1px solid var(--border-color)", backgroundColor: "#FCFCFC" }}>
                  <div style={{ fontSize: "10px", fontWeight: "700", color: "var(--text-muted)", marginBottom: "6px", textTransform: "uppercase" }}>
                    All Plates Detected In Frame ({selectedViolation.allPlatesDetected.length})
                  </div>
                  <div style={{ display: "flex", gap: "6px", flexWrap: "wrap" }}>
                    {selectedViolation.allPlatesDetected.map((p, i) => (
                      <div key={i} style={{
                        display: "flex", alignItems: "center", gap: "6px",
                        border: "1px solid var(--border-color)", borderRadius: "4px",
                        padding: "4px 8px", fontSize: "10px", backgroundColor: "#fff"
                      }}>
                        <span className="mono" style={{ fontWeight: "700" }}>{p.plateText}</span>
                        <span style={{ color: "var(--text-muted)" }}>{p.confidence}%</span>
                        <span style={{ color: p.isValid ? "#22c55e" : "#94a3b8" }}>{p.state}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </>
          ) : (
            <div style={{ textAlign: "center", padding: "80px 10px", color: "var(--text-muted)" }}>
              No violations available. Choose an incident from the Violation Center to populate this workspace.
            </div>
          )}
        </section>

        {/* Right Column: Workflow Action & Metadata Inspector */}
        <section className="inspector-pane">
          {selectedViolation ? (
            <>
              <div className="inspector-section" style={{ backgroundColor: "var(--bg-tertiary)" }}>
                <span className="form-label" style={{ color: "var(--text-accent)" }}>Workflow Status</span>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginTop: "4px" }}>
                  <span className={`badge ${STATUS_BADGE_CLASS[selectedViolation.status]}`}>
                    {STATUS_LABELS[selectedViolation.status]}
                  </span>
                  <span className="mono" style={{ fontSize: "11px", fontWeight: "600" }}>
                    Confidence: {selectedViolation.confidenceScore}%
                  </span>
                </div>
              </div>

              <div className="inspector-section">
                <h4 style={{ fontWeight: "700", fontSize: "11px", marginBottom: "8px", textTransform: "uppercase" }}>Incident Details</h4>
                <div style={{ display: "flex", flexDirection: "column", gap: "6px", fontSize: "11px" }}>
                  <div style={{ display: "flex", justifyContent: "space-between" }}>
                    <span style={{ color: "var(--text-muted)" }}>Incident Type:</span>
                    <span style={{ fontWeight: "600", color: "var(--text-accent)" }}>{selectedViolation.type}</span>
                  </div>
                  <div style={{ display: "flex", justifyContent: "space-between" }}>
                    <span style={{ color: "var(--text-muted)" }}>Severity:</span>
                    <span style={{ fontWeight: "700", textTransform: "uppercase", fontSize: "10px", color: SEVERITY_COLOR[selectedViolation.severity] }}>
                      {selectedViolation.severity}
                    </span>
                  </div>
                  <div style={{ display: "flex", justifyContent: "space-between" }}>
                    <span style={{ color: "var(--text-muted)" }}>Fine Amount:</span>
                    <span className="mono" style={{ fontWeight: "600" }}>₹{selectedViolation.fineAmountInr}</span>
                  </div>
                  <div style={{ display: "flex", justifyContent: "space-between" }}>
                    <span style={{ color: "var(--text-muted)" }}>Camera / Source:</span>
                    <span className="mono" style={{ fontWeight: "600" }}>{selectedViolation.cameraId}</span>
                  </div>
                  <div style={{ display: "flex", justifyContent: "space-between" }}>
                    <span style={{ color: "var(--text-muted)" }}>Location:</span>
                    <span style={{ fontWeight: "600" }}>{selectedViolation.location}</span>
                  </div>
                  <div style={{ display: "flex", justifyContent: "space-between" }}>
                    <span style={{ color: "var(--text-muted)" }}>Timestamp:</span>
                    <span className="mono" style={{ fontWeight: "500" }}>
                      {new Date(selectedViolation.timestamp).toLocaleString()}
                    </span>
                  </div>
                  <div style={{ display: "flex", justifyContent: "space-between" }}>
                    <span style={{ color: "var(--text-muted)" }}>Vehicle Class:</span>
                    <span style={{ fontWeight: "600" }}>{selectedViolation.vehicleType}</span>
                  </div>
                </div>
              </div>

              {/* Per-violation review — one image, several findings, each
                  resolved independently on this same page */}
              {selectedViolation.violationItems && selectedViolation.violationItems.length > 1 && (
                <div className="inspector-section">
                  <h4 style={{ fontWeight: "700", fontSize: "11px", marginBottom: "8px", textTransform: "uppercase" }}>
                    Violations In This Image ({selectedViolation.violationItems.length})
                  </h4>
                  <div style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
                    {selectedViolation.violationItems.map((item, i) => (
                      <div key={i} style={{
                        display: "flex", justifyContent: "space-between", alignItems: "center",
                        padding: "6px 8px", border: "1px solid var(--border-color)", borderRadius: "4px",
                        backgroundColor: item.reviewStatus === "pending" ? "#FFFFFF" : "#FAFAFA",
                      }}>
                        <div>
                          <div style={{ fontSize: "11px", fontWeight: "600", color: SEVERITY_COLOR[item.severity] || "var(--text-primary)" }}>
                            {item.type} {item.plateText && item.plateText !== "UNCLEAR" && `[${item.plateText}]`}
                          </div>
                          <div style={{ fontSize: "9px", color: "var(--text-muted)" }}>
                            {item.confidence}% confidence · ₹{item.fineAmountInr}
                            {item.tier === 1 && item.reviewStatus === "auto_confirmed" && " · auto-confirmed (clear)"}
                          </div>
                        </div>
                        {item.reviewStatus === "pending" ? (
                          <div style={{ display: "flex", gap: "4px" }}>
                            <button className="btn btn-success btn-sm" title="Approve this finding" onClick={() => handleItemDecision(i, "Approved")}>✓</button>
                            <button className="btn btn-danger btn-sm" title="Reject this finding" onClick={() => handleItemDecision(i, "Rejected")}>✕</button>
                          </div>
                        ) : (
                          <span className={`badge ${
                            item.reviewStatus === "confirmed" || item.reviewStatus === "auto_confirmed" ? "approved" : "rejected"
                          }`}>
                            {item.reviewStatus === "auto_confirmed" ? "AUTO-CONFIRMED" : item.reviewStatus.toUpperCase()}
                          </span>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* ANPR / OCR correction block */}
              <div className="inspector-section">
                <h4 style={{ fontWeight: "700", fontSize: "11px", marginBottom: "6px", textTransform: "uppercase" }}>ANPR Plate Override</h4>

                <div style={{
                  backgroundColor: "#E2E8F0",
                  border: "2px solid #94a3b8",
                  borderRadius: "4px",
                  padding: "10px",
                  textAlign: "center",
                  margin: "8px 0",
                  fontFamily: "var(--font-mono)",
                  fontWeight: "bold",
                  fontSize: "18px",
                  color: "#1E293B",
                  letterSpacing: "2px",
                }}>
                  {selectedViolation.plateNumber}
                  <div style={{ fontSize: "10px", fontWeight: "500", marginTop: "4px", color: "#475569" }}>
                    OCR confidence: {selectedViolation.plateConfidence}%
                  </div>
                </div>

                <div className="form-group" style={{ marginBottom: "0" }}>
                  <label className="form-label">Manually Verified License Plate</label>
                  <input
                    type="text"
                    className="form-input mono"
                    style={{ textTransform: "uppercase" }}
                    value={correctedPlate}
                    onChange={(e) => setCorrectedPlate(e.target.value.toUpperCase())}
                    disabled={selectedViolation.status !== "pending"}
                  />
                  <span style={{ fontSize: "9px", color: "var(--text-muted)", marginTop: "2px", display: "block" }}>
                    Corrected text is saved to the violation record when you approve or reject.
                  </span>
                </div>
              </div>

              {/* Real driver-state alerts (drowsiness / yawn / phone use) */}
              {selectedViolation.driverAlerts && selectedViolation.driverAlerts.length > 0 && (
                <div className="inspector-section">
                  <h4 style={{ fontWeight: "700", fontSize: "11px", marginBottom: "8px", textTransform: "uppercase" }}>Driver State Alerts</h4>
                  <div style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
                    {selectedViolation.driverAlerts.map((a, i) => (
                      <div key={i} style={{ fontSize: "11px", display: "flex", justifyContent: "space-between" }}>
                        <span>{a.alertType.replace(/_/g, " ")}</span>
                        <span className="mono" style={{ color: SEVERITY_COLOR[a.severity] || "var(--text-muted)" }}>
                          {Math.round(a.confidence * 100)}%
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Real ML pipeline diagnostics */}
              {selectedViolation.processing && (
                <div className="inspector-section">
                  <h4 style={{ fontWeight: "700", fontSize: "11px", marginBottom: "8px", textTransform: "uppercase" }}>Pipeline Diagnostics</h4>
                  <div style={{ display: "flex", flexDirection: "column", gap: "6px", fontSize: "11px" }}>
                    <div style={{ display: "flex", justifyContent: "space-between" }}>
                      <span style={{ color: "var(--text-muted)" }}>Detector Model:</span>
                      <span className="mono">{selectedViolation.processing.model}</span>
                    </div>
                    <div style={{ display: "flex", justifyContent: "space-between" }}>
                      <span style={{ color: "var(--text-muted)" }}>OCR Engine:</span>
                      <span className="mono">{selectedViolation.processing.ocrEngine}</span>
                    </div>
                    <div style={{ display: "flex", justifyContent: "space-between" }}>
                      <span style={{ color: "var(--text-muted)" }}>Inference Time:</span>
                      <span className="mono">{selectedViolation.processing.inferenceTimeMs}ms</span>
                    </div>
                    <div style={{ display: "flex", justifyContent: "space-between" }}>
                      <span style={{ color: "var(--text-muted)" }}>Vehicles / Persons:</span>
                      <span className="mono">{selectedViolation.processing.vehiclesDetected} / {selectedViolation.processing.personsDetected}</span>
                    </div>
                  </div>
                </div>
              )}
              {detailLoading && (
                <div style={{ padding: "6px 12px", fontSize: "10px", color: "var(--text-muted)" }}>Loading full evidence record…</div>
              )}

              {/* Real review/audit history for this violation */}
              {reviewHistory.length > 0 && (
                <div className="inspector-section">
                  <h4 style={{ fontWeight: "700", fontSize: "11px", marginBottom: "8px", textTransform: "uppercase" }}>Review History</h4>
                  <div style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
                    {reviewHistory.map((r) => (
                      <div key={r.id} style={{ fontSize: "10px", borderLeft: "2px solid var(--border-accent-dark)", paddingLeft: "6px" }}>
                        <div style={{ fontWeight: "700" }}>{r.action.replace("CITATION_", "")} — {r.actor}</div>
                        <div style={{ color: "var(--text-muted)" }}>{new Date(r.timestamp).toLocaleString()}</div>
                        {r.details && <div style={{ fontStyle: "italic", color: "var(--text-secondary)" }}>{r.details}</div>}
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Decision Action Deck */}
              <div className="inspector-section" style={{ marginTop: "auto", borderTop: "2px solid var(--border-accent)" }}>
                <h4 style={{ fontWeight: "700", fontSize: "11px", marginBottom: "8px", textTransform: "uppercase" }}>ENFORCEMENT DECISION</h4>

                {/* Manual SMS trigger button */}
                <button
                  onClick={() => sendChallanSms(selectedViolation.id)}
                  className="btn btn-secondary btn-sm"
                  style={{ width: "100%", marginBottom: "8px", fontWeight: "700", color: "var(--text-accent)", borderColor: "var(--border-accent-dark)" }}
                >
                  ✉ SEND CHALLAN SMS (MANUAL)
                </button>

                {(selectedViolation.violationItems?.length || 0) > 1 ? (
                  // Multiple findings in this image are resolved individually
                  // above — escalation is still a whole-case action.
                  selectedViolation.status === "pending" ? (
                    <div style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
                      <div style={{ fontSize: "10px", color: "var(--text-muted)" }}>
                        Approve or reject each finding above — escalate the whole case if it needs supervisor attention.
                      </div>
                      <button
                        onClick={handleEscalate}
                        className="btn btn-secondary"
                        style={{ color: "var(--purple)", borderColor: "#d8b4fe" }}
                      >
                        ⚡ ESCALATE WHOLE CASE
                      </button>
                    </div>
                  ) : (
                    <div style={{
                      padding: "8px", borderRadius: "4px", fontSize: "11px",
                      backgroundColor: "#f8fafc", border: "1px solid var(--border-color)"
                    }}>
                      <div style={{ fontWeight: "700" }}>{STATUS_LABELS[selectedViolation.status].toUpperCase()}</div>
                      {selectedViolation.officerId && <div>Officer: <strong>{selectedViolation.officerId}</strong></div>}
                    </div>
                  )
                ) : selectedViolation.status === "pending" ? (
                  <div style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
                    <button onClick={handleApprove} className="btn btn-success" style={{ width: "100%", fontWeight: "700" }}>
                      ✓ CONFIRM & APPROVE CITATION
                    </button>

                    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "6px" }}>
                      <button onClick={() => setShowRejectModal(true)} className="btn btn-danger">
                        ✕ REJECT (VOID)
                      </button>
                      <button
                        onClick={handleEscalate}
                        className="btn btn-secondary"
                        style={{ color: "var(--purple)", borderColor: "#d8b4fe" }}
                      >
                        ⚡ ESCALATE
                      </button>
                    </div>
                    {selectedViolation.officerId && (
                      <div style={{ fontSize: "10px", color: "var(--text-muted)", marginTop: "2px" }}>
                        Claimed by <strong>{selectedViolation.officerId}</strong> (escalated — still awaiting final decision)
                      </div>
                    )}
                  </div>
                ) : (
                  <div style={{
                    padding: "8px",
                    borderRadius: "4px",
                    fontSize: "11px",
                    backgroundColor: "#f8fafc",
                    border: "1px solid var(--border-color)"
                  }}>
                    <div style={{ fontWeight: "700" }}>{STATUS_LABELS[selectedViolation.status].toUpperCase()}</div>
                    <div style={{ marginTop: "4px", display: "flex", flexDirection: "column", gap: "2px" }}>
                      {selectedViolation.officerId && (
                        <div>Officer: <strong>{selectedViolation.officerId}</strong></div>
                      )}
                    </div>
                  </div>
                )}
              </div>
            </>
          ) : null}
        </section>

      </div>

      {/* Reject Modal dialog box */}
      {showRejectModal && (
        <div style={{
          position: "fixed",
          top: 0, left: 0, right: 0, bottom: 0,
          backgroundColor: "rgba(0, 0, 0, 0.4)",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          zIndex: 1000
        }}>
          <div className="card" style={{ width: "350px", transform: "none" }}>
            <div className="card-title">
              <span>REJECT CITATION CASE</span>
              <button onClick={() => setShowRejectModal(false)} style={{ background: "none", border: "none", cursor: "pointer" }}>
                <CloseIcon size={16} />
              </button>
            </div>

            <form onSubmit={handleRejectSubmit}>
              <div className="form-group">
                <label className="form-label">Void Reason / Classification</label>
                <select
                  className="form-input"
                  value={rejectionReason}
                  onChange={(e) => setRejectionReason(e.target.value)}
                  required
                >
                  <option value="">-- Choose Void Reason --</option>
                  <option value="Model false positive: plate obscured or unreadable">Plate Obscured / Unreadable</option>
                  <option value="Emergency vehicle: authorized exemption">Emergency Vehicle Exemption</option>
                  <option value="No infraction: vehicle compliant on closer review">No Infraction On Review</option>
                  <option value="Image quality: motion blur or low light">Image Quality Issue</option>
                  <option value="Other classification mismatch">Other Classification Mismatch</option>
                </select>
              </div>

              <div style={{ display: "flex", gap: "8px", marginTop: "12px", justifyContent: "flex-end" }}>
                <button type="button" className="btn btn-secondary" onClick={() => setShowRejectModal(false)}>
                  CANCEL
                </button>
                <button type="submit" className="btn btn-danger">
                  VOID CITATION RECORD
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

    </div>
  );
}
