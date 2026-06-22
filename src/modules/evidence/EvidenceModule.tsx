"use client";

import React, { useEffect, useState, useCallback } from "react";
import { useRouter } from "next/navigation";
import { usePlatform } from "@/context/PlatformContext";
import { UploadIcon, RefreshIcon, PlayIcon, CloseIcon } from "@/components/Icons";
import {
  uploadSingle,
  uploadBatch,
  fetchTestGalleryList,
  testGalleryImageUrl,
  deleteJob,
  clearAllJobs,
} from "@/lib/evidence";

type UploadMode = "Image" | "Batch" | "Video";

const STATUS_BADGE_CLASS: Record<string, string> = {
  Queued: "review",
  Processing: "review",
  Completed: "approved",
  Failed: "rejected",
};

export default function EvidenceModule() {
  const { jobs, cameras, token } = usePlatform();
  const router = useRouter();

  const [mode, setMode] = useState<UploadMode>("Image");
  const [jobName, setJobName] = useState("");
  const [cameraId, setCameraId] = useState<string>("");
  const [singleFile, setSingleFile] = useState<File | null>(null);
  const [batchFiles, setBatchFiles] = useState<File[]>([]);
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [galleryFiles, setGalleryFiles] = useState<string[]>([]);
  const [galleryLoading, setGalleryLoading] = useState(false);
  const [hiddenJobIds, setHiddenJobIds] = useState<Set<string>>(new Set());
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const [clearingAll, setClearingAll] = useState(false);

  useEffect(() => {
    fetchTestGalleryList().then(setGalleryFiles);
  }, []);

  const resetForm = useCallback(() => {
    setJobName("");
    setSingleFile(null);
    setBatchFiles([]);
    setVideoFile(null);
    setError(null);
  }, []);

  const handleModeChange = (m: UploadMode) => {
    setMode(m);
    resetForm();
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);

    if (mode === "Image" && !singleFile) return setError("Choose an image file.");
    if (mode === "Video" && !videoFile) return setError("Choose a video file.");
    if (mode === "Batch" && batchFiles.length === 0) return setError("Choose at least one image.");

    const name = jobName.trim() || (mode === "Image" ? singleFile!.name : mode === "Video" ? videoFile!.name : `Batch (${batchFiles.length} images)`);

    setSubmitting(true);
    try {
      if (mode === "Batch") {
        await uploadBatch({ name, files: batchFiles, cameraId: cameraId || null, token });
      } else {
        await uploadSingle({
          name,
          sourceType: mode,
          file: mode === "Image" ? singleFile! : videoFile!,
          cameraId: cameraId || null,
          token,
        });
      }
      resetForm();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Upload failed.");
    } finally {
      setSubmitting(false);
    }
  };

  const handleGalleryPick = async (filename: string) => {
    setGalleryLoading(true);
    setError(null);
    try {
      const url = testGalleryImageUrl(filename);
      const res = await fetch(url);
      const blob = await res.blob();
      const file = new File([blob], filename, { type: blob.type || "image/jpeg" });
      await uploadSingle({ name: filename, sourceType: "Image", file, cameraId: cameraId || null, token });
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load test image.");
    } finally {
      setGalleryLoading(false);
    }
  };

  const handleDeleteJob = async (jobId: string) => {
    if (!window.confirm(`Delete job ${jobId}? This removes its violations and evidence images permanently.`)) return;
    setDeletingId(jobId);
    try {
      await deleteJob(jobId, token);
      setHiddenJobIds((prev) => new Set(prev).add(jobId));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Delete failed.");
    } finally {
      setDeletingId(null);
    }
  };

  const handleClearAll = async () => {
    if (!window.confirm("Clear ALL jobs, violations, and evidence images? This cannot be undone.")) return;
    setClearingAll(true);
    try {
      await clearAllJobs(token);
      setHiddenJobIds(new Set(jobs.map((j) => j.id)));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Clear all failed.");
    } finally {
      setClearingAll(false);
    }
  };

  const sortedJobs = [...jobs]
    .filter((j) => !hiddenJobIds.has(j.id))
    .sort((a, b) => new Date(b.uploadTime).getTime() - new Date(a.uploadTime).getTime());

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "12px" }}>

      <div>
        <h1 style={{ fontSize: "20px", fontWeight: "700", letterSpacing: "-0.5px" }}>EVIDENCE PIPELINE</h1>
        <p style={{ fontSize: "11px", color: "var(--text-muted)", textTransform: "uppercase", marginTop: "2px" }}>
          Run media through the real detection pipeline and inspect a full, clubbed step-by-step breakdown
        </p>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "380px 1fr", gap: "12px", alignItems: "start" }}>

        {/* Upload card */}
        <div className="card" style={{ transform: "none" }}>
          <div className="card-title">
            <span>SUBMIT FOR PROCESSING</span>
          </div>

          <div className="filter-bar" style={{ marginBottom: "10px" }}>
            {(["Image", "Batch", "Video"] as UploadMode[]).map((m) => (
              <div
                key={m}
                className={`filter-item ${mode === m ? "active" : ""}`}
                onClick={() => handleModeChange(m)}
                style={{ cursor: "pointer" }}
              >
                {m === "Image" ? "SINGLE IMAGE" : m === "Batch" ? "BATCH IMAGES" : "VIDEO"}
              </div>
            ))}
          </div>

          <form onSubmit={handleSubmit}>
            <div className="form-group">
              <label className="form-label">Job Name (optional)</label>
              <input
                type="text"
                className="form-input"
                value={jobName}
                onChange={(e) => setJobName(e.target.value)}
                placeholder="e.g. Junction-12 Morning Sweep"
              />
            </div>

            <div className="form-group">
              <label className="form-label">Calibrated Camera (optional)</label>
              <select className="form-input" value={cameraId} onChange={(e) => setCameraId(e.target.value)}>
                <option value="">-- None / Uncalibrated --</option>
                {cameras.map((c) => (
                  <option key={c.id} value={c.id}>{c.id} — {c.location}</option>
                ))}
              </select>
            </div>

            {mode === "Image" && (
              <div className="form-group">
                <label className="form-label">Image File</label>
                <input
                  type="file"
                  accept="image/*"
                  className="form-input"
                  onChange={(e) => setSingleFile(e.target.files?.[0] || null)}
                />
              </div>
            )}

            {mode === "Batch" && (
              <div className="form-group" style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
                <div>
                  <label className="form-label">Select Multiple Images</label>
                  <input
                    type="file"
                    accept="image/*"
                    multiple
                    className="form-input"
                    onChange={(e) => setBatchFiles(Array.from(e.target.files || []))}
                  />
                </div>
                <div style={{ display: "flex", alignItems: "center", justifyContent: "center", margin: "4px 0", color: "var(--text-muted)", fontSize: "10px" }}>
                  — OR —
                </div>
                <div>
                  <label className="form-label">Upload a Whole Folder</label>
                  <input
                    ref={(el) => {
                      if (el) {
                        el.setAttribute("webkitdirectory", "");
                        el.setAttribute("directory", "");
                      }
                    }}
                    type="file"
                    multiple
                    className="form-input"
                    onChange={(e) => {
                      const files = Array.from(e.target.files || []);
                      const imageFiles = files.filter(f => f.type.startsWith("image/") || /\.(jpg|jpeg|png|webp)$/i.test(f.name));
                      setBatchFiles(imageFiles);
                    }}
                  />
                </div>
                {batchFiles.length > 0 && (
                  <span style={{ fontSize: "10px", color: "var(--text-muted)", marginTop: "2px", display: "block" }}>
                    {batchFiles.length} image file(s) selected — clubbed into one result
                  </span>
                )}
              </div>
            )}

            {mode === "Video" && (
              <div className="form-group">
                <label className="form-label">Video File</label>
                <input
                  type="file"
                  accept="video/*,.mp4,.mkv,.avi,.mov,.webm"
                  className="form-input"
                  onChange={(e) => setVideoFile(e.target.files?.[0] || null)}
                />
              </div>
            )}

            {error && (
              <div style={{ fontSize: "11px", color: "var(--danger)", marginBottom: "8px" }}>{error}</div>
            )}

            <button type="submit" className="btn btn-primary" disabled={submitting} style={{ width: "100%" }}>
              <UploadIcon size={14} /> {submitting ? "SUBMITTING…" : "RUN THROUGH PIPELINE"}
            </button>
          </form>

          {galleryFiles.length > 0 && (
            <div style={{ marginTop: "14px", borderTop: "1px solid var(--border-color)", paddingTop: "10px" }}>
              <div style={{ fontSize: "10px", fontWeight: "700", color: "var(--text-muted)", marginBottom: "6px", textTransform: "uppercase" }}>
                Quick Pick — Test Gallery
              </div>
              <div style={{ display: "flex", gap: "6px", flexWrap: "wrap" }}>
                {galleryFiles.slice(0, 12).map((f) => (
                  <button
                    key={f}
                    type="button"
                    disabled={galleryLoading}
                    onClick={() => handleGalleryPick(f)}
                    className="btn btn-secondary btn-sm"
                    title={f}
                    style={{ padding: "2px 0", width: "44px", height: "44px", overflow: "hidden" }}
                  >
                    <img src={testGalleryImageUrl(f)} alt={f} style={{ width: "100%", height: "100%", objectFit: "cover" }} />
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Recent jobs */}
        <div className="card" style={{ transform: "none" }}>
          <div className="card-title">
            <span>RECENT JOBS</span>
            <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
              <button
                className="btn btn-danger btn-sm"
                onClick={handleClearAll}
                disabled={clearingAll || sortedJobs.length === 0}
              >
                {clearingAll ? "CLEARING…" : "CLEAR ALL"}
              </button>
              <RefreshIcon size={14} />
            </div>
          </div>
          <div className="table-container">
            <table className="dense-table">
              <thead>
                <tr>
                  <th>JOB</th>
                  <th>TYPE</th>
                  <th>UPLOADED</th>
                  <th>PROGRESS</th>
                  <th>STATUS</th>
                  <th>VIOLATIONS</th>
                  <th></th>
                </tr>
              </thead>
              <tbody>
                {sortedJobs.length === 0 && (
                  <tr><td colSpan={7} style={{ textAlign: "center", color: "var(--text-muted)", padding: "20px" }}>No jobs submitted yet.</td></tr>
                )}
                {sortedJobs.map((j) => (
                  <tr
                    key={j.id}
                    onClick={() => j.status === "Completed" && router.push(`/evidence/${j.id}`)}
                    style={{ cursor: j.status === "Completed" ? "pointer" : "default" }}
                  >
                    <td>
                      <div className="mono" style={{ fontWeight: "700" }}>{j.id}</div>
                      <div style={{ fontSize: "10px", color: "var(--text-muted)" }}>{j.name}</div>
                    </td>
                    <td>{j.sourceType}</td>
                    <td className="mono" style={{ fontSize: "10px" }}>{new Date(j.uploadTime).toLocaleString()}</td>
                    <td>
                      <div className="progress-bar-outer">
                        <div
                          className={`progress-bar-inner ${j.status === "Completed" ? "completed" : j.status === "Failed" ? "failed" : ""}`}
                          style={{ width: `${j.progress}%` }}
                        />
                      </div>
                    </td>
                    <td><span className={`badge ${STATUS_BADGE_CLASS[j.status] || ""}`}>{j.status}</span></td>
                    <td className="mono">{j.violationsFound}</td>
                    <td>
                      <div style={{ display: "flex", gap: "4px" }}>
                        {j.status === "Completed" && (
                          <button
                            className="btn btn-secondary btn-sm"
                            onClick={(e) => { e.stopPropagation(); router.push(`/evidence/${j.id}`); }}
                          >
                            <PlayIcon size={11} /> VIEW
                          </button>
                        )}
                        <button
                          className="btn btn-danger btn-sm"
                          disabled={deletingId === j.id}
                          onClick={(e) => { e.stopPropagation(); handleDeleteJob(j.id); }}
                          title="Delete this job, its violations, and its evidence images"
                        >
                          <CloseIcon size={11} />
                        </button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
}
