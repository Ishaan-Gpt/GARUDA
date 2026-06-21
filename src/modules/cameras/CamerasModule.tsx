"use client";

import React, { useState } from "react";
import { usePlatform, Camera, CameraStatus } from "@/context/PlatformContext";
import { CameraIcon, PlusIcon, CloseIcon, CheckIcon, RefreshIcon } from "@/components/Icons";

export default function CamerasModule() {
  const { cameras, role, addCamera, toggleCameraStatus, deleteCamera, testCameraConnection, isBackendConnected } = usePlatform();

  // Component states
  const [selectedCameraId, setSelectedCameraId] = useState<string>("CAM-101");
  const [showAddForm, setShowAddForm] = useState(false);
  const [testingCamId, setTestingCamId] = useState<string | null>(null);
  const [testResult, setTestResult] = useState<{ id: string; success: boolean } | null>(null);

  // Form states for new camera
  const [newId, setNewId] = useState("");
  const [newName, setNewName] = useState("");
  const [newRtsp, setNewRtsp] = useState("");
  const [newLoc, setNewLoc] = useState("");
  const [newRes, setNewRes] = useState("1920x1080");

  const selectedCamera = cameras.find(c => c.id === selectedCameraId) || cameras[0];

  const handleTestConnection = async (id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    setTestingCamId(id);
    setTestResult(null);
    const success = await testCameraConnection(id);
    setTestingCamId(null);
    setTestResult({ id, success });
    setTimeout(() => setTestResult(null), 4000); // clear result message after 4s
  };

  const handleAddCamera = (e: React.FormEvent) => {
    e.preventDefault();
    if (!newId || !newName || !newRtsp || !newLoc) {
      alert("Please fill in all registry fields.");
      return;
    }
    addCamera({
      id: newId,
      name: newName,
      rtspUrl: newRtsp,
      location: newLoc,
      status: "Active",
      resolution: newRes
    });
    // Reset Form
    setNewId("");
    setNewName("");
    setNewRtsp("");
    setNewLoc("");
    setShowAddForm(false);
  };

  const handleToggleStatus = (id: string, currentStatus: CameraStatus, e: React.MouseEvent) => {
    e.stopPropagation();
    const nextStatus: CameraStatus = currentStatus === "Active" ? "Disabled" : "Active";
    toggleCameraStatus(id, nextStatus);
  };

  const handleDeleteCamera = (id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    if (confirm(`Are you sure you want to remove camera ${id} from the GARUDA Registry?`)) {
      deleteCamera(id);
      if (selectedCameraId === id) {
        setSelectedCameraId(cameras.find(c => c.id !== id)?.id || "");
      }
    }
  };

  // Check roles permissions
  const canModifyRegistry = role === "Admin" || role === "Supervisor";

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "12px" }}>
      <div>
        <h1 style={{ fontSize: "20px", fontWeight: "700", letterSpacing: "-0.5px" }}>CAMERA REGISTRY & STREAM CONTROLS</h1>
        <p style={{ fontSize: "11px", color: "var(--text-muted)", textTransform: "uppercase", marginTop: "2px" }}>
          Manage RTSP connection endpoints, system heartbeats, and sensor diagnostic values
        </p>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 360px", gap: "12px" }}>
        
        {/* Left Registry Table Card */}
        <div className="card">
          <div className="card-title">
            <span>REGISTERED SENSORS ({cameras.length})</span>
            {canModifyRegistry && (
              <button 
                className="btn btn-primary btn-sm"
                onClick={() => setShowAddForm(true)}
              >
                <PlusIcon size={12} /> ADD NEW ENDPOINT
              </button>
            )}
          </div>

          <div className="table-container">
            <table className="dense-table">
              <thead>
                <tr>
                  <th>ID</th>
                  <th>Location Name / RTSP Stream</th>
                  <th>Status</th>
                  <th>Resolution</th>
                  <th>Last Heartbeat</th>
                  <th style={{ textAlign: "right" }}>Actions</th>
                </tr>
              </thead>
              <tbody>
                {cameras.map(c => {
                  const isSelected = c.id === selectedCameraId;
                  return (
                    <tr 
                      key={c.id} 
                      onClick={() => setSelectedCameraId(c.id)}
                      style={{ 
                        cursor: "pointer",
                        backgroundColor: isSelected ? "var(--bg-tertiary)" : "transparent",
                        fontWeight: isSelected ? "600" : "normal"
                      }}
                    >
                      <td className="mono" style={{ color: "var(--text-accent)" }}>{c.id}</td>
                      <td>
                        <div style={{ fontSize: "12px" }}>{c.name}</div>
                        <div className="mono" style={{ fontSize: "9px", color: "var(--text-muted)", wordBreak: "break-all" }}>{c.rtspUrl}</div>
                      </td>
                      <td>
                        <span className={`badge ${c.status.toLowerCase()}`}>
                          {c.status}
                        </span>
                      </td>
                      <td className="mono">{c.resolution || "Unknown"}</td>
                      <td className="mono" style={{ fontSize: "10px" }}>
                        {c.status === "Active" 
                          ? new Date(c.lastHeartbeat).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }) 
                          : "---"
                        }
                      </td>
                      <td style={{ textAlign: "right" }} onClick={(e) => e.stopPropagation()}>
                        <div style={{ display: "flex", gap: "4px", justifyContent: "flex-end" }}>
                          <button 
                            className="btn btn-secondary btn-sm" 
                            title="Test Endpoint Connection"
                            onClick={(e) => handleTestConnection(c.id, e)}
                            disabled={testingCamId === c.id || c.status === "Disabled"}
                          >
                            {testingCamId === c.id ? (
                              <span style={{ fontSize: "10px", display: "inline-block", animation: "spin 1s linear infinite" }}>⚙</span>
                            ) : (
                              "TEST"
                            )}
                          </button>

                          <button 
                            className={`btn btn-sm ${c.status === "Active" ? "btn-danger" : "btn-success"}`}
                            onClick={(e) => handleToggleStatus(c.id, c.status, e)}
                          >
                            {c.status === "Active" ? "DISABLE" : "ENABLE"}
                          </button>

                          {canModifyRegistry && (
                            <button 
                              className="btn btn-secondary btn-sm"
                              style={{ color: "var(--danger)", borderColor: "#fca5a5" }}
                              onClick={(e) => handleDeleteCamera(c.id, e)}
                            >
                              DELETE
                            </button>
                          )}
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          {testResult && (
            <div style={{ 
              marginTop: "8px", 
              padding: "6px 12px", 
              borderRadius: "4px", 
              fontSize: "11px",
              backgroundColor: testResult.success ? "var(--success-bg)" : "var(--danger-bg)",
              color: testResult.success ? "var(--success)" : "var(--danger)",
              border: `1px solid ${testResult.success ? "#bbf7d0" : "#fecaca"}`,
              fontWeight: "600"
            }}>
              {testResult.success 
                ? `Connection to ${testResult.id} validated successfully. RTSP session established.` 
                : `Connection to ${testResult.id} failed. Connection timed out. Socket unreachable.`
              }
            </div>
          )}
        </div>

        {/* Right Preview and Config Pane */}
        <div style={{ display: "flex", flexDirection: "column", gap: "12px" }}>
          {selectedCamera ? (
            <div className="card">
              <div className="card-title">
                <span>FEED INSPECTOR: {selectedCamera.id} {isBackendConnected ? "(CCTV TELEMETRY ACTIVE)" : "(SIMULATED)"}</span>
                <span style={{ fontFamily: "var(--font-mono)", fontSize: "10px" }}>{selectedCamera.resolution}</span>
              </div>

              {/* Feed Stream Box */}
              <div style={{ 
                height: "220px", 
                backgroundColor: "#000", 
                borderRadius: "4px", 
                overflow: "hidden", 
                position: "relative",
                border: "1px solid var(--border-color)"
              }}>
                {selectedCamera.status === "Active" ? (
                  <ExpandedAnimatedCCTV cameraId={selectedCamera.id} />
                ) : (
                  <div style={{ height: "100%", width: "100%", display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", color: "#64748b" }}>
                    <CloseIcon size={40} style={{ color: "var(--danger)" }} />
                    <span style={{ fontSize: "12px", marginTop: "8px", fontWeight: "bold" }}>STREAM TERMINATED</span>
                    <span style={{ fontSize: "10px", color: "var(--text-muted)" }}>Camera status is {selectedCamera.status.toUpperCase()}</span>
                  </div>
                )}
              </div>

              <div style={{ marginTop: "12px", display: "flex", flexDirection: "column", gap: "6px" }}>
                <div style={{ display: "flex", justifyContent: "space-between", fontSize: "11px" }}>
                  <span style={{ color: "var(--text-muted)" }}>Endpoint Target:</span>
                  <span className="mono" style={{ fontWeight: "600" }}>{selectedCamera.rtspUrl}</span>
                </div>
                <div style={{ display: "flex", justifyContent: "space-between", fontSize: "11px" }}>
                  <span style={{ color: "var(--text-muted)" }}>Sensor Location:</span>
                  <span style={{ fontWeight: "600" }}>{selectedCamera.location}</span>
                </div>
                <div style={{ display: "flex", justifyContent: "space-between", fontSize: "11px" }}>
                  <span style={{ color: "var(--text-muted)" }}>Diagnostics:</span>
                  <span style={{ color: selectedCamera.status === "Active" ? "var(--success)" : "var(--text-muted)", fontWeight: "bold" }}>
                    {selectedCamera.status === "Active" ? "SYSTEMS NOMINAL" : "UNREACHABLE"}
                  </span>
                </div>
              </div>

              {/* simulated session debug logs */}
              <div style={{ 
                marginTop: "10px", 
                background: "#0F172A", 
                borderRadius: "4px", 
                padding: "8px", 
                fontSize: "9px", 
                fontFamily: "var(--font-mono)", 
                color: "#A7F3D0",
                maxHeight: "100px",
                overflowY: "auto"
              }}>
                <div>[RTCP] Session heartbeat initialized</div>
                <div>[RTSP] GET DESCRIBE sdp content matched ok</div>
                <div>[INFERENCE] Load tensor config v3.4... SUCCESS</div>
                {selectedCamera.status === "Active" ? (
                  <>
                    <div>[DECODER] H.264 Annex B stream frames decoded.</div>
                    <div>[DETECTION] YOLOV8 inference clock ticking.</div>
                    <div>[SOCKET] Stream payload bound to internal loop.</div>
                  </>
                ) : (
                  <div style={{ color: "#FCA5A5" }}>[FATAL] RTSP socket closed. Code 404 (Not Found)</div>
                )}
              </div>
            </div>
          ) : (
            <div className="card" style={{ padding: "40px 10px", textAlign: "center", color: "var(--text-muted)" }}>
              Select a registered sensor from the table list to run direct telemetry inspector.
            </div>
          )}

          {/* Add Camera Form Popup Card */}
          {showAddForm && (
            <div style={{ 
              position: "fixed", 
              top: 0, left: 0, right: 0, bottom: 0, 
              backgroundColor: "rgba(0, 0, 0, 0.4)", 
              display: "flex", 
              alignItems: "center", 
              justifyContent: "center",
              zIndex: 1000 
            }}>
              <div className="card" style={{ width: "380px", margin: "16px", transform: "none" }}>
                <div className="card-title">
                  <span>ADD NEW RTSP CAMERA</span>
                  <button 
                    onClick={() => setShowAddForm(false)}
                    style={{ background: "none", border: "none", cursor: "pointer" }}
                  >
                    <CloseIcon size={16} />
                  </button>
                </div>

                <form onSubmit={handleAddCamera} style={{ display: "flex", flexDirection: "column", gap: "10px" }}>
                  <div className="form-group">
                    <label className="form-label">Camera identifier code (Unique)</label>
                    <input 
                      type="text" 
                      className="form-input mono" 
                      placeholder="e.g. CAM-606" 
                      value={newId} 
                      onChange={(e) => setNewId(e.target.value)} 
                      required 
                    />
                  </div>

                  <div className="form-group">
                    <label className="form-label">Display Name / Title</label>
                    <input 
                      type="text" 
                      className="form-input" 
                      placeholder="e.g. Main St North Intersection" 
                      value={newName} 
                      onChange={(e) => setNewName(e.target.value)} 
                      required 
                    />
                  </div>

                  <div className="form-group">
                    <label className="form-label">RTSP Endpoint URL</label>
                    <input 
                      type="text" 
                      className="form-input mono" 
                      placeholder="rtsp://admin:password@192.168.1.100/stream" 
                      value={newRtsp} 
                      onChange={(e) => setNewRtsp(e.target.value)} 
                      required 
                    />
                  </div>

                  <div className="form-group">
                    <label className="form-label">Location Sector</label>
                    <input 
                      type="text" 
                      className="form-input" 
                      placeholder="e.g. Zone D Highway" 
                      value={newLoc} 
                      onChange={(e) => setNewLoc(e.target.value)} 
                      required 
                    />
                  </div>

                  <div className="form-group">
                    <label className="form-label">Sensor Resolution</label>
                    <select className="form-input" value={newRes} onChange={(e) => setNewRes(e.target.value)}>
                      <option value="1920x1080">1920x1080 (1080p FHD)</option>
                      <option value="1280x720">1280x720 (720p HD)</option>
                      <option value="2560x1440">2560x1440 (2K QHD)</option>
                    </select>
                  </div>

                  <div style={{ display: "flex", gap: "8px", marginTop: "6px", justifyContent: "flex-end" }}>
                    <button type="button" className="btn btn-secondary" onClick={() => setShowAddForm(false)}>
                      CANCEL
                    </button>
                    <button type="submit" className="btn btn-primary">
                      REGISTER ENDPOINT
                    </button>
                  </div>
                </form>
              </div>
            </div>
          )}

        </div>

      </div>
    </div>
  );
}

// Expanded SVG camera simulator for detailed preview
function ExpandedAnimatedCCTV({ cameraId }: { cameraId: string }) {
  return (
    <svg width="100%" height="100%" viewBox="0 0 320 220" style={{ display: "block" }}>
      {/* Background road lanes */}
      <rect width="320" height="220" fill="#0f172a" />
      
      {/* Road asphalt lane markers */}
      <rect x="0" y="80" width="320" height="60" fill="#1e293b" />
      <line x1="0" y1="110" x2="320" y2="110" stroke="#64748b" strokeWidth="2" strokeDasharray="8,8" />
      <line x1="0" y1="80" x2="320" y2="80" stroke="#94a3b8" strokeWidth="1" />
      <line x1="0" y1="140" x2="320" y2="140" stroke="#94a3b8" strokeWidth="1" />
      
      {/* Stop Line bar */}
      <line x1="240" y1="80" x2="240" y2="140" stroke="#f8fafc" strokeWidth="3" />
      
      {/* Intersection crosswalk blocks */}
      <rect x="250" y="80" width="8" height="60" fill="repeating-linear-gradient(45deg, #cbd5e1, #cbd5e1 5px, #1e293b 5px, #1e293b 10px)" />
      
      {/* Traffic signal pole & light */}
      <line x1="242" y1="50" x2="242" y2="80" stroke="#cbd5e1" strokeWidth="2" />
      <rect x="238" y="36" width="8" height="15" rx="2" fill="#334155" />
      <circle cx="242" cy="40" r="2" fill="#ef4444" />
      <circle cx="242" cy="44" r="2" fill="#1e293b" />
      <circle cx="242" cy="48" r="2" fill="#1e293b" />

      {/* Moving Simulated vehicle on lanes */}
      <g>
        <animateTransform
          attributeName="transform"
          type="translate"
          from="-60 0"
          to="380 0"
          dur={cameraId === "CAM-101" ? "4s" : (cameraId === "CAM-202" ? "3s" : "5s")}
          repeatCount="indefinite"
        />
        {/* Car object */}
        <rect x="0" y="88" width="36" height="18" rx="4" fill="#3B82F6" stroke="#1D4ED8" strokeWidth="1.5" />
        <rect x="24" y="91" width="8" height="4" fill="#E2E8F0" />
        <rect x="24" y="99" width="8" height="4" fill="#E2E8F0" />
        <circle cx="8" cy="107" r="3" fill="#000" />
        <circle cx="28" cy="107" r="3" fill="#000" />
        
        {/* Detection Bounding Box overlay */}
        <rect x="-2" y="86" width="40" height="24" fill="none" stroke="#EAB308" strokeWidth="1.5" />
        {/* Box metadata tag */}
        <rect x="-2" y="74" width="34" height="12" fill="#EAB308" />
        <text x="0" y="82" fill="#000" fontSize="8" fontFamily="monospace" fontWeight="bold">CAR 94%</text>

        {/* License plate coordinates bounding box */}
        <rect x="28" y="94" width="10" height="6" fill="none" stroke="#10B981" strokeWidth="1" />
      </g>

      {/* Overlay OSD diagnostics */}
      <text x="12" y="20" fill="#4ade80" fontSize="9" fontFamily="monospace">CAM_STREAM_{cameraId}</text>
      <text x="12" y="32" fill="#94a3b8" fontSize="8" fontFamily="monospace">INFERENCE RUNTIME: YOLOV8-COCO</text>
      <text x="12" y="205" fill="#f8fafc" fontSize="8" fontFamily="monospace">FPS: 30.00 / LATENCY: 22ms</text>
      <text x="210" y="205" fill="#f8fafc" fontSize="8" fontFamily="monospace">60.200.41.10</text>
    </svg>
  );
}
