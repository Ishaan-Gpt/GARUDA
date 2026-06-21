"use client";

import React, { useState, useMemo } from "react";
import { usePlatform } from "@/context/PlatformContext";
import { CameraIcon, AlertIcon, CloseIcon, RefreshIcon } from "@/components/Icons";
import { VIOLATION_TYPES, STATUS_LABELS, STATUS_BADGE_CLASS } from "@/lib/violations";

export default function DashboardModule() {
  const { cameras, violations, isBackendConnected } = usePlatform();

  // Filters State
  const [filterCamera, setFilterCamera] = useState("all");
  const [filterLocation, setFilterLocation] = useState("");
  const [filterType, setFilterType] = useState("all");
  const [filterDate, setFilterDate] = useState("all"); // "all", "today", "yesterday"

  // Reset Filters
  const handleResetFilters = () => {
    setFilterCamera("all");
    setFilterLocation("");
    setFilterType("all");
    setFilterDate("all");
  };

  // Unique list of locations for auto-suggestions / dropdown details
  const locations = useMemo(() => {
    const locSet = new Set(cameras.map(c => c.location));
    return Array.from(locSet);
  }, [cameras]);

  // Filter logic
  const filteredViolations = useMemo(() => {
    return violations.filter(vio => {
      // Camera filter
      if (filterCamera !== "all" && vio.cameraId !== filterCamera) return false;
      // Location filter
      if (filterLocation && !vio.location.toLowerCase().includes(filterLocation.toLowerCase())) return false;
      // Violation type filter
      if (filterType !== "all" && vio.type !== filterType) return false;
      // Date filter
      if (filterDate === "today") {
        const todayStr = new Date().toDateString();
        return new Date(vio.timestamp).toDateString() === todayStr;
      }
      if (filterDate === "yesterday") {
        const yesterday = new Date();
        yesterday.setDate(yesterday.getDate() - 1);
        return new Date(vio.timestamp).toDateString() === yesterday.toDateString();
      }
      return true;
    });
  }, [violations, filterCamera, filterLocation, filterType, filterDate]);

  // Derived Metrics based on Filtered Violations
  const stats = useMemo(() => {
    const totalCams = cameras.length;
    const activeStreams = cameras.filter(c => c.status === "Active").length;
    
    const todayStr = new Date().toDateString();
    const todayVios = violations.filter(v => new Date(v.timestamp).toDateString() === todayStr).length;
    
    const pendingReviews = violations.filter(v => v.status === "pending").length;
    const confirmed = violations.filter(v => v.status === "confirmed" || v.status === "auto_challan").length;
    const falsePositives = violations.filter(v => v.status === "rejected").length;

    // Avg confidence calculation
    const confVios = filteredViolations.filter(v => v.confidenceScore > 0);
    const avgConfidence = confVios.length > 0
      ? (confVios.reduce((acc, curr) => acc + curr.confidenceScore, 0) / confVios.length).toFixed(1)
      : "0.0";

    // Throughput: real violations per minute, counted over the trailing 2-minute window
    const recentCount = violations.filter(v => Date.now() - new Date(v.timestamp).getTime() < 120000).length;
    const throughput = (recentCount / 2).toFixed(1); // events per minute

    return {
      totalCams,
      activeStreams,
      totalViolations: violations.length,
      todayViolations: todayVios,
      pendingReviews,
      confirmed,
      falsePositives,
      avgConfidence,
      throughput
    };
  }, [cameras, violations, filteredViolations]);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "12px" }}>
      {/* Page Header Title */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <div>
          <h1 style={{ fontSize: "20px", fontWeight: "700", letterSpacing: "-0.5px" }}>OPERATIONAL CONTROL DASHBOARD</h1>
          <p style={{ fontSize: "11px", color: "var(--text-muted)", textTransform: "uppercase", marginTop: "2px" }}>
            Real-Time Edge Inference & Operator Validation Workspace
          </p>
        </div>
      </div>

      {/* KPI Stats Grid */}
      <section className="dashboard-grid">
        <div className="metric-card">
          <span className="metric-title">Total Cameras</span>
          <span className="metric-value">{stats.totalCams}</span>
          <span className="metric-footer">Registered Feeds</span>
        </div>

        <div className="metric-card">
          <span className="metric-title">Active Streams</span>
          <span className="metric-value" style={{ color: "var(--success)" }}>{stats.activeStreams}</span>
          <span className="metric-footer">
            <span className="pulse-green"></span> Registered Stream Grid
          </span>
        </div>

        <div className="metric-card">
          <span className="metric-title">Total Violations</span>
          <span className="metric-value">{stats.totalViolations}</span>
          <span className="metric-footer">Database Records</span>
        </div>

        <div className="metric-card">
          <span className="metric-title">Today's Violations</span>
          <span className="metric-value" style={{ color: "var(--text-accent)" }}>{stats.todayViolations}</span>
          <span className="metric-footer">Since 12:00 AM</span>
        </div>

        <div className="metric-card">
          <span className="metric-title">Pending Reviews</span>
          <span className="metric-value" style={{ color: "var(--warning)" }}>{stats.pendingReviews}</span>
          <span className="metric-footer">Human Action Required</span>
        </div>

        <div className="metric-card">
          <span className="metric-title">Confirmed Violations</span>
          <span className="metric-value" style={{ color: "var(--success)" }}>{stats.confirmed}</span>
          <span className="metric-footer">Approved for Summons</span>
        </div>

        <div className="metric-card">
          <span className="metric-title">False Positives</span>
          <span className="metric-value" style={{ color: "var(--danger)" }}>{stats.falsePositives}</span>
          <span className="metric-footer">Operator Rejected</span>
        </div>

        <div className="metric-card">
          <span className="metric-title">Throughput</span>
          <span className="metric-value">{stats.throughput} /m</span>
          <span className="metric-footer">Inference Discoveries</span>
        </div>

        <div className="metric-card">
          <span className="metric-title">Avg Confidence</span>
          <span className="metric-value">{stats.avgConfidence}%</span>
          <span className="metric-footer">Edge Model Average</span>
        </div>
      </section>

      {/* Filter and Content Row */}
      <div style={{ display: "grid", gridTemplateColumns: "240px 1fr", gap: "12px" }}>
        
        {/* Left Filter Card */}
        <aside className="card" style={{ height: "fit-content", position: "sticky", top: "0" }}>
          <div className="card-title">
            <span>DASHBOARD FILTERS</span>
            <button 
              onClick={handleResetFilters} 
              style={{ background: "none", border: "none", color: "var(--text-accent)", fontWeight: "600", fontSize: "10px", cursor: "pointer", display: "flex", alignItems: "center", gap: "2px" }}
            >
              <RefreshIcon size={10} /> RESET
            </button>
          </div>

          <div style={{ display: "flex", flexDirection: "column", gap: "10px" }}>
            <div className="form-group">
              <label className="form-label">Date Filter</label>
              <select className="form-input" value={filterDate} onChange={(e) => setFilterDate(e.target.value)}>
                <option value="all">All Dates</option>
                <option value="today">Today's Data</option>
                <option value="yesterday">Yesterday's Data</option>
              </select>
            </div>

            <div className="form-group">
              <label className="form-label">Camera Stream</label>
              <select className="form-input" value={filterCamera} onChange={(e) => setFilterCamera(e.target.value)}>
                <option value="all">All Cameras</option>
                {cameras.map(c => (
                  <option key={c.id} value={c.id}>{c.id} - {c.name}</option>
                ))}
              </select>
            </div>

            <div className="form-group">
              <label className="form-label">Location Search</label>
              <input 
                type="text" 
                className="form-input" 
                placeholder="e.g. Zone A" 
                value={filterLocation}
                onChange={(e) => setFilterLocation(e.target.value)}
              />
            </div>

            <div className="form-group">
              <label className="form-label">Violation Type</label>
              <select className="form-input" value={filterType} onChange={(e) => setFilterType(e.target.value)}>
                <option value="all">All Types</option>
                {VIOLATION_TYPES.map(t => (
                  <option key={t} value={t}>{t}</option>
                ))}
              </select>
            </div>
          </div>
        </aside>

        {/* Right Dashboard Contents */}
        <div style={{ display: "flex", flexDirection: "column", gap: "12px" }}>
          
          {/* Active Live Streams Grid */}
          <div className="card">
            <div className="card-title">
              <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
                <CameraIcon size={14} style={{ color: "var(--text-accent)" }} />
                <span>LIVE FEED OVERVIEWS {isBackendConnected ? "(EDGE INF Nom)" : "(SIMULATED FALLBACK)"}</span>
              </div>
              <span className="brand-badge">{isBackendConnected ? "TELEMETRY ACTIVE" : "STANDALONE MODE"}</span>
            </div>

            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: "10px" }}>
              {cameras.map(c => (
                <div key={c.id} style={{ 
                  border: "1px solid var(--border-color)", 
                  borderRadius: "6px", 
                  overflow: "hidden", 
                  backgroundColor: "#0A0A0A"
                }}>
                  <div style={{ 
                    padding: "4px 8px", 
                    backgroundColor: "#1E293B", 
                    color: "#FFF", 
                    display: "flex", 
                    justifyContent: "space-between", 
                    alignItems: "center",
                    fontSize: "10px" 
                  }}>
                    <span style={{ fontWeight: "700" }}>{c.id}</span>
                    <span style={{ display: "flex", alignItems: "center", gap: "4px" }}>
                      {c.status === "Active" ? (
                        <>
                          <span className="pulse-green"></span>
                          <span style={{ color: "#4ade80" }}>LIVE</span>
                        </>
                      ) : (
                        <>
                          <span className="pulse-red"></span>
                          <span style={{ color: "#ef4444" }}>OFFLINE</span>
                        </>
                      )}
                    </span>
                  </div>

                  {/* Simulated SVG Camera feed */}
                  <div style={{ height: "130px", position: "relative", overflow: "hidden", cursor: "pointer" }}>
                    {c.status === "Active" ? (
                      <AnimatedSVGStream cameraId={c.id} />
                    ) : (
                      <div style={{ height: "100%", width: "100%", display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", color: "#64748b" }}>
                        <CloseIcon size={28} />
                        <span style={{ fontSize: "10px", marginTop: "4px", fontWeight: "bold" }}>STREAM UNAVAILABLE</span>
                      </div>
                    )}
                  </div>

                  <div style={{ padding: "6px 8px", backgroundColor: "#FCFCFC", borderTop: "1px solid var(--border-color)", fontSize: "10px" }}>
                    <div style={{ fontWeight: "600" }}>{c.name}</div>
                    <div style={{ color: "var(--text-muted)", fontSize: "9px" }}>{c.location}</div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Recent Violations Ticker */}
          <div className="card">
            <div className="card-title">
              <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
                <AlertIcon size={14} style={{ color: "var(--danger)" }} />
                <span>REAL-TIME INFERENCE TICKER ({filteredViolations.length} RECORDS FILTERED)</span>
              </div>
            </div>

            {filteredViolations.length === 0 ? (
              <div style={{ textAlign: "center", padding: "30px 10px", color: "var(--text-muted)" }}>
                No violations match the specified filters.
              </div>
            ) : (
              <div className="table-container">
                <table className="dense-table">
                  <thead>
                    <tr>
                      <th>Violation ID</th>
                      <th>Camera / Location</th>
                      <th>Violation Type</th>
                      <th>Timestamp</th>
                      <th>Plate Number</th>
                      <th>Confidence</th>
                      <th>Review Status</th>
                    </tr>
                  </thead>
                  <tbody>
                    {filteredViolations.slice(0, 7).map(v => (
                      <tr key={v.id}>
                        <td className="mono" style={{ fontWeight: "700" }}>{v.id}</td>
                        <td>
                          <div style={{ fontWeight: "500" }}>{v.cameraId}</div>
                          <div style={{ fontSize: "9px", color: "var(--text-muted)" }}>{v.location}</div>
                        </td>
                        <td>
                          <span style={{ fontWeight: "600", color: v.type === "Red Light" ? "var(--danger)" : "var(--text-accent)" }}>
                            {v.type}
                          </span>
                        </td>
                        <td className="mono" style={{ fontSize: "11px" }}>
                          {new Date(v.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })}
                        </td>
                        <td className="mono" style={{ fontWeight: "bold", background: "var(--bg-tertiary)", padding: "2px 6px", borderRadius: "4px" }}>
                          {v.plateNumber}
                        </td>
                        <td className="mono" style={{ fontWeight: "600" }}>
                          {v.confidenceScore}%
                        </td>
                        <td>
                          <span className={`badge ${STATUS_BADGE_CLASS[v.status]}`}>
                            {STATUS_LABELS[v.status]}
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>

        </div>
      </div>
    </div>
  );
}

// Inline Animation helper for CCTV previews
function AnimatedSVGStream({ cameraId }: { cameraId: string }) {
  // Use unique intervals/animations based on cameraId to prevent synchronization
  const animateDuration = cameraId === "CAM-101" ? "3s" : (cameraId === "CAM-202" ? "2.5s" : "4s");
  const roadColor = "#1e293b";
  const markerColor = "#475569";

  return (
    <svg width="100%" height="100%" viewBox="0 0 200 130" style={{ display: "block" }}>
      {/* Background road lanes */}
      <rect width="200" height="130" fill={roadColor} />
      
      {/* Intersection Lines */}
      <line x1="100" y1="0" x2="100" y2="130" stroke={markerColor} strokeWidth="1.5" strokeDasharray="3,3" />
      <line x1="0" y1="65" x2="200" y2="65" stroke={markerColor} strokeWidth="1.5" strokeDasharray="3,3" />
      
      {/* Stop Line */}
      <line x1="85" y1="0" x2="85" y2="130" stroke="#f1f5f9" strokeWidth="1" />
      <line x1="0" y1="50" x2="200" y2="50" stroke="#f1f5f9" strokeWidth="1" />
      
      {/* Traffic Light indicator */}
      <circle cx="180" cy="20" r="4" fill={cameraId === "CAM-101" ? "#ef4444" : "#22c55e"} />

      {/* Moving Vehicle A - Rightbound */}
      <g>
        <animateTransform
          attributeName="transform"
          type="translate"
          from="-30 55"
          to="230 55"
          dur={animateDuration}
          repeatCount="indefinite"
        />
        {/* Car Shape */}
        <rect width="18" height="8" rx="2" fill="#FEF08A" stroke="#EAB308" strokeWidth="1" />
        <rect width="4" height="2" x="12" y="1" fill="#fff" />
        <rect width="4" height="2" x="12" y="5" fill="#fff" />
        {/* Bounding box for testing visual focus */}
        <rect width="22" height="12" x="-2" y="-2" fill="none" stroke="#EAB308" strokeWidth="0.8" strokeDasharray="2,1" />
      </g>

      {/* Moving Vehicle B - Downbound */}
      <g>
        <animateTransform
          attributeName="transform"
          type="translate"
          from="105 -25"
          to="105 160"
          dur={cameraId === "CAM-303" ? "3.2s" : animateDuration}
          repeatCount="indefinite"
        />
        {/* Truck Shape */}
        <rect width="9" height="20" rx="1" fill="#cbd5e1" stroke="#475569" strokeWidth="1" />
        <rect width="7" height="6" x="1" y="2" fill="#1e293b" />
      </g>

      {/* Lens Overlay */}
      <text x="8" y="15" fill="#ef4444" fontSize="8" fontFamily="monospace" fontWeight="bold">● REC</text>
      <text x="145" y="122" fill="#cbd5e1" fontSize="7" fontFamily="monospace">G-ANALYSIS</text>
    </svg>
  );
}
