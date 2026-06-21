"use client";

import React, { useState, useMemo } from "react";
import { usePlatform } from "@/context/PlatformContext";
import { ChevronLeftIcon, ChevronRightIcon } from "@/components/Icons";
import { VIOLATION_TYPES, STATUS_LABELS, STATUS_BADGE_CLASS, SEVERITY_COLOR, ViolationStatus } from "@/lib/violations";
import Link from "next/link";

export default function ViolationsModule() {
  const { violations, cameras, role, reviewViolation } = usePlatform();

  // Search & Filter state
  const [searchTerm, setSearchTerm] = useState("");
  const [filterCamera, setFilterCamera] = useState("all");
  const [filterLocation, setFilterLocation] = useState("all");
  const [filterType, setFilterType] = useState("all");
  const [filterStatus, setFilterStatus] = useState("all");
  const [minConfidence, setMinConfidence] = useState(0);
  const [filterDate, setFilterDate] = useState("");

  // Pagination State
  const [currentPage, setCurrentPage] = useState(1);
  const itemsPerPage = 10;

  // Unique list of locations for the location filter
  const locations = useMemo(() => {
    return Array.from(new Set(cameras.map(c => c.location)));
  }, [cameras]);

  // Filter logic
  const filteredViolations = useMemo(() => {
    return violations.filter(v => {
      if (searchTerm && !v.plateNumber.toLowerCase().includes(searchTerm.toLowerCase()) && !v.id.toLowerCase().includes(searchTerm.toLowerCase())) return false;
      if (filterCamera !== "all" && v.cameraId !== filterCamera) return false;
      if (filterLocation !== "all" && v.location !== filterLocation) return false;
      if (filterType !== "all" && v.type !== filterType) return false;
      if (filterStatus !== "all" && v.status !== filterStatus) return false;
      if (v.confidenceScore < minConfidence) return false;
      if (filterDate && new Date(filterDate).toDateString() !== new Date(v.timestamp).toDateString()) return false;
      return true;
    });
  }, [violations, searchTerm, filterCamera, filterLocation, filterType, filterStatus, minConfidence, filterDate]);

  // Pagination calculation
  const totalPages = Math.max(1, Math.ceil(filteredViolations.length / itemsPerPage));
  const paginatedViolations = useMemo(() => {
    const startIndex = (currentPage - 1) * itemsPerPage;
    return filteredViolations.slice(startIndex, startIndex + itemsPerPage);
  }, [filteredViolations, currentPage]);

  const handlePageChange = (newPage: number) => {
    if (newPage >= 1 && newPage <= totalPages) {
      setCurrentPage(newPage);
    }
  };

  // Quick inline actions (Available for Admin/Supervisor/Reviewer) — only
  // applies to violations actually awaiting a decision ("pending"); auto-
  // cleared/confirmed/rejected events are already resolved.
  const canQuickReview = role !== "Operator";

  const handleQuickAction = (id: string, action: "Approved" | "Rejected") => {
    reviewViolation(id, action, `Officer ${role}`, "Quick action from Violation Center");
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "12px" }}>
      <div>
        <h1 style={{ fontSize: "20px", fontWeight: "700", letterSpacing: "-0.5px" }}>VIOLATION CENTER & EVENT REGISTRY</h1>
        <p style={{ fontSize: "11px", color: "var(--text-muted)", textTransform: "uppercase", marginTop: "2px" }}>
          Real-time query database for all AI edge-generated traffic citation events
        </p>
      </div>

      {/* Advanced Filter Panel */}
      <div className="filter-bar">
        <div className="filter-item" style={{ flex: 1.5 }}>
          <label className="form-label">Search Plate / ID</label>
          <div style={{ position: "relative" }}>
            <input
              type="text"
              className="form-input"
              placeholder="e.g. KA01AB1234 or VIO-..."
              value={searchTerm}
              onChange={(e) => { setSearchTerm(e.target.value); setCurrentPage(1); }}
            />
          </div>
        </div>

        <div className="filter-item">
          <label className="form-label">Sensor / Camera</label>
          <select
            className="form-input"
            value={filterCamera}
            onChange={(e) => { setFilterCamera(e.target.value); setCurrentPage(1); }}
          >
            <option value="all">All Cameras</option>
            {cameras.map(c => (
              <option key={c.id} value={c.id}>{c.id}</option>
            ))}
          </select>
        </div>

        <div className="filter-item">
          <label className="form-label">Location / Source</label>
          <select
            className="form-input"
            value={filterLocation}
            onChange={(e) => { setFilterLocation(e.target.value); setCurrentPage(1); }}
          >
            <option value="all">All Locations</option>
            {locations.map(loc => (
              <option key={loc} value={loc}>{loc}</option>
            ))}
          </select>
        </div>

        <div className="filter-item">
          <label className="form-label">Violation Category</label>
          <select
            className="form-input"
            value={filterType}
            onChange={(e) => { setFilterType(e.target.value); setCurrentPage(1); }}
          >
            <option value="all">All Types</option>
            {VIOLATION_TYPES.map(t => (
              <option key={t} value={t}>{t}</option>
            ))}
          </select>
        </div>

        <div className="filter-item">
          <label className="form-label">Exact Date</label>
          <input
            type="date"
            className="form-input"
            value={filterDate}
            onChange={(e) => { setFilterDate(e.target.value); setCurrentPage(1); }}
          />
        </div>

        <div className="filter-item">
          <label className="form-label">Review Status</label>
          <select
            className="form-input"
            value={filterStatus}
            onChange={(e) => { setFilterStatus(e.target.value); setCurrentPage(1); }}
          >
            <option value="all">All States</option>
            {(Object.keys(STATUS_LABELS) as ViolationStatus[]).map(s => (
              <option key={s} value={s}>{STATUS_LABELS[s]}</option>
            ))}
          </select>
        </div>

        <div className="filter-item" style={{ minWidth: "180px" }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
            <label className="form-label">Min Confidence</label>
            <span className="mono" style={{ fontSize: "10px", fontWeight: "bold" }}>{minConfidence}%</span>
          </div>
          <input
            type="range"
            min="0"
            max="95"
            step="5"
            value={minConfidence}
            onChange={(e) => { setMinConfidence(parseInt(e.target.value)); setCurrentPage(1); }}
            style={{ width: "100%", accentColor: "var(--border-accent-dark)", cursor: "pointer" }}
          />
        </div>
      </div>

      {/* Database Event Table */}
      <div className="card">
        <div className="card-title">
          <span>VIOLATION EVENT LOGS ({filteredViolations.length} RECORDS FOUND)</span>
          <span className="brand-badge">LIVE FEED + REST SYNC</span>
        </div>

        {filteredViolations.length === 0 ? (
          <div style={{ textAlign: "center", padding: "60px 10px", color: "var(--text-muted)" }}>
            No violation events found matching the specified parameters.
          </div>
        ) : (
          <>
            <div className="table-container">
              <table className="dense-table">
                <thead>
                  <tr>
                    <th>Event ID</th>
                    <th>Violation Category</th>
                    <th>Severity</th>
                    <th>Date / Time</th>
                    <th>Source / Location</th>
                    <th>Vehicle</th>
                    <th>OCR Plate</th>
                    <th>Confidence</th>
                    <th>Fine</th>
                    <th>Status</th>
                    <th style={{ textAlign: "right" }}>Workflow Action</th>
                  </tr>
                </thead>
                <tbody>
                  {paginatedViolations.map((v) => (
                    <tr key={v.id}>
                      <td className="mono" style={{ fontWeight: "700" }}>{v.id}</td>
                      <td>
                        <span style={{ color: "var(--text-accent)", fontWeight: "600" }}>
                          {v.type}
                        </span>
                      </td>
                      <td>
                        <span style={{
                          fontSize: "10px",
                          fontWeight: "700",
                          textTransform: "uppercase",
                          color: SEVERITY_COLOR[v.severity] || "var(--text-muted)"
                        }}>
                          {v.severity}
                        </span>
                      </td>
                      <td className="mono" style={{ fontSize: "11px" }}>
                        {new Date(v.timestamp).toLocaleDateString()} {new Date(v.timestamp).toLocaleTimeString()}
                      </td>
                      <td>
                        <div style={{ fontSize: "12px" }}>{v.location}</div>
                        <div className="mono" style={{ fontSize: "9px", color: "var(--text-muted)" }}>{v.cameraId}</div>
                      </td>
                      <td>{v.vehicleType}</td>
                      <td>
                        <span className="mono" style={{
                          fontWeight: "bold",
                          background: "#FEF9C3",
                          border: "1px solid var(--border-accent-dark)",
                          padding: "2px 6px",
                          borderRadius: "4px",
                          color: "var(--text-accent)"
                        }}>
                          {v.plateNumber}
                        </span>
                      </td>
                      <td className="mono" style={{ fontWeight: "700" }}>{v.confidenceScore}%</td>
                      <td className="mono" style={{ fontSize: "11px" }}>₹{v.fineAmountInr}</td>
                      <td>
                        <span className={`badge ${STATUS_BADGE_CLASS[v.status]}`}>
                          {STATUS_LABELS[v.status]}
                        </span>
                      </td>
                      <td style={{ textAlign: "right" }}>
                        <div style={{ display: "flex", gap: "4px", justifyContent: "flex-end" }}>
                          <Link href={`/review?id=${v.id}`} className="btn btn-secondary btn-sm">
                            INSPECT
                          </Link>

                          {canQuickReview && v.status === "pending" && (
                            <>
                              <button
                                className="btn btn-success btn-sm"
                                title="Approve Citation"
                                onClick={() => handleQuickAction(v.id, "Approved")}
                              >
                                ✓
                              </button>
                              <button
                                className="btn btn-danger btn-sm"
                                title="Reject Citation"
                                onClick={() => handleQuickAction(v.id, "Rejected")}
                              >
                                ✕
                              </button>
                            </>
                          )}
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {/* Pagination Controls */}
            <div style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
              marginTop: "12px",
              paddingTop: "12px",
              borderTop: "1px solid var(--border-color)"
            }}>
              <span style={{ fontSize: "11px", color: "var(--text-muted)" }}>
                Showing page <strong style={{ color: "var(--text-primary)" }}>{currentPage}</strong> of <strong style={{ color: "var(--text-primary)" }}>{totalPages}</strong> ({filteredViolations.length} total events)
              </span>

              <div style={{ display: "flex", gap: "6px" }}>
                <button
                  className="btn btn-secondary btn-sm"
                  onClick={() => handlePageChange(currentPage - 1)}
                  disabled={currentPage === 1}
                  style={{ display: "flex", alignItems: "center", gap: "4px" }}
                >
                  <ChevronLeftIcon size={12} /> PREV
                </button>
                <button
                  className="btn btn-secondary btn-sm"
                  onClick={() => handlePageChange(currentPage + 1)}
                  disabled={currentPage === totalPages}
                  style={{ display: "flex", alignItems: "center", gap: "4px" }}
                >
                  NEXT <ChevronRightIcon size={12} />
                </button>
              </div>
            </div>
          </>
        )}

      </div>
    </div>
  );
}
