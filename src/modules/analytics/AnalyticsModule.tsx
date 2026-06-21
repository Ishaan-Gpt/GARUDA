"use client";

import React, { useState, useMemo } from "react";
import { usePlatform } from "@/context/PlatformContext";
import { ChartIcon } from "@/components/Icons";

export default function AnalyticsModule() {
  const { violations, cameras } = usePlatform();
  const [timeframe, setTimeframe] = useState<"Hourly" | "Daily" | "Weekly" | "Monthly">("Daily");

  // Real trend data, bucketed from actual violation timestamps — no seed/sample values.
  const trendData = useMemo(() => {
    const times = violations.map(v => new Date(v.timestamp)).filter(d => !isNaN(d.getTime()));
    const now = new Date();

    switch (timeframe) {
      case "Hourly": {
        // Today only, bucketed into 4-hour blocks
        const labels = ["00:00", "04:00", "08:00", "12:00", "16:00", "20:00"];
        const counts = new Array(6).fill(0);
        times.forEach(d => {
          if (d.toDateString() === now.toDateString()) {
            counts[Math.floor(d.getHours() / 4)]++;
          }
        });
        return labels.map((label, i) => ({ label, val: counts[i] }));
      }
      case "Daily": {
        // Last 7 calendar days, oldest to newest
        const dayLabels = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
        const days: { label: string; val: number }[] = [];
        for (let i = 6; i >= 0; i--) {
          const d = new Date(now);
          d.setDate(now.getDate() - i);
          const count = times.filter(t => t.toDateString() === d.toDateString()).length;
          days.push({ label: dayLabels[d.getDay()], val: count });
        }
        return days;
      }
      case "Weekly": {
        // Last 4 weeks (7-day buckets ending today)
        const weeks: { label: string; val: number }[] = [];
        for (let i = 3; i >= 0; i--) {
          const end = new Date(now);
          end.setDate(now.getDate() - i * 7);
          const start = new Date(end);
          start.setDate(end.getDate() - 6);
          const count = times.filter(t => t >= start && t <= end).length;
          weeks.push({ label: `Week ${4 - i}`, val: count });
        }
        return weeks;
      }
      case "Monthly": {
        // Last 6 calendar months
        const monthLabels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
        const months: { label: string; val: number }[] = [];
        for (let i = 5; i >= 0; i--) {
          const d = new Date(now.getFullYear(), now.getMonth() - i, 1);
          const count = times.filter(t => t.getFullYear() === d.getFullYear() && t.getMonth() === d.getMonth()).length;
          months.push({ label: monthLabels[d.getMonth()], val: count });
        }
        return months;
      }
    }
  }, [timeframe, violations]);

  // Camera Performance calculations
  const cameraPerformance = useMemo(() => {
    const counts: Record<string, number> = {};
    violations.forEach(v => {
      counts[v.cameraId] = (counts[v.cameraId] || 0) + 1;
    });
    // Fill in 0s for active cameras with no violations
    cameras.forEach(c => {
      if (!counts[c.id]) counts[c.id] = 0;
    });
    return Object.entries(counts).map(([id, count]) => ({ id, count }));
  }, [violations, cameras]);

  // Vehicle type distribution calculations
  const vehicleStats = useMemo(() => {
    const counts = { sedan: 0, suv: 0, motorcycle: 0, truck: 0 };
    violations.forEach(v => {
      const type = v.vehicleType.toLowerCase();
      if (type.includes("sedan") || type.includes("passenger")) counts.sedan++;
      else if (type.includes("suv")) counts.suv++;
      else if (type.includes("motorcycle")) counts.motorcycle++;
      else if (type.includes("truck") || type.includes("heavy")) counts.truck++;
    });
    return counts;
  }, [violations]);

  const daysOfWeek = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];
  const hourBlocks = ["00-04", "04-08", "08-12", "12-16", "16-20", "20-00"];

  // Real heatmap: bucket every violation's timestamp by day-of-week x 4-hour block.
  const heatmapData = useMemo(() => {
    const grid = daysOfWeek.map(() => new Array(6).fill(0));
    violations.forEach(v => {
      const d = new Date(v.timestamp);
      if (isNaN(d.getTime())) return;
      const dayIdx = (d.getDay() + 6) % 7; // getDay(): 0=Sun..6=Sat -> 0=Mon..6=Sun
      const hourBlock = Math.floor(d.getHours() / 4);
      grid[dayIdx][hourBlock]++;
    });
    return grid;
  }, [violations]);
  const heatmapMax = Math.max(1, ...heatmapData.flat());

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "12px" }}>
      
      {/* Page Header */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <div>
          <h1 style={{ fontSize: "20px", fontWeight: "700", letterSpacing: "-0.5px" }}>TRAFFIC VIOLATION ANALYTICS</h1>
          <p style={{ fontSize: "11px", color: "var(--text-muted)", textTransform: "uppercase", marginTop: "2px" }}>
            Aggregate citation trends, sensor activity metrics, vehicle distribution and reviewer validation audits
          </p>
        </div>

        {/* Timeframe switch */}
        <div style={{ display: "flex", border: "1px solid var(--border-color)", borderRadius: "4px", overflow: "hidden", backgroundColor: "#FFF" }}>
          {(["Hourly", "Daily", "Weekly", "Monthly"] as const).map((t) => (
            <button
              key={t}
              onClick={() => setTimeframe(t)}
              style={{
                padding: "4px 10px",
                border: "none",
                fontSize: "11px",
                fontWeight: "600",
                cursor: "pointer",
                backgroundColor: timeframe === t ? "var(--border-accent)" : "#FFF",
                color: timeframe === t ? "var(--text-accent)" : "var(--text-secondary)",
                transition: "background 0.15s"
              }}
            >
              {t.toUpperCase()}
            </button>
          ))}
        </div>
      </div>

      {/* Main Charts Grid */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "12px" }}>
        
        {/* Chart 1: Violation Trend (SVG Line/Area Chart) */}
        <div className="card">
          <div className="card-title">
            <span>VIOLATION INCIDENT TRENDS ({timeframe.toUpperCase()})</span>
          </div>

          <div style={{ padding: "8px 0" }}>
            {/* Hand-crafted SVG Chart */}
            <svg viewBox="0 0 400 180" className="chart-svg">
              {/* Grid Lines */}
              <line x1="40" y1="20" x2="380" y2="20" className="chart-gridline" />
              <line x1="40" y1="60" x2="380" y2="60" className="chart-gridline" />
              <line x1="40" y1="100" x2="380" y2="100" className="chart-gridline" />
              <line x1="40" y1="140" x2="380" y2="140" className="chart-gridline" />
              
              {/* Axes */}
              <line x1="40" y1="140" x2="380" y2="140" className="chart-axis" />
              <line x1="40" y1="20" x2="40" y2="140" className="chart-axis" />

              {/* Draw Area & Line dynamically */}
              {(() => {
                const maxVal = Math.max(...trendData.map(d => d.val), 10);
                const points = trendData.map((d, index) => {
                  const x = 40 + (index * (340 / (trendData.length - 1)));
                  const y = 140 - (d.val / maxVal) * 110;
                  return { x, y };
                });

                const linePath = points.map((p, i) => `${i === 0 ? "M" : "L"} ${p.x} ${p.y}`).join(" ");
                const areaPath = `${linePath} L ${points[points.length - 1].x} 140 L ${points[0].x} 140 Z`;

                return (
                  <>
                    {/* Area fill */}
                    <path d={areaPath} className="chart-area" />
                    {/* Stroke line */}
                    <path d={linePath} className="chart-line" />
                    
                    {/* Points & Labels */}
                    {points.map((p, i) => (
                      <g key={i}>
                        <circle cx={p.x} cy={p.y} r="4" fill="var(--border-accent-dark)" stroke="#FFF" strokeWidth="1.5" />
                        <text x={p.x} y={p.y - 8} textAnchor="middle" className="chart-label" style={{ fontWeight: "700", fill: "var(--text-primary)" }}>
                          {trendData[i].val}
                        </text>
                        {/* X Axis Label */}
                        <text x={p.x} y="155" textAnchor="middle" className="chart-label">
                          {trendData[i].label}
                        </text>
                      </g>
                    ))}
                  </>
                );
              })()}
            </svg>
          </div>
        </div>

        {/* Chart 2: Camera Performance (Horizontal SVG Bar Chart) */}
        <div className="card">
          <div className="card-title">
            <span>DETECTIONS BY CAMERA REGISTRY ENDPOINT</span>
          </div>

          <div style={{ padding: "8px 0" }}>
            <svg viewBox="0 0 400 180" className="chart-svg">
              {/* Y Axis line */}
              <line x1="60" y1="10" x2="60" y2="150" className="chart-axis" />
              <line x1="60" y1="150" x2="380" y2="150" className="chart-axis" />

              {/* Draw Horizontal Bars */}
              {(() => {
                const maxVal = Math.max(...cameraPerformance.map(c => c.count), 5);
                const barHeight = 16;
                const gap = 12;

                return cameraPerformance.map((cam, idx) => {
                  const y = 20 + idx * (barHeight + gap);
                  const barWidth = Math.max(5, (cam.count / maxVal) * 280);

                  return (
                    <g key={cam.id}>
                      {/* Camera ID label */}
                      <text x="50" y={y + 11} textAnchor="end" className="chart-label" style={{ fontWeight: "600" }}>
                        {cam.id}
                      </text>
                      
                      {/* Bar rectangle */}
                      <rect 
                        x="60" 
                        y={y} 
                        width={barWidth} 
                        height={barHeight} 
                        rx="2"
                        className="chart-bar" 
                      />

                      {/* Value label */}
                      <text x={65 + barWidth} y={y + 11} className="chart-label" style={{ fontWeight: "700" }}>
                        {cam.count} vios
                      </text>
                    </g>
                  );
                });
              })()}
            </svg>
          </div>
        </div>

        {/* Chart 3: Vehicle Category distribution (SVG Donut Chart) */}
        <div className="card">
          <div className="card-title">
            <span>VEHICLE CLASS DISTRIBUTION</span>
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "1fr 150px", gap: "10px", alignItems: "center", padding: "10px 0" }}>
            {/* Custom SVG Donut */}
            <svg viewBox="0 0 100 100" style={{ width: "100%", height: "130px" }}>
              {(() => {
                const total = vehicleStats.sedan + vehicleStats.suv + vehicleStats.motorcycle + vehicleStats.truck;
                const safeTotal = total || 1; // avoid div-by-zero when there's no data yet
                const pSedan = (vehicleStats.sedan / safeTotal) * 100;
                const pSuv = (vehicleStats.suv / safeTotal) * 100;
                const pMoto = (vehicleStats.motorcycle / safeTotal) * 100;
                const pTruck = (vehicleStats.truck / safeTotal) * 100;

                // SVG stroke-dasharray segments calculations (Radius 36, Circumference = ~226.2)
                const c = 226.2;
                const dSedan = (pSedan / 100) * c;
                const dSuv = (pSuv / 100) * c;
                const dMoto = (pMoto / 100) * c;
                const dTruck = (pTruck / 100) * c;

                return (
                  <>
                    <circle cx="50" cy="50" r="36" fill="transparent" stroke="#E2E8F0" strokeWidth="12" />
                    
                    {/* Sedan section (Pastel Yellow / Primary Accent) */}
                    <circle 
                      cx="50" cy="50" r="36" fill="transparent" 
                      stroke="var(--border-accent-dark)" 
                      strokeWidth="12" 
                      strokeDasharray={`${dSedan} ${c}`}
                      strokeDashoffset="0"
                    />

                    {/* SUV section (Blue) */}
                    <circle 
                      cx="50" cy="50" r="36" fill="transparent" 
                      stroke="#3B82F6" 
                      strokeWidth="12" 
                      strokeDasharray={`${dSuv} ${c}`}
                      strokeDashoffset={-dSedan}
                    />

                    {/* Motorcycle section (Green) */}
                    <circle 
                      cx="50" cy="50" r="36" fill="transparent" 
                      stroke="#10B981" 
                      strokeWidth="12" 
                      strokeDasharray={`${dMoto} ${c}`}
                      strokeDashoffset={-(dSedan + dSuv)}
                    />

                    {/* Heavy Truck section (Purple) */}
                    <circle 
                      cx="50" cy="50" r="36" fill="transparent" 
                      stroke="#8B5CF6" 
                      strokeWidth="12" 
                      strokeDasharray={`${dTruck} ${c}`}
                      strokeDashoffset={-(dSedan + dSuv + dMoto)}
                    />

                    {/* Center details tag */}
                    <circle cx="50" cy="50" r="26" fill="#FFF" />
                    <text x="50" y="47" textAnchor="middle" fontSize="7" fill="var(--text-muted)" fontWeight="600">TOTAL</text>
                    <text x="50" y="58" textAnchor="middle" fontSize="11" fill="var(--text-primary)" fontFamily="var(--font-mono)" fontWeight="700">
                      {total}
                    </text>
                  </>
                );
              })()}
            </svg>

            {/* Donut Legend */}
            <div style={{ display: "flex", flexDirection: "column", gap: "6px", fontSize: "11px" }}>
              <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
                <span style={{ width: "10px", height: "10px", backgroundColor: "var(--border-accent-dark)", borderRadius: "2px" }}></span>
                <span>Passenger Car ({vehicleStats.sedan})</span>
              </div>
              <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
                <span style={{ width: "10px", height: "10px", backgroundColor: "#3B82F6", borderRadius: "2px" }}></span>
                <span>SUV ({vehicleStats.suv})</span>
              </div>
              <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
                <span style={{ width: "10px", height: "10px", backgroundColor: "#10B981", borderRadius: "2px" }}></span>
                <span>Motorcycle ({vehicleStats.motorcycle})</span>
              </div>
              <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
                <span style={{ width: "10px", height: "10px", backgroundColor: "#8B5CF6", borderRadius: "2px" }}></span>
                <span>Heavy Truck ({vehicleStats.truck})</span>
              </div>
            </div>
          </div>
        </div>

        {/* Chart 4: Hourly Heatmap distribution grid */}
        <div className="card">
          <div className="card-title">
            <span>WEEKLY HOUR SPAN HEATMAP (INFRACTION SPECTRUM)</span>
          </div>

          <div style={{ padding: "8px 0" }}>
            {/* Render Heatmap grid */}
            <div style={{ display: "flex", flexDirection: "column", gap: "3px" }}>
              
              {/* Hour Blocks Headers */}
              <div style={{ display: "flex", gap: "3px" }}>
                <div style={{ width: "35px" }}></div>
                {hourBlocks.map((b) => (
                  <div key={b} style={{ flex: 1, textAnchor: "middle", textAlign: "center", fontSize: "8px", fontWeight: "600", color: "var(--text-muted)" }}>
                    {b}
                  </div>
                ))}
              </div>

              {/* Heatmap Matrix rows */}
              {daysOfWeek.map((day, dIdx) => (
                <div key={day} style={{ display: "flex", gap: "3px", alignItems: "center" }}>
                  {/* Day Header Label */}
                  <div style={{ width: "35px", fontSize: "9px", fontWeight: "700", color: "var(--text-secondary)" }}>{day}</div>
                  
                  {/* Block cells */}
                  {heatmapData[dIdx].map((val, hIdx) => {
                    const opacity = val === 0 ? 0.06 : Math.min(0.95, Math.max(0.15, val / heatmapMax));
                    const cellBg = `rgba(234, 179, 8, ${opacity})`;
                    const textColor = opacity > 0.5 ? "#FFF" : "var(--text-primary)";

                    return (
                      <div 
                        key={hIdx} 
                        style={{ 
                          flex: 1, 
                          height: "18px", 
                          backgroundColor: cellBg, 
                          borderRadius: "2px", 
                          display: "flex", 
                          alignItems: "center", 
                          justifyContent: "center",
                          fontSize: "8px",
                          fontFamily: "var(--font-mono)",
                          fontWeight: "bold",
                          color: textColor,
                          cursor: "pointer"
                        }}
                        title={`${day} between ${hourBlocks[hIdx]} had ${val} violations`}
                      >
                        {val}
                      </div>
                    );
                  })}
                </div>
              ))}
            </div>

            {/* Heatmap key */}
            <div style={{ display: "flex", justifyContent: "flex-end", alignItems: "center", gap: "6px", marginTop: "8px", fontSize: "9px", color: "var(--text-muted)" }}>
              <span>Low activity</span>
              <span style={{ width: "12px", height: "12px", backgroundColor: "rgba(234, 179, 8, 0.1)", borderRadius: "2px" }}></span>
              <span style={{ width: "12px", height: "12px", backgroundColor: "rgba(234, 179, 8, 0.4)", borderRadius: "2px" }}></span>
              <span style={{ width: "12px", height: "12px", backgroundColor: "rgba(234, 179, 8, 0.8)", borderRadius: "2px" }}></span>
              <span>High activity</span>
            </div>
          </div>
        </div>

      </div>
    </div>
  );
}
