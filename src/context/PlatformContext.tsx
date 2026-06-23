"use client";

import React, { createContext, useContext, useState, useEffect, useCallback, useRef } from "react";
import {
  ViolationStatus,
  PlateDetection,
  DriverAlert,
  ProcessingInfo,
  normalizeConfidencePct,
} from "@/lib/violations";

// Definitions
export type CameraStatus = "Active" | "Offline" | "Disabled";
export type { ViolationStatus };
export type JobStatus = "Queued" | "Processing" | "Completed" | "Failed";
export type UserRole = "Admin" | "Supervisor" | "Reviewer" | "Operator";

export interface Camera {
  id: string;
  name: string;
  rtspUrl: string;
  location: string;
  status: CameraStatus;
  lastHeartbeat: string;
  resolution: string;
}

// Mirrors backend/models/schemas.py ViolationResponse / ViolationDetailResponse
// exactly. allPlates/driverAlerts/processing are only populated when the full
// json_record is available (single-violation fetch, job-violations fetch) —
// they are undefined, not fake, when only the list endpoint was used.
export interface Violation {
  id: string;
  type: string;            // one of VIOLATION_TYPES — real display label
  timestamp: string;
  location: string;        // free-text source label: camera location, uploaded filename, or patrol/public source
  cameraId: string;
  vehicleType: string;
  plateNumber: string;
  plateConfidence: number; // 0-100
  confidenceScore: number; // 0-100
  severity: string;        // low | medium | high | critical
  tier: number;            // 1 = auto-cleared, 2 = needs review, 3 = low priority
  action: string;          // "PASSED" | "HUMAN_REVIEW" etc — raw backend action code
  fineAmountInr: number;
  status: ViolationStatus;
  officerId?: string;
  createdAt: string;
  annotatedImg: string;    // path to the real annotated evidence JPEG
  rawImg: string;          // path to the real raw frame JPEG

  // Detail-only real fields (present after fetching json_record)
  allPlatesDetected?: PlateDetection[];
  driverAlerts?: DriverAlert[];
  processing?: ProcessingInfo;
  // Every individual violation clubbed into this one record (one image can
  // carry several — helmet + triple riding on the same frame). Each can be
  // approved/rejected independently via reviewViolationItem(). Length 1 for
  // the common single-violation case.
  violationItems?: ViolationItem[];
}

export interface ViolationItem {
  type: string;
  confidence: number; // 0-100
  severity: string;
  fineAmountInr: number;
  bbox: number[];
  tier: number;
  reviewStatus: "pending" | "auto_confirmed" | "confirmed" | "rejected";
  escalated?: boolean;
  plateText?: string;
}

export interface ProcessingJob {
  id: string;
  name: string;
  sourceType: "Image" | "Video" | "Batch";
  uploadTime: string;
  duration: number; // seconds
  framesProcessed: number;
  violationsFound: number;
  status: JobStatus;
  progress: number; // 0 to 100
  cameraId?: string | null; // registered camera this job was calibrated against, if any
}

export interface SystemNotification {
  id: string;
  type: "Camera Offline" | "Processing Failure" | "Model Failure" | "Violation Spike" | "System Alert";
  message: string;
  timestamp: string;
  read: boolean;
  severity: "low" | "medium" | "high";
}

interface PlatformContextType {
  // Auth state
  token: string | null;
  user: { id: string; name: string; email: string; role: UserRole; status: string } | null;
  isAuthenticated: boolean;
  authLoading: boolean;
  login: (email: string, password: string) => Promise<boolean>;
  register: (name: string, email: string, password: string, role: string) => Promise<string>;
  logout: () => void;
  usersList: any[];
  auditLogs: any[];
  updateUserRole: (userId: string, newRole: UserRole) => Promise<boolean>;

  role: UserRole;
  setRole: (role: UserRole) => void;
  cameras: Camera[];
  violations: Violation[];
  jobs: ProcessingJob[];
  notifications: SystemNotification[];
  
  // Actions
  addCamera: (camera: Omit<Camera, "lastHeartbeat">) => void;
  toggleCameraStatus: (id: string, status: CameraStatus) => void;
  testCameraConnection: (id: string) => Promise<boolean>;
  deleteCamera: (id: string) => void;
  deleteViolation: (id: string) => Promise<void>;
  batchDeleteViolations: (ids: string[]) => Promise<void>;
  sendChallanSms: (id: string) => Promise<void>;
  
  reviewViolation: (
    id: string,
    action: "Approved" | "Rejected" | "Escalated",
    reviewer: string,
    reason?: string,
    correctedPlateText?: string,
  ) => void;
  reviewViolationItem: (
    id: string,
    itemIndex: number,
    action: "Approved" | "Rejected" | "Escalated",
    reviewer: string,
    reason?: string,
  ) => void;

  submitUploadJob: (name: string, type: "Image" | "Video", file?: File) => void;
  fetchViolationDetail: (id: string) => Promise<Violation | null>;
  clearNotifications: () => void;
  markNotificationRead: (id: string) => void;
  
  // Simulation Controller state
  isSimulating: boolean;
  setIsSimulating: (sim: boolean) => void;
  simulationInterval: number; // ms
  setSimulationInterval: (interval: number) => void;
  isBackendConnected: boolean;
  isWsConnected: boolean;
}

const PlatformContext = createContext<PlatformContextType | undefined>(undefined);

// API Endpoints
let API_BASE = "http://localhost:8000/api/v1";
let WS_BASE = "ws://localhost:8000";

if (process.env.NEXT_PUBLIC_API_URL) {
  const url = process.env.NEXT_PUBLIC_API_URL.replace(/\/$/, "");
  API_BASE = url.endsWith("/api/v1") ? url : `${url}/api/v1`;
  WS_BASE = url.startsWith("https://")
    ? url.replace("https://", "wss://")
    : url.replace("http://", "ws://");
} else if (typeof window !== "undefined") {
  const protocol = window.location.protocol === "https:" ? "https" : "http";
  const wsProtocol = window.location.protocol === "https:" ? "wss" : "ws";
  API_BASE = `${protocol}://${window.location.hostname}:8000/api/v1`;
  WS_BASE = `${wsProtocol}://${window.location.hostname}:8000`;
}

const initialNotifications: SystemNotification[] = [
  { id: "NOT-001", type: "System Alert", message: "Garuda Intelligence Platform Online. Connecting to backend...", timestamp: new Date().toISOString(), read: false, severity: "low" }
];

export const PlatformProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  // Auth state
  const [token, setToken] = useState<string | null>(null);
  const [user, setUser] = useState<{ id: string; name: string; email: string; role: UserRole; status: string } | null>(null);
  const [authLoading, setAuthLoading] = useState(true);
  const [usersList, setUsersList] = useState<any[]>([]);
  const [auditLogs, setAuditLogs] = useState<any[]>([]);

  const [role, setRole] = useState<UserRole>("Admin");
  const [cameras, setCameras] = useState<Camera[]>([]);
  const [violations, setViolations] = useState<Violation[]>([]);
  const [jobs, setJobs] = useState<ProcessingJob[]>([]);
  const [notifications, setNotifications] = useState<SystemNotification[]>(initialNotifications);
  
  const [isSimulating, setIsSimulating] = useState(false);
  const [simulationInterval, setSimulationInterval] = useState(12000); 
  const [isBackendConnected, setIsBackendConnected] = useState(false);
  const [isWsConnected, setIsWsConnected] = useState(false);

  const socketRef = useRef<WebSocket | null>(null);

  // Helper to trigger internal notification
  const triggerNotification = useCallback((type: SystemNotification["type"], message: string, severity: SystemNotification["severity"] = "medium") => {
    const newNot: SystemNotification = {
      id: `NOT-${Date.now()}-${Math.floor(Math.random() * 1000)}`,
      type,
      message,
      timestamp: new Date().toISOString(),
      read: false,
      severity
    };
    setNotifications(prev => [newNot, ...prev]);
  }, []);

  // Map backend JSON structures to UI violations
  const mapBackendViolationToUI = useCallback((v: any): Violation => {
    const confidenceScore = v.confidence > 1 ? v.confidence : parseFloat((v.confidence * 100).toFixed(1));
    
    let record: any = {};
    try {
      if (typeof v.json_record === "string") {
        record = JSON.parse(v.json_record || "{}");
      } else {
        record = v.json_record || {};
      }
    } catch (e) {
      record = {};
    }
    
    // status is always one of the 4 values the backend actually persists
    // (backend/api/reviews.py / database.py) — no fictional intermediate states.
    const status: ViolationStatus =
      v.status === "confirmed" || v.status === "auto_challan" || v.status === "rejected"
        ? v.status
        : "pending";

    const allPlatesDetected = Array.isArray(record.all_plates_detected)
      ? record.all_plates_detected.map((p: any) => ({
          plateText: p.plate_text || "UNCLEAR",
          confidence: normalizeConfidencePct(p.confidence || 0),
          vehicleClass: p.vehicle_class || "unknown",
          bbox: p.bbox || [],
          ocrEngine: p.ocr_engine || "unknown",
          state: p.state || "Unknown",
          isValid: !!p.is_valid,
        }))
      : undefined;

    const driverAlerts = Array.isArray(record.driver_state?.alerts)
      ? record.driver_state.alerts.map((a: any) => ({
          alertType: a.alert_type,
          severity: a.severity,
          action: a.action,
          confidence: a.confidence,
          trackId: a.track_id ?? null,
          metadata: a.metadata || {},
        }))
      : undefined;

    const processing = record.processing
      ? {
          inferenceDevice: record.processing.inference_device || "Unknown",
          inferenceTimeMs: record.processing.inference_time_ms || 0,
          model: record.processing.model || "Unknown",
          ocrEngine: record.processing.ocr_engine || "Unknown",
          vehiclesDetected: record.processing.vehicles_detected || 0,
          personsDetected: record.processing.persons_detected || 0,
        }
      : undefined;

    const violationItems: ViolationItem[] | undefined = Array.isArray(record.violations)
      ? record.violations.map((rv: any) => ({
          type: rv.type || "Unknown",
          confidence: normalizeConfidencePct(rv.confidence || 0),
          severity: rv.severity || "medium",
          fineAmountInr: rv.fine_amount_inr || 0,
          bbox: rv.bbox || [],
          tier: rv.tier ?? 2,
          reviewStatus: rv.review_status || "pending",
          escalated: !!rv.escalated,
          plateText: rv.plate_text,
        }))
      : undefined;

    return {
      id: v.id,
      type: v.violation_type,
      timestamp: v.timestamp,
      location: v.location,
      cameraId: v.camera_id,
      vehicleType: v.vehicle_class || "unknown",
      plateNumber: v.plate_text || "UNCLEAR",
      plateConfidence: normalizeConfidencePct(v.plate_conf || 0),
      confidenceScore,
      severity: v.severity || "medium",
      tier: v.tier ?? 2,
      action: v.action || "",
      fineAmountInr: v.fine_amount || 0,
      status,
      officerId: v.officer_id || undefined,
      createdAt: v.created_at || v.timestamp,
      annotatedImg: v.annotated_img || "",
      rawImg: v.raw_img || "",
      allPlatesDetected,
      driverAlerts,
      processing,
      violationItems,
    };
  }, []);

  // Fetch all database records from Backend REST API
  const syncWithBackend = useCallback(async () => {
    try {
      const headers: HeadersInit = token ? { "Authorization": `Bearer ${token}` } : {};

      // Sync Cameras
      const camRes = await fetch(`${API_BASE}/cameras`, { headers });
      if (camRes.ok) {
        const camsData = await camRes.json();
        if (camsData.length > 0) {
          const mappedCams: Camera[] = camsData.map((c: any) => ({
            id: c.id,
            name: c.description || `Camera ${c.id}`,
            rtspUrl: c.rtsp_url || "",
            location: c.location,
            status: c.status === "active" ? "Active" : "Offline",
            lastHeartbeat: c.last_seen || new Date().toISOString(),
            resolution: c.resolution || "Unknown"
          }));
          setCameras(mappedCams);
        }
      }

      // Sync Violations
      const vioRes = await fetch(`${API_BASE}/violations?page_size=100`, { headers });
      if (vioRes.ok) {
        const viosData = await vioRes.json();
        if (viosData.violations) {
          const mappedVios = viosData.violations.map(mapBackendViolationToUI);
          setViolations(mappedVios);
        }
      }

      // Sync Ingestion Jobs
      const jobRes = await fetch(`${API_BASE}/jobs`, { headers });
      if (jobRes.ok) {
        const jobsData = await jobRes.json();
        const mappedJobs: ProcessingJob[] = jobsData.map((j: any) => ({
          id: j.id,
          name: j.name,
          sourceType: j.source_type,
          uploadTime: j.upload_time,
          duration: j.duration,
          framesProcessed: j.frames_processed,
          violationsFound: j.violations_found,
          status: j.status,
          progress: j.progress,
          cameraId: j.camera_id ?? null,
        }));
        setJobs(mappedJobs);
      }

      // Sync Users roster and Audit logs if authenticated
      if (token) {
        const usersRes = await fetch(`${API_BASE}/users`, { headers });
        if (usersRes.ok) {
          const uData = await usersRes.json();
          setUsersList(uData);
        }
        
        const logsRes = await fetch(`${API_BASE}/audit-logs`, { headers });
        if (logsRes.ok) {
          const lData = await logsRes.json();
          setAuditLogs(lData);
        }
      }

      setIsBackendConnected(true);
    } catch (error) {
      console.log("FastAPI backend offline. Running in standalone simulated mode.");
      setIsBackendConnected(false);
    }
  }, [mapBackendViolationToUI, token]);

  // Auth bootstrap on mount
  useEffect(() => {
    const bootstrapAuth = async () => {
      const storedToken = localStorage.getItem("garuda_token");
      if (storedToken) {
        try {
          const res = await fetch(`${API_BASE}/auth/me`, {
            headers: { "Authorization": `Bearer ${storedToken}` }
          });
          if (res.ok) {
            const data = await res.json();
            setToken(storedToken);
            setUser(data);
            setRole(data.role);
          } else {
            localStorage.removeItem("garuda_token");
          }
        } catch {
          console.log("Auth API offline on startup. Fallback to mock session.");
          // Fallback mockup
          setToken(storedToken);
          setUser({ id: "USR-MOCK", name: "Officer Keshav", email: "keshav@enforcement.gov", role: "Admin", status: "Active" });
          setRole("Admin");
        }
      }
      setAuthLoading(false);
    };

    bootstrapAuth();
  }, []);

  // Establish WebSockets Feed connection
  useEffect(() => {
    const connectWS = () => {
      if (socketRef.current) return;
      
      const ws = new WebSocket(`${WS_BASE}/ws/feed`);
      
      ws.onopen = () => {
        console.log("WebSocket connected to GARUDA live feed.");
        socketRef.current = ws;
        setIsWsConnected(true);
        triggerNotification("System Alert", "Connected to live FastAPI WebSocket event feed", "low");
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.event === "violation_detected") {
            const mappedVio = mapBackendViolationToUI({
              id: data.violation_id,
              camera_id: data.camera_id,
              location: data.location,
              timestamp: data.timestamp,
              violation_type: data.violation_type,
              confidence: data.confidence,
              severity: data.severity || "medium",
              tier: data.tier,
              action: data.tier === 1 ? "PASSED" : "HUMAN_REVIEW",
              plate_text: data.plate,
              vehicle_class: "unknown",
              annotated_img: data.annotated_image_url || "",
              status: "pending",
              created_at: data.timestamp,
            });

            setViolations(prev => {
              if (prev.some(v => v.id === mappedVio.id)) return prev;
              return [mappedVio, ...prev];
            });

            triggerNotification(
              "Violation Spike",
              `Real-time alert: ${data.violation_type} detected on camera ${data.camera_id} (${data.confidence}%)`,
              "high"
            );
          }
        } catch (e) {
          console.error("Error parsing WebSocket event data:", e);
        }
      };

      ws.onclose = () => {
        console.log("WebSocket connection closed. Retrying...");
        socketRef.current = null;
        setIsWsConnected(false);
        setTimeout(connectWS, 5000); // Reconnect loop
      };

      ws.onerror = () => {
        ws.close();
      };
    };

    connectWS();
    syncWithBackend();

    // Set polling clock for jobs and status updates
    const pollInterval = setInterval(() => {
      syncWithBackend();
    }, 4000);

    return () => {
      clearInterval(pollInterval);
      if (socketRef.current) {
        socketRef.current.close();
      }
    };
  }, [syncWithBackend, isBackendConnected, mapBackendViolationToUI, triggerNotification]);

  // Auth Operations
  const login = async (email: string, password: string): Promise<boolean> => {
    try {
      const res = await fetch(`${API_BASE}/auth/login`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email, password })
      });
      
      const data = await res.json();
      if (res.ok) {
        localStorage.setItem("garuda_token", data.access_token);
        setToken(data.access_token);
        setUser({
          id: data.id || data.user_id || "USR-001",
          name: data.username || data.name,
          email: data.email,
          role: data.role as UserRole,
          status: "Active"
        });
        setRole(data.role as UserRole);
        triggerNotification("System Alert", `Welcome back, ${data.username || data.name}! Session authenticated.`, "low");
        return true;
      } else {
        alert(data.detail || "Invalid login credentials.");
        return false;
      }
    } catch (e) {
      alert("Cannot connect to backend. Please ensure the server is running.");
      return false;
    }
  };

  const register = async (name: string, email: string, password: string, roleSelected: string): Promise<string> => {
    try {
      const res = await fetch(`${API_BASE}/auth/register`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name, email, password, role: roleSelected })
      });
      const data = await res.json();
      if (res.ok) {
        return data.message;
      } else {
        return data.detail || "Registration failed. Verify user options.";
      }
    } catch (e) {
      return "Backend authentication API is offline. Cannot process dynamic signup.";
    }
  };

  const logout = () => {
    localStorage.removeItem("garuda_token");
    setToken(null);
    setUser(null);
    triggerNotification("System Alert", "Successfully logged out of security session.", "low");
  };

  const updateUserRole = async (userId: string, newRole: UserRole): Promise<boolean> => {
    const headers: HeadersInit = token ? { "Authorization": `Bearer ${token}` } : {};
    
    // Optimistic UI state update
    setUsersList(prev => prev.map(u => u.id === userId ? { ...u, role: newRole } : u));
    
    if (isBackendConnected) {
      try {
        const res = await fetch(`${API_BASE}/users/${userId}/role`, {
          method: "PUT",
          headers: {
            "Content-Type": "application/json",
            ...headers
          },
          body: JSON.stringify({ role: newRole })
        });
        if (res.ok) {
          syncWithBackend();
          return true;
        }
      } catch (e) {
        console.error("Error setting role on database", e);
      }
    }
    return true;
  };

  // Action methods with API calls
  const addCamera = useCallback(async (cam: Omit<Camera, "lastHeartbeat">) => {
    const newCam: Camera = { ...cam, lastHeartbeat: new Date().toISOString() };
    setCameras(prev => [...prev, newCam]);
    triggerNotification("System Alert", `New Camera Registered: ${cam.name} (${cam.id})`, "low");

    const headers: HeadersInit = token ? { 
      "Content-Type": "application/json", 
      "Authorization": `Bearer ${token}` 
    } : { "Content-Type": "application/json" };

    try {
      await fetch(`${API_BASE}/cameras`, {
        method: "POST",
        headers,
        body: JSON.stringify({
          id: cam.id,
          location: cam.location,
          stop_line_y: 380,
          description: cam.name,
          rtsp_url: cam.rtspUrl,
          resolution: cam.resolution
        })
      });
    } catch (e) {
      console.log("API offline, saved camera in memory");
    }
  }, [triggerNotification, token]);

  const toggleCameraStatus = useCallback(async (id: string, status: CameraStatus) => {
    setCameras(prev => prev.map(c => c.id === id ? {
      ...c,
      status,
      lastHeartbeat: new Date().toISOString()
    } : c));
    
    triggerNotification(
      status === "Offline" ? "Camera Offline" : "System Alert",
      `Camera ${id} status changed to ${status}`,
      status === "Offline" ? "high" : "low"
    );

    const headers: HeadersInit = token ? { 
      "Content-Type": "application/json", 
      "Authorization": `Bearer ${token}` 
    } : { "Content-Type": "application/json" };

    try {
      await fetch(`${API_BASE}/cameras/${id}/config`, {
        method: "PUT",
        headers,
        body: JSON.stringify({
          description: status === "Active" ? "Active Stream" : "Disabled Stream"
        })
      });
    } catch (e) {
      console.log("API offline, updated status in memory");
    }
  }, [triggerNotification, token]);

  const deleteCamera = useCallback(async (id: string) => {
    setCameras(prev => prev.filter(c => c.id !== id));
    triggerNotification("System Alert", `Camera deleted from registry: ${id}`, "medium");

    const headers: HeadersInit = token ? { "Authorization": `Bearer ${token}` } : {};

    try {
      await fetch(`${API_BASE}/cameras/${id}`, {
        method: "DELETE",
        headers
      });
    } catch (e) {
      console.log("API offline, deleted camera from memory");
    }
  }, [triggerNotification, token]);

  const deleteViolation = useCallback(async (id: string) => {
    setViolations(prev => prev.filter(v => v.id !== id));
    triggerNotification("System Alert", `Violation deleted: ${id}`, "low");

    const headers: HeadersInit = token ? { "Authorization": `Bearer ${token}` } : {};

    try {
      await fetch(`${API_BASE}/violations/${id}`, {
        method: "DELETE",
        headers
      });
    } catch (e) {
      console.log("API offline, deleted violation from memory");
    }
  }, [triggerNotification, token]);

  const sendChallanSms = useCallback(async (id: string) => {
    triggerNotification("System Alert", `Initiating SMS Challan send for violation: ${id}`, "low");

    const headers: HeadersInit = token ? { "Authorization": `Bearer ${token}` } : {};

    try {
      const res = await fetch(`${API_BASE}/violations/${id}/send-sms`, {
        method: "POST",
        headers
      });
      if (res.ok) {
        triggerNotification("System Alert", `SMS Challan sent successfully for violation: ${id}`, "low");
      } else {
        throw new Error(`Server returned status ${res.status}`);
      }
    } catch (e) {
      triggerNotification("System Alert", `Failed to send SMS Challan: ${e instanceof Error ? e.message : String(e)}`, "high");
    }
  }, [triggerNotification, token]);

  const batchDeleteViolations = useCallback(async (ids: string[]) => {
    const idSet = new Set(ids);
    setViolations(prev => prev.filter(v => !idSet.has(v.id)));
    triggerNotification("System Alert", `Batch deleted ${ids.length} violations`, "low");

    const headers: HeadersInit = {
      "Content-Type": "application/json",
      ...(token ? { "Authorization": `Bearer ${token}` } : {})
    };

    try {
      await fetch(`${API_BASE}/violations/batch-delete`, {
        method: "POST",
        headers,
        body: JSON.stringify({ ids })
      });
    } catch (e) {
      console.log("API offline, batch deleted violations from memory");
    }
  }, [triggerNotification, token]);

  const testCameraConnection = useCallback(async (id: string): Promise<boolean> => {
    if (isBackendConnected) {
      try {
        const headers: HeadersInit = token ? { "Authorization": `Bearer ${token}` } : {};
        const res = await fetch(`${API_BASE}/cameras/${id}`, { headers });
        return res.ok;
      } catch {
        return false;
      }
    }
    return new Promise(resolve => {
      setTimeout(() => {
        const cam = cameras.find(c => c.id === id);
        resolve(cam ? cam.status !== "Offline" : false);
      }, 1500);
    });
  }, [cameras, isBackendConnected, token]);

  // Real officer review action — calls POST /api/v1/reviews, the single
  // backend endpoint that correctly handles all three actions (the older
  // /violations/{id}/confirm|reject pair has no "escalate" path, so routing
  // "Escalated" through it would silently reject the violation instead).
  const reviewViolation = useCallback(async (
    id: string,
    action: "Approved" | "Rejected" | "Escalated",
    reviewer: string,
    reason?: string,
    correctedPlateText?: string,
  ) => {
    const newStatus: ViolationStatus =
      action === "Approved" ? "confirmed" : action === "Rejected" ? "rejected" : "pending";

    setViolations(prev => prev.map(v => v.id === id ? {
      ...v,
      status: newStatus,
      officerId: reviewer,
      plateNumber: correctedPlateText ? correctedPlateText.toUpperCase() : v.plateNumber,
    } : v));

    triggerNotification("System Alert", `Violation ${id} was ${action.toLowerCase()} by ${reviewer}.`, "low");

    const headers: HeadersInit = token ? {
      "Content-Type": "application/json",
      "Authorization": `Bearer ${token}`
    } : { "Content-Type": "application/json" };

    try {
      await fetch(`${API_BASE}/reviews`, {
        method: "POST",
        headers,
        body: JSON.stringify({
          violation_id: id,
          action,
          reviewer,
          reason: reason || "",
          corrected_plate_text: correctedPlateText || undefined,
        })
      });
    } catch (e) {
      console.log("API offline, updated review status in memory");
    }
  }, [triggerNotification, token]);

  // Approve/reject ONE violation inside a clubbed multi-violation record
  // (same image, different finding) without resolving the whole row — the
  // row's overall status only flips once every item in it has a decision.
  const reviewViolationItem = useCallback(async (
    id: string,
    itemIndex: number,
    action: "Approved" | "Rejected" | "Escalated",
    reviewer: string,
    reason?: string,
  ) => {
    const itemStatus: "confirmed" | "rejected" | "pending" =
      action === "Approved" ? "confirmed" : action === "Rejected" ? "rejected" : "pending";

    setViolations(prev => prev.map(v => {
      if (v.id !== id || !v.violationItems) return v;
      const items = v.violationItems.map((it, i) =>
        i === itemIndex ? { ...it, reviewStatus: itemStatus, escalated: action === "Escalated" } : it
      );
      const statuses = items.map(it => it.reviewStatus);
      const rowStatus: ViolationStatus = statuses.some(s => s === "pending")
        ? "pending"
        : statuses.some(s => s === "confirmed")
          ? "confirmed"
          : "rejected";
      return { ...v, violationItems: items, status: rowStatus, officerId: reviewer };
    }));

    triggerNotification("System Alert", `Item #${itemIndex + 1} of ${id} was ${action.toLowerCase()} by ${reviewer}.`, "low");

    const headers: HeadersInit = token ? {
      "Content-Type": "application/json",
      "Authorization": `Bearer ${token}`
    } : { "Content-Type": "application/json" };

    try {
      await fetch(`${API_BASE}/reviews`, {
        method: "POST",
        headers,
        body: JSON.stringify({
          violation_id: id,
          action,
          reviewer,
          reason: reason || "",
          item_index: itemIndex,
        })
      });
    } catch (e) {
      console.log("API offline, updated review status in memory");
    }
  }, [triggerNotification, token]);

  // Fetch the full ViolationDetailResponse (includes json_record) for one
  // violation — the list/WS-synced `violations` array intentionally only
  // carries summary fields, so the Review Queue calls this to get real
  // per-plate OCR detections, driver-state alerts, and processing info.
  const fetchViolationDetail = useCallback(async (id: string): Promise<Violation | null> => {
    try {
      const headers: HeadersInit = token ? { "Authorization": `Bearer ${token}` } : {};
      const res = await fetch(`${API_BASE}/violations/${id}`, { headers });
      if (!res.ok) return null;
      const data = await res.json();
      const detailed = mapBackendViolationToUI(data);
      setViolations(prev => prev.map(v => v.id === id ? { ...v, ...detailed } : v));
      return detailed;
    } catch (e) {
      console.error("Error fetching violation detail:", e);
      return null;
    }
  }, [token, mapBackendViolationToUI]);

  const clearNotifications = useCallback(() => {
    setNotifications([]);
  }, []);

  const markNotificationRead = useCallback((id: string) => {
    setNotifications(prev => prev.map(n => n.id === id ? { ...n, read: true } : n));
  }, []);

  // Submit Batch Upload job — uses /jobs/upload (multipart) when file is present
  const submitUploadJob = useCallback(async (name: string, type: "Image" | "Video", file?: File) => {
    const jobId = `JOB-${Date.now()}`;
    const newJob: ProcessingJob = {
      id: jobId,
      name,
      sourceType: type,
      uploadTime: new Date().toISOString(),
      duration: 0,
      framesProcessed: 0,
      violationsFound: 0,
      status: "Queued",
      progress: 0
    };
    
    setJobs(prev => [newJob, ...prev]);
    triggerNotification("System Alert", `Inference Job queued for ${name}`, "low");

    if (isBackendConnected) {
      try {
        if (file) {
          // Real file upload — use multipart form data
          const formData = new FormData();
          formData.append("name", name);
          formData.append("source_type", type);
          formData.append("file", file, file.name);

          const headers: HeadersInit = token ? { "Authorization": `Bearer ${token}` } : {};
          await fetch(`${API_BASE}/jobs/upload`, {
            method: "POST",
            headers,  // Do NOT set Content-Type for FormData — browser sets it with boundary
            body: formData
          });
        } else {
          // No file — JSON create
          const headers: HeadersInit = token ? { 
            "Content-Type": "application/json", 
            "Authorization": `Bearer ${token}` 
          } : { "Content-Type": "application/json" };
          await fetch(`${API_BASE}/jobs`, {
            method: "POST",
            headers,
            body: JSON.stringify({ name, source_type: type })
          });
        }
      } catch (e) {
        console.log("API offline, queuing locally");
      }
    }
  }, [triggerNotification, isBackendConnected, token]);

  // When the backend is offline, we let the syncWithBackend REST calls retrieve
  // the exact database status. We do NOT run a local client-side interval timer
  // that forces queued/processing jobs to Failed status. This is to avoid
  // race conditions on temporary WebSocket drops.

  // Periodic heartbeat for active cameras (only when backend offline)
  useEffect(() => {
    if (isBackendConnected) return;
    const hbTimer = setInterval(() => {
      setCameras(prev => prev.map(c => c.status === "Active" ? {
        ...c,
        lastHeartbeat: new Date().toISOString()
      } : c));
    }, 10000);
    return () => clearInterval(hbTimer);
  }, [isBackendConnected]);

  return (
    <PlatformContext.Provider value={{
      token,
      user,
      isAuthenticated: !!token,
      authLoading,
      login,
      register,
      logout,
      usersList,
      auditLogs,
      updateUserRole,

      role,
      setRole,
      cameras,
      violations,
      jobs,
      notifications,
      addCamera,
      toggleCameraStatus,
      testCameraConnection,
      deleteCamera,
      deleteViolation,
      batchDeleteViolations,
      sendChallanSms,
      reviewViolation,
      reviewViolationItem,
      submitUploadJob,
      fetchViolationDetail,
      clearNotifications,
      markNotificationRead,
      isSimulating,
      setIsSimulating,
      simulationInterval,
      setSimulationInterval,
      isBackendConnected,
      isWsConnected
    }}>
      {children}
    </PlatformContext.Provider>
  );
};

export const usePlatform = () => {
  const context = useContext(PlatformContext);
  if (!context) {
    throw new Error("usePlatform must be used within a PlatformProvider");
  }
  return context;
};
