"use client";

import React, { createContext, useContext, useState, useEffect, useCallback, useRef } from "react";

// Definitions
export type CameraStatus = "Active" | "Offline" | "Disabled";
export type ViolationStatus = "Detected" | "Under Review" | "Approved" | "Rejected";
export type JobStatus = "Queued" | "Processing" | "Completed" | "Failed";
export type UserRole = "Admin" | "Supervisor" | "Reviewer" | "Operator";

export interface Camera {
  id: string;
  name: string;
  rtspUrl: string;
  location: string;
  status: CameraStatus;
  lastHeartbeat: string;
  fps: number;
  resolution: string;
}

export interface BoundingBox {
  x: number;
  y: number;
  w: number;
  h: number;
}

export interface EvidenceFrame {
  timestamp: string;
  vehicleBox: BoundingBox;
  plateBox: BoundingBox;
  speed?: number;
  lightColor?: "Red" | "Yellow" | "Green";
  vehicleSvgType: "sedan" | "suv" | "motorcycle" | "truck";
  color: string;
}

export interface Violation {
  id: string;
  type: string; // "Red Light", "Speeding", "Wrong Way", "Seatbelt", "No Helmet"
  timestamp: string;
  location: string;
  cameraId: string;
  vehicleType: string;
  plateNumber: string;
  confidenceScore: number; // 0-100
  status: ViolationStatus;
  reviewer?: string;
  reviewedAt?: string;
  actionReason?: string;
  
  // Custom interactive frames
  beforeFrame: EvidenceFrame;
  violationFrame: EvidenceFrame;
  afterFrame: EvidenceFrame;
}

export interface ProcessingJob {
  id: string;
  name: string;
  sourceType: "Image" | "Video";
  uploadTime: string;
  duration: number; // seconds
  framesProcessed: number;
  violationsFound: number;
  status: JobStatus;
  progress: number; // 0 to 100
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
  
  addViolation: (violation: Violation) => void;
  updateViolationStatus: (id: string, status: ViolationStatus, reviewer: string, reason?: string) => void;
  
  submitUploadJob: (name: string, type: "Image" | "Video", file?: File) => void;
  clearNotifications: () => void;
  markNotificationRead: (id: string) => void;
  
  // Simulation Controller state
  isSimulating: boolean;
  setIsSimulating: (sim: boolean) => void;
  simulationInterval: number; // ms
  setSimulationInterval: (interval: number) => void;
  isBackendConnected: boolean;
}

const PlatformContext = createContext<PlatformContextType | undefined>(undefined);

// API Endpoints
let API_BASE = "http://localhost:8000/api/v1";
let WS_BASE = "ws://localhost:8000";

if (typeof window !== "undefined") {
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
    
    const beforeFrame = record.beforeFrame || {
      timestamp: new Date(new Date(v.timestamp).getTime() - 1000).toISOString(),
      vehicleBox: { x: 50, y: 120, w: 90, h: 55 },
      plateBox: { x: 90, y: 155, w: 20, h: 10 },
      lightColor: v.violation_type === "Red Light" ? "Green" : undefined,
      speed: v.violation_type === "Speeding" ? 54 : undefined,
      vehicleSvgType: v.vehicle_class?.toLowerCase().includes("motorcycle") ? "motorcycle" : "sedan",
      color: "#3B82F6"
    };

    const violationFrame = record.violationFrame || {
      timestamp: v.timestamp,
      vehicleBox: { x: 120, y: 100, w: 95, h: 58 },
      plateBox: { x: 165, y: 138, w: 22, h: 11 },
      lightColor: v.violation_type === "Red Light" ? "Red" : undefined,
      speed: v.violation_type === "Speeding" ? 82 : undefined,
      vehicleSvgType: v.vehicle_class?.toLowerCase().includes("motorcycle") ? "motorcycle" : "sedan",
      color: "#3B82F6"
    };

    const afterFrame = record.afterFrame || {
      timestamp: new Date(new Date(v.timestamp).getTime() + 1000).toISOString(),
      vehicleBox: { x: 220, y: 80, w: 90, h: 55 },
      plateBox: { x: 260, y: 115, w: 20, h: 10 },
      lightColor: v.violation_type === "Red Light" ? "Red" : undefined,
      speed: v.violation_type === "Speeding" ? 84 : undefined,
      vehicleSvgType: v.vehicle_class?.toLowerCase().includes("motorcycle") ? "motorcycle" : "sedan",
      color: "#3B82F6"
    };

    let status: ViolationStatus = "Detected";
    if (v.status === "confirmed") status = "Approved";
    else if (v.status === "rejected") status = "Rejected";
    else if (v.status === "pending") status = "Under Review";

    return {
      id: v.id,
      type: v.violation_type,
      timestamp: v.timestamp,
      location: v.location,
      cameraId: v.camera_id,
      vehicleType: v.vehicle_class || "Passenger Vehicle",
      plateNumber: v.plate_text || "UNCLEAR",
      confidenceScore: confidenceScore,
      status: status,
      reviewer: v.officer_id || undefined,
      reviewedAt: v.created_at || undefined,
      beforeFrame,
      violationFrame,
      afterFrame
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
            rtspUrl: `rtsp://10.200.41.10/live/${c.id.toLowerCase()}`,
            location: c.location,
            status: c.status === "active" ? "Active" : "Offline",
            lastHeartbeat: c.last_seen || new Date().toISOString(),
            fps: c.status === "active" ? 30 : 0,
            resolution: "1920x1080"
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
          progress: j.progress
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
        setIsBackendConnected(true);
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
              action: "HUMAN_REVIEW",
              plate_text: data.plate,
              vehicle_class: "Passenger Vehicle",
              status: "pending"
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
        setIsBackendConnected(false);
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
          description: cam.name
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
      fps: status === "Active" ? 30 : 0,
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

  const addViolation = useCallback((vio: Violation) => {
    setViolations(prev => [vio, ...prev]);
  }, []);

  const updateViolationStatus = useCallback(async (id: string, status: ViolationStatus, reviewer: string, reason?: string) => {
    setViolations(prev => prev.map(v => v.id === id ? {
      ...v,
      status,
      reviewer,
      reviewedAt: new Date().toISOString(),
      actionReason: reason
    } : v));
    
    triggerNotification("System Alert", `Violation ${id} was ${status.toLowerCase()} by ${reviewer}.`, "low");

    const headers: HeadersInit = token ? { 
      "Content-Type": "application/json", 
      "Authorization": `Bearer ${token}` 
    } : { "Content-Type": "application/json" };

    try {
      const endpoint = status === "Approved" ? "confirm" : "reject";
      await fetch(`${API_BASE}/violations/${id}/${endpoint}`, {
        method: "POST",
        headers,
        body: JSON.stringify({
          officer_id: reviewer,
          notes: reason || ""
        })
      });
    } catch (e) {
      console.log("API offline, updated review status in memory");
    }
  }, [triggerNotification, token]);

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

  // Client-side fallback progress simulator (runs ONLY if backend is completely offline)
  useEffect(() => {
    if (isBackendConnected) return;

    const jobTimer = setInterval(() => {
      setJobs(prevJobs => {
        let updated = false;
        const nextJobs = prevJobs.map(job => {
          if (job.status === "Queued") {
            updated = true;
            return { ...job, status: "Processing" as JobStatus, progress: 10 };
          }
          if (job.status === "Processing") {
            updated = true;
            const step = job.sourceType === "Video" ? 15 : 45;
            const nextProgress = Math.min(job.progress + step, 100);
            const frames = job.sourceType === "Video" ? Math.floor((nextProgress / 100) * 240) : 1;
            const duration = Math.round((Date.now() - new Date(job.uploadTime).getTime()) / 1000);
            
            if (nextProgress >= 100) {
              const count = Math.random() > 0.4 ? (job.sourceType === "Video" ? 2 : 1) : 0;
              
              setTimeout(() => {
                const camList = cameras.filter(c => c.status === "Active");
                const targetCam = camList[Math.floor(Math.random() * camList.length)] || initialCameras[0];
                
                for (let i = 0; i < count; i++) {
                  const vType = ["Speeding", "Red Light", "Seatbelt", "Wrong Way"][Math.floor(Math.random() * 4)];
                  const pNum = generatePlate();
                  const vId = `VIO-JOB-${Date.now()}-${i}`;
                  
                  const colors = ["#EF4444", "#3B82F6", "#10B981", "#F59E0B", "#8B5CF6"];
                  const selectedColor = colors[Math.floor(Math.random() * colors.length)];
                  const svgTypes: EvidenceFrame["vehicleSvgType"][] = ["sedan", "suv", "motorcycle", "truck"];
                  const selectedSvg = svgTypes[Math.floor(Math.random() * svgTypes.length)];
                  const vTypesStr = selectedSvg === "motorcycle" ? "Motorcycle" : (selectedSvg === "truck" ? "Heavy Truck" : "Passenger Vehicle");

                  const newVio: Violation = {
                    id: vId,
                    type: vType,
                    timestamp: new Date().toISOString(),
                    location: targetCam.location,
                    cameraId: targetCam.id,
                    vehicleType: vTypesStr,
                    plateNumber: pNum,
                    confidenceScore: parseFloat((85 + Math.random() * 14).toFixed(1)),
                    status: "Detected",
                    beforeFrame: { timestamp: new Date(Date.now() - 1000).toISOString(), vehicleBox: { x: 50, y: 120, w: 90, h: 55 }, plateBox: { x: 90, y: 155, w: 20, h: 10 }, lightColor: vType === "Red Light" ? "Green" : undefined, speed: vType === "Speeding" ? 58 : undefined, vehicleSvgType: selectedSvg, color: selectedColor },
                    violationFrame: { timestamp: new Date().toISOString(), vehicleBox: { x: 120, y: 100, w: 95, h: 58 }, plateBox: { x: 165, y: 138, w: 22, h: 11 }, lightColor: vType === "Red Light" ? "Red" : undefined, speed: vType === "Speeding" ? 82 : undefined, vehicleSvgType: selectedSvg, color: selectedColor },
                    afterFrame: { timestamp: new Date(Date.now() + 1000).toISOString(), vehicleBox: { x: 220, y: 80, w: 90, h: 55 }, plateBox: { x: 260, y: 115, w: 20, h: 10 }, lightColor: vType === "Red Light" ? "Red" : undefined, speed: vType === "Speeding" ? 83 : undefined, vehicleSvgType: selectedSvg, color: selectedColor }
                  };
                  addViolation(newVio);
                }
                
                triggerNotification(
                  "System Alert",
                  `Simulated local Ingestion Complete: Job found ${count} infractions`,
                  count > 0 ? "medium" : "low"
                );
              }, 100);

              return {
                ...job,
                progress: 100,
                status: "Completed" as JobStatus,
                framesProcessed: frames,
                duration,
                violationsFound: count
              };
            }
            
            return {
              ...job,
              progress: nextProgress,
              framesProcessed: frames,
              duration
            };
          }
          return job;
        });
        return updated ? nextJobs : prevJobs;
      });
    }, 1000);
    return () => clearInterval(jobTimer);
  }, [cameras, isBackendConnected, addViolation, triggerNotification]);

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
      addViolation,
      updateViolationStatus,
      submitUploadJob,
      clearNotifications,
      markNotificationRead,
      isSimulating,
      setIsSimulating,
      simulationInterval,
      setSimulationInterval,
      isBackendConnected
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
