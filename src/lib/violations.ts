// Single source of truth for the violation taxonomy actually produced by the
// ML pipeline (ml/pipeline/violation_classifier.py — ViolationType enum,
// VIOLATION_SEVERITY, FINE_AMOUNTS_INR) and the real status vocabulary the
// backend persists (backend/core/database.py — ViolationModel.status,
// VIOLATION_TYPE_DISPLAY_MAP). Do not invent types/statuses beyond this list —
// anything not here cannot actually be produced by the system.

export const VIOLATION_TYPES = [
  "No Helmet",
  "Seatbelt",
  "Triple Riding",
  "Wrong Way",
  "Stop Line",
  "Red Light",
  "Illegal Parking",
  "Phone Use",
  "Drowsy",
] as const;

export type ViolationTypeLabel = (typeof VIOLATION_TYPES)[number];

export const VIOLATION_SEVERITY_BY_TYPE: Record<string, "low" | "medium" | "high" | "critical"> = {
  "No Helmet": "high",
  "Seatbelt": "medium",
  "Triple Riding": "high",
  "Wrong Way": "critical",
  "Stop Line": "medium",
  "Red Light": "high",
  "Illegal Parking": "low",
  "Phone Use": "high",
  "Drowsy": "critical",
};

export const VIOLATION_FINE_INR: Record<string, number> = {
  "No Helmet": 1000,
  "Seatbelt": 1000,
  "Triple Riding": 2000,
  "Wrong Way": 5000,
  "Stop Line": 500,
  "Red Light": 1000,
  "Illegal Parking": 500,
  "Phone Use": 5000,
  "Drowsy": 2000,
};

// Real backend status values (backend/api/reviews.py + database.py). There is
// no "Detected" / "Under Review" / "Escalated" status distinct from these —
// escalation logs an audit entry and assigns officer_id but the violation
// stays "pending" until actually confirmed or rejected.
export type ViolationStatus = "pending" | "auto_challan" | "confirmed" | "rejected";

export const STATUS_LABELS: Record<ViolationStatus, string> = {
  pending: "Pending Review",
  auto_challan: "Auto-Cleared",
  confirmed: "Confirmed",
  rejected: "Rejected",
};

export const STATUS_BADGE_CLASS: Record<ViolationStatus, string> = {
  pending: "review",
  auto_challan: "approved",
  confirmed: "approved",
  rejected: "rejected",
};

export const SEVERITY_COLOR: Record<string, string> = {
  low: "#64748b",
  medium: "#eab308",
  high: "#f97316",
  critical: "#ef4444",
};

// Backend stores confidence as a 0-1 fraction almost everywhere, but a couple
// of code paths (live WS broadcast payloads) send 0-100. Normalize once here.
export function normalizeConfidencePct(raw: number): number {
  if (raw == null || isNaN(raw)) return 0;
  const pct = raw > 1 ? raw : raw * 100;
  return Math.round(pct * 10) / 10;
}

export interface PlateDetection {
  plateText: string;
  confidence: number; // 0-100
  vehicleClass: string;
  bbox: number[];
  ocrEngine: string;
  state: string;
  isValid: boolean;
}

export interface DriverAlert {
  alertType: string;
  severity: string;
  action: string;
  confidence: number; // 0-1
  trackId: number | null;
  metadata: Record<string, unknown>;
}

export interface ProcessingInfo {
  inferenceDevice: string;
  inferenceTimeMs: number;
  model: string;
  ocrEngine: string;
  vehiclesDetected: number;
  personsDetected: number;
}
