// Types + API client for the Evidence pipeline page. These interfaces mirror
// the RAW backend pipeline record (backend/api/jobs.py::_classify_and_package)
// field-for-field (snake_case, unmapped) — this page exists specifically to
// show the real ml/pipeline/* execution, so it reads the backend's own
// uncollapsed JSON directly via GET /jobs/{job_id}/result rather than going
// through the normalized camelCase Violation shape used elsewhere.

export interface RawViolation {
  type: string; // one of VIOLATION_TYPES (src/lib/violations.ts) — backend already emits the display label
  confidence: number; // 0-1
  severity: string;
  fine_amount_inr: number;
  bbox: number[];
  metadata: Record<string, unknown>;
  plate_text?: string;
  track_id?: number | null;
}

export interface RawVehicleInfo {
  vehicle_class: string;
  license_plate: string;
  plate_confidence: number;
  plate_valid: boolean;
  plate_state?: string;
  track_id?: number | null;
}

export interface RawPlateDetection {
  plate_text: string;
  confidence: number;
  vehicle_class: string;
  bbox: number[];
  ocr_engine: string;
  state: string;
  is_valid: boolean;
  track_id?: number | null;
}

export interface RawDriverAlert {
  alert_type: string; // "DROWSY_DRIVER" | "YAWNING_DETECTED" | "PHONE_USE_WHILE_DRIVING"
  severity: string;
  action: string;
  confidence: number;
  track_id: number | null;
  metadata: Record<string, unknown>;
}

export interface RawProcessingInfo {
  inference_device: string;
  inference_time_ms: number;
  model: string;
  ocr_engine: string;
  vehicles_detected: number;
  persons_detected: number;
  camera_calibrated: boolean;
}

// One per uploaded image, or one per sampled video frame.
export interface PipelineRecord {
  violation_id: string;
  tier: number; // 1 = auto-challan, 2 = human review, 3 = logged/discarded
  action: string;
  timestamp: string;
  camera: { id: string; location: string; coordinates: Record<string, number> };
  vehicle: RawVehicleInfo;
  violations: RawViolation[];
  all_plates_detected: RawPlateDetection[];
  processing: RawProcessingInfo;
  driver_state: { alerts: RawDriverAlert[]; total_alerts: number };
  evidence: { annotated_image: string; raw_frame: string; demo_image: string };
}

export interface JobSummary {
  id: string;
  name: string;
  source_type: string; // "Image" | "Video" | "Batch"
  progress: number;
  status: string;
  duration: number;
  frames_processed: number;
  violations_found: number;
  upload_time: string;
  camera_id: string | null;
}

export interface JobResult {
  job: JobSummary;
  records: PipelineRecord[];
  video_url?: string | null;
  demo_video_url?: string | null;
  rendering_eta?: number | null;
}

// ---------------------------------------------------------------------------
// API base — same host-derived construction used in PlatformContext.tsx
// ---------------------------------------------------------------------------

export function getApiBase(): string {
  if (typeof window === "undefined") return "http://localhost:8000/api/v1";
  const protocol = window.location.protocol === "https:" ? "https" : "http";
  return `${protocol}://${window.location.hostname}:8000/api/v1`;
}

function authHeaders(token?: string | null): HeadersInit {
  return token ? { Authorization: `Bearer ${token}` } : {};
}

// ---------------------------------------------------------------------------
// Reads
// ---------------------------------------------------------------------------

export async function fetchJobResult(jobId: string, token?: string | null): Promise<JobResult> {
  const res = await fetch(`${getApiBase()}/jobs/${jobId}/result`, { headers: authHeaders(token) });
  if (!res.ok) throw new Error(`Failed to fetch job result (${res.status})`);
  return res.json();
}

export async function fetchTestGalleryList(): Promise<string[]> {
  const res = await fetch(`${getApiBase()}/evidence/test-gallery/list`);
  if (!res.ok) return [];
  return res.json();
}

export function testGalleryImageUrl(filename: string): string {
  return `${getApiBase()}/evidence/test-gallery/image/${encodeURIComponent(filename)}?t=${Date.now()}`;
}

export function evidenceFileUrl(path: string): string {
  // path is already like "/evidence/annotated/{job_id}/{id}.jpg"
  const base = getApiBase().replace(/\/api\/v1$/, "");
  return `${base}${path}`;
}

// ---------------------------------------------------------------------------
// Uploads
// ---------------------------------------------------------------------------

export async function uploadSingle(opts: {
  name: string;
  sourceType: "Image" | "Video";
  file: File;
  cameraId?: string | null;
  stopLineY?: number | null;
  parkingTimer?: number | null;
  token?: string | null;
}): Promise<JobSummary> {
  const formData = new FormData();
  formData.append("name", opts.name);
  formData.append("source_type", opts.sourceType);
  if (opts.cameraId) formData.append("camera_id", opts.cameraId);
  if (opts.stopLineY !== undefined && opts.stopLineY !== null) formData.append("stop_line_y", String(opts.stopLineY));
  if (opts.parkingTimer !== undefined && opts.parkingTimer !== null) formData.append("parking_timer", String(opts.parkingTimer));
  formData.append("file", opts.file, opts.file.name);

  const res = await fetch(`${getApiBase()}/jobs/upload`, {
    method: "POST",
    headers: authHeaders(opts.token),
    body: formData,
  });
  if (!res.ok) throw new Error(`Upload failed (${res.status})`);
  return res.json();
}

export async function uploadBatch(opts: {
  name: string;
  files: File[];
  cameraId?: string | null;
  token?: string | null;
}): Promise<JobSummary> {
  const formData = new FormData();
  formData.append("name", opts.name);
  if (opts.cameraId) formData.append("camera_id", opts.cameraId);
  for (const f of opts.files) formData.append("files", f, f.name);

  const res = await fetch(`${getApiBase()}/jobs/upload-batch`, {
    method: "POST",
    headers: authHeaders(opts.token),
    body: formData,
  });
  if (!res.ok) throw new Error(`Batch upload failed (${res.status})`);
  return res.json();
}

// ---------------------------------------------------------------------------
// Deletes
// ---------------------------------------------------------------------------

export async function deleteJob(jobId: string, token?: string | null): Promise<void> {
  const res = await fetch(`${getApiBase()}/jobs/${jobId}`, {
    method: "DELETE",
    headers: authHeaders(token),
  });
  if (!res.ok && res.status !== 404) throw new Error(`Delete failed (${res.status})`);
}

export async function clearAllJobs(token?: string | null): Promise<void> {
  const res = await fetch(`${getApiBase()}/jobs`, {
    method: "DELETE",
    headers: authHeaders(token),
  });
  if (!res.ok) throw new Error(`Clear all failed (${res.status})`);
}

// ---------------------------------------------------------------------------
// Real-pipeline stage aggregation — every number here is derived straight
// from records[], nothing fabricated. This is what backs the step-by-step
// breakdown view for a clubbed (1 or many) result.
// ---------------------------------------------------------------------------

export interface ViolationTypeAggregate {
  type: string;
  count: number;
  methods: Record<string, number>; // metadata.method -> count, e.g. {"calibrated_zone": 1, "static_position": 2}
}

export interface StageAggregate {
  itemCount: number;
  modelsUsed: string[];
  ocrEngine: string;
  totalVehicles: number;
  totalPersons: number;
  tracked: boolean; // true if any record's violation metadata indicates track-based detection
  calibratedCount: number;
  violationsByType: ViolationTypeAggregate[];
  driverAlertsByType: Record<string, number>;
  platesRead: { text: string; confidence: number; valid: boolean }[];
  tierCounts: { tier1: number; tier2: number; tier3: number };
  totalInferenceMs: number;
  evidenceFolder: string | null;
  fileCount: number;
}

// Only these 4 violation types have a track-based vs. static-fallback split
// in ml/pipeline/violation_classifier.py (check_wrong_side/check_stop_line/
// check_red_light/check_illegal_parking vs. their _check_*_static
// counterparts). Helmet/seatbelt/triple/phone's metadata.method values
// (e.g. "aicity_9cls", "yolov11s_classifier") describe which model/fallback
// fired, not tracking — they must not influence the Tracker stage.
const TRACKER_RELEVANT_TYPES = ["Wrong Way", "Stop Line", "Red Light", "Illegal Parking"];

// Methods emitted by the static/single-frame fallbacks for the 4 types
// above. Track-based detections never set metadata.method at all.
const STATIC_METHOD_MARKERS = [
  "static_position",
  "calibrated_zone",
  "static_zone_check",
  "static_lane_heuristic",
];

export function aggregateRecords(records: PipelineRecord[]): StageAggregate {
  const agg: StageAggregate = {
    itemCount: records.length,
    modelsUsed: [],
    ocrEngine: "",
    totalVehicles: 0,
    totalPersons: 0,
    tracked: false,
    calibratedCount: 0,
    violationsByType: [],
    driverAlertsByType: {},
    platesRead: [],
    tierCounts: { tier1: 0, tier2: 0, tier3: 0 },
    totalInferenceMs: 0,
    evidenceFolder: null,
    fileCount: 0,
  };

  const modelSet = new Set<string>();
  const ocrSet = new Set<string>();
  const typeMap = new Map<string, ViolationTypeAggregate>();

  // Determine if this is a tracked video job by checking for valid track IDs
  const isVideo = records.some(r => 
    (r.vehicle?.track_id !== undefined && r.vehicle.track_id !== null) ||
    (r.violations || []).some(v => v.track_id !== undefined && v.track_id !== null)
  );

  if (isVideo) {
    const uniqueTrackIds = new Set<string | number>();
    let maxPersonsInFrame = 0;
    
    for (const r of records) {
      if (r.vehicle?.track_id !== undefined && r.vehicle.track_id !== null) {
        uniqueTrackIds.add(r.vehicle.track_id);
      }
      for (const v of r.violations || []) {
        const tId = v.track_id !== undefined && v.track_id !== null ? v.track_id : r.vehicle?.track_id;
        if (tId !== undefined && tId !== null) {
          uniqueTrackIds.add(tId);
        }
      }
      for (const p of r.all_plates_detected || []) {
        if (p.track_id !== undefined && p.track_id !== null) {
          uniqueTrackIds.add(p.track_id);
        }
      }
      
      const pCount = r.processing?.persons_detected ?? 0;
      if (pCount > maxPersonsInFrame) {
        maxPersonsInFrame = pCount;
      }
    }
    
    agg.totalVehicles = uniqueTrackIds.size;
    agg.totalPersons = maxPersonsInFrame;
  } else {
    for (const r of records) {
      agg.totalVehicles += r.processing?.vehicles_detected ?? 0;
      agg.totalPersons += r.processing?.persons_detected ?? 0;
    }
  }

  // Set to map unique violations: (trackId, type) for video, or every violation for static/batch
  const uniqueViolationsSet = new Set<string>();

  for (const r of records) {
    agg.totalInferenceMs += r.processing?.inference_time_ms ?? 0;
    if (r.processing?.model) modelSet.add(r.processing.model);
    if (r.processing?.ocr_engine) ocrSet.add(r.processing.ocr_engine);
    if (r.processing?.camera_calibrated) agg.calibratedCount += 1;

    if (r.tier === 1) agg.tierCounts.tier1 += 1;
    else if (r.tier === 2) agg.tierCounts.tier2 += 1;
    else agg.tierCounts.tier3 += 1;

    for (const v of r.violations || []) {
      const method = String((v.metadata?.method as string) || "");
      if (TRACKER_RELEVANT_TYPES.includes(v.type) && !STATIC_METHOD_MARKERS.includes(method)) {
        agg.tracked = true;
      }

      const tId = v.track_id !== undefined && v.track_id !== null ? v.track_id : r.vehicle?.track_id;
      const trackKey = tId !== undefined && tId !== null ? String(tId) : `frame_${r.violation_id}`;
      const vioKey = `${trackKey}_${v.type}`;

      if (isVideo) {
        // Video: count violation type only once per vehicle track ID
        if (!uniqueViolationsSet.has(vioKey)) {
          uniqueViolationsSet.add(vioKey);
          
          if (!typeMap.has(v.type)) typeMap.set(v.type, { type: v.type, count: 0, methods: {} });
          const entry = typeMap.get(v.type)!;
          entry.count += 1;
          const key = method || "model";
          entry.methods[key] = (entry.methods[key] || 0) + 1;
        }
      } else {
        // Static/Batch: count every occurrence
        if (!typeMap.has(v.type)) typeMap.set(v.type, { type: v.type, count: 0, methods: {} });
        const entry = typeMap.get(v.type)!;
        entry.count += 1;
        const key = method || "model";
        entry.methods[key] = (entry.methods[key] || 0) + 1;
      }
    }

    for (const a of r.driver_state?.alerts || []) {
      agg.driverAlertsByType[a.alert_type] = (agg.driverAlertsByType[a.alert_type] || 0) + 1;
    }

    if (r.vehicle?.license_plate && r.vehicle.license_plate !== "UNCLEAR" && r.vehicle.license_plate !== "PLATE-UNREAD") {
      agg.platesRead.push({
        text: r.vehicle.license_plate,
        confidence: r.vehicle.plate_confidence,
        valid: r.vehicle.plate_valid,
      });
    }

    agg.fileCount += [r.evidence?.raw_frame, r.evidence?.annotated_image, r.evidence?.demo_image]
      .filter(Boolean).length;
    if (!agg.evidenceFolder && r.evidence?.annotated_image) {
      agg.evidenceFolder = r.evidence.annotated_image.split("/").slice(0, -1).join("/");
    }
  }

  agg.modelsUsed = Array.from(modelSet);
  agg.ocrEngine = Array.from(ocrSet).join(", ");
  agg.violationsByType = Array.from(typeMap.values());
  return agg;
}
