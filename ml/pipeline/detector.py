"""
GARUDA ML Pipeline — Vehicle & Person Detector
================================================
Model  : YOLOv8m (Ultralytics) — auto-downloads on first run
Formats: PyTorch (.pt), TensorRT (.engine), TFLite (.tflite)
Purpose: Detect and localize all traffic participants in a frame

Detected COCO classes used:
  0=person, 1=bicycle, 2=car, 3=motorcycle, 5=bus, 7=truck
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Class mappings
# ---------------------------------------------------------------------------

VEHICLE_CLASS_IDS = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck", 1: "bicycle"}
PERSON_CLASS_ID = 0
PHONE_CLASS_ID  = 67   # COCO "cell phone"
ALL_TRAFFIC_CLASS_IDS = {**VEHICLE_CLASS_IDS, PERSON_CLASS_ID: "person", PHONE_CLASS_ID: "cell_phone"}

TWO_WHEELER_IDS = {1, 3}   # bicycle + motorcycle
FOUR_WHEELER_IDS = {2, 5, 7}  # car + bus + truck


# ---------------------------------------------------------------------------
# Detection data class
# ---------------------------------------------------------------------------

class Detection:
    """
    Single bounding-box detection result.

    Attributes
    ----------
    bbox        : [x1, y1, x2, y2] in pixel coordinates
    confidence  : model confidence score 0..1
    class_id    : COCO class integer
    class_name  : human-readable class label
    track_id    : ByteTrack persistent ID (None for single-frame detect)
    """

    __slots__ = ("bbox", "confidence", "class_id", "class_name", "track_id", "plate_text", "plate_conf")

    def __init__(
        self,
        bbox: List[float],
        confidence: float,
        class_id: int,
        track_id: Optional[int] = None,
    ) -> None:
        self.bbox = bbox
        self.confidence = float(confidence)
        self.class_id = int(class_id)
        self.class_name = ALL_TRAFFIC_CLASS_IDS.get(self.class_id, f"cls_{class_id}")
        self.track_id = track_id
        self.plate_text = None
        self.plate_conf = 0.0

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    @property
    def center(self) -> Tuple[float, float]:
        return (self.bbox[0] + self.bbox[2]) / 2, (self.bbox[1] + self.bbox[3]) / 2

    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def is_vehicle(self) -> bool:
        return self.class_id in VEHICLE_CLASS_IDS

    @property
    def is_person(self) -> bool:
        return self.class_id == PERSON_CLASS_ID

    @property
    def is_two_wheeler(self) -> bool:
        return self.class_id in TWO_WHEELER_IDS

    @property
    def is_four_wheeler(self) -> bool:
        return self.class_id in FOUR_WHEELER_IDS

    def iou(self, other: "Detection") -> float:
        """Intersection-over-Union with another detection"""
        ix1 = max(self.bbox[0], other.bbox[0])
        iy1 = max(self.bbox[1], other.bbox[1])
        ix2 = min(self.bbox[2], other.bbox[2])
        iy2 = min(self.bbox[3], other.bbox[3])
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0

    def contains_point(self, x: float, y: float) -> bool:
        return self.bbox[0] <= x <= self.bbox[2] and self.bbox[1] <= y <= self.bbox[3]

    def to_dict(self) -> dict:
        res = {
            "bbox": [round(v, 2) for v in self.bbox],
            "confidence": round(self.confidence, 4),
            "class_id": self.class_id,
            "class_name": self.class_name,
            "track_id": self.track_id,
        }
        if self.plate_text:
            res["plate_text"] = self.plate_text
            res["plate_conf"] = round(self.plate_conf, 4)
        return res

    def __repr__(self) -> str:
        tid = f" id={self.track_id}" if self.track_id is not None else ""
        return (
            f"Detection({self.class_name}{tid} "
            f"conf={self.confidence:.2f} bbox={[round(v,1) for v in self.bbox]})"
        )


# ---------------------------------------------------------------------------
# Main detector
# ---------------------------------------------------------------------------

class VehicleDetector:
    """
    YOLO11n-based vehicle and person detector.

    Automatically selects the best available runtime:
      TensorRT (.engine) → fastest on Jetson
      PyTorch  (.pt)     → standard GPU/CPU

    Parameters
    ----------
    model_path  : Path to model file. None = auto-download yolo11n.pt
    device      : "cpu", "cuda:0", "0", or "auto"
    conf        : Detection confidence threshold
    iou         : NMS IoU threshold
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cpu",
        conf: float = 0.25,
        iou: float = 0.45,
    ) -> None:
        self.device = device
        self.conf = conf
        self.iou = iou
        self._model = None
        self._model_name = "yolov8m"
        self._load_model(model_path)

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _load_model(self, model_path: Optional[str]) -> None:
        try:
            from ultralytics import YOLO  # type: ignore

            if model_path:
                # Either a local file or an ultralytics shorthand name
                # ("yolov8n.pt") — ultralytics auto-downloads/caches the
                # latter, so this also covers requesting a lighter model
                # that isn't already on disk (e.g. the live-analysis path).
                self._model = YOLO(model_path)
                self._model_name = Path(model_path).stem
                logger.info("Loaded model: %s", model_path)
            else:
                # Use local copy in weights dir; ultralytics auto-downloads if missing
                local = Path(__file__).parent.parent / "models" / "weights" / "detection" / "yolov8m.pt"
                self._model = YOLO(str(local) if local.exists() else "yolov8m.pt")
                self._model_name = "yolov8m"
                logger.info("Loaded yolov8m from %s", local if local.exists() else "ultralytics cache")

        except ImportError as exc:
            raise ImportError(
                "ultralytics not installed. Run:  pip install ultralytics"
            ) from exc

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def detect(
        self,
        image: np.ndarray,
        classes: Optional[List[int]] = None,
        imgsz: Optional[int] = None,
    ) -> List[Detection]:
        """
        Single-frame detection (no tracking).

        Parameters
        ----------
        image   : BGR numpy array
        classes : COCO class IDs to keep; None = all traffic classes
        imgsz   : Inference resolution YOLO resizes to internally; None uses
                  the model's default (640). Lowering this (e.g. 384) cuts
                  CPU inference time roughly quadratically — used by the
                  live-analysis path where per-frame latency matters more
                  than the extra few % mAP a full 640px pass buys.

        Returns
        -------
        List[Detection] sorted by confidence descending
        """
        filter_cls = classes if classes is not None else list(ALL_TRAFFIC_CLASS_IDS)

        predict_kwargs = dict(
            source=image,
            conf=self.conf,
            iou=self.iou,
            classes=filter_cls,
            device=self.device,
            verbose=False,
            stream=False,
        )
        if imgsz:
            predict_kwargs["imgsz"] = imgsz
        results = self._model.predict(**predict_kwargs)

        detections: List[Detection] = []
        for result in results:
            for box in result.boxes:
                cid = int(box.cls[0])
                if cid not in ALL_TRAFFIC_CLASS_IDS:
                    continue
                detections.append(
                    Detection(
                        bbox=box.xyxy[0].tolist(),
                        confidence=float(box.conf[0]),
                        class_id=cid,
                    )
                )

        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections

    def detect_with_tracking(
        self,
        image: np.ndarray,
        persist: bool = True,
        tracker: str = "bytetrack.yaml",
        imgsz: Optional[int] = None,
    ) -> List[Detection]:
        """
        Detection + ByteTrack tracking. Returns persistent track_ids.

        Parameters
        ----------
        image   : BGR numpy array
        persist : Keep track state between consecutive calls (use True for video)
        tracker : Tracker config name ("bytetrack.yaml" or "botsort.yaml")
        imgsz   : See ``detect()`` — lower this for faster CPU-only live inference.
        """
        track_kwargs = dict(
            source=image,
            persist=persist,
            tracker=tracker,
            conf=self.conf,
            iou=self.iou,
            classes=list(ALL_TRAFFIC_CLASS_IDS),
            device=self.device,
            verbose=False,
        )
        if imgsz:
            track_kwargs["imgsz"] = imgsz
        results = self._model.track(**track_kwargs)

        detections: List[Detection] = []
        for result in results:
            for box in result.boxes:
                cid = int(box.cls[0])
                if cid not in ALL_TRAFFIC_CLASS_IDS:
                    continue
                tid = int(box.id[0]) if box.id is not None else None
                detections.append(
                    Detection(
                        bbox=box.xyxy[0].tolist(),
                        confidence=float(box.conf[0]),
                        class_id=cid,
                        track_id=tid,
                    )
                )

        return detections

    def stream_video(
        self,
        source,  # str path, int webcam, or cv2.VideoCapture
        use_tracking: bool = True,
    ) -> Iterator[Tuple[np.ndarray, List[Detection]]]:
        """
        Generator that yields (frame, detections) for a video source.
        Memory-efficient streaming for long recordings.

        Usage:
            for frame, detections in detector.stream_video("traffic.mp4"):
                ...
        """
        cap = cv2.VideoCapture(source) if isinstance(source, (str, int)) else source
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                if use_tracking:
                    dets = self.detect_with_tracking(frame)
                else:
                    dets = self.detect(frame)
                yield frame, dets
        finally:
            cap.release()

    # ------------------------------------------------------------------
    # Filter helpers
    # ------------------------------------------------------------------

    def get_vehicles(self, detections: List[Detection]) -> List[Detection]:
        return [d for d in detections if d.is_vehicle]

    def get_persons(self, detections: List[Detection]) -> List[Detection]:
        return [d for d in detections if d.is_person]

    def get_phones(self, detections: List[Detection]) -> List[Detection]:
        return [d for d in detections if d.class_id == PHONE_CLASS_ID]

    def get_two_wheelers(self, detections: List[Detection]) -> List[Detection]:
        return [d for d in detections if d.is_two_wheeler]

    def get_four_wheelers(self, detections: List[Detection]) -> List[Detection]:
        return [d for d in detections if d.is_four_wheeler]

    # ------------------------------------------------------------------
    # Crop utility
    # ------------------------------------------------------------------

    def crop(
        self,
        image: np.ndarray,
        detection: Detection,
        padding: int = 8,
    ) -> np.ndarray:
        """
        Crop image to detection bbox with symmetric padding.
        Clamps to image boundaries automatically.
        """
        h, w = image.shape[:2]
        x1 = max(0, int(detection.bbox[0]) - padding)
        y1 = max(0, int(detection.bbox[1]) - padding)
        x2 = min(w, int(detection.bbox[2]) + padding)
        y2 = min(h, int(detection.bbox[3]) + padding)
        return image[y1:y2, x1:x2].copy()

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_tensorrt(
        self,
        half: bool = True,
        int8: bool = False,
        calib_data: Optional[str] = None,
    ) -> str:
        """
        Export model to TensorRT for Jetson deployment.

        Parameters
        ----------
        half      : FP16 export (recommended for Jetson Orin NX)
        int8      : INT8 quantization (maximum compression, slight accuracy loss)
        calib_data: Path to calibration dataset YAML (required for int8)

        Returns path to exported .engine file.
        """
        logger.info("Exporting to TensorRT (FP16=%s, INT8=%s)…", half, int8)
        kwargs: Dict = {"format": "engine", "device": 0}
        if int8 and calib_data:
            kwargs.update(int8=True, data=calib_data)
        elif half:
            kwargs["half"] = True

        self._model.export(**kwargs)
        engine_path = str(Path(self._model.ckpt_path).with_suffix(".engine"))
        logger.info("TensorRT engine exported: %s", engine_path)
        return engine_path

    def export_tflite(self, int8: bool = True) -> str:
        """Export to TFLite for Raspberry Pi / Coral TPU"""
        logger.info("Exporting to TFLite (INT8=%s)…", int8)
        self._model.export(format="tflite", int8=int8)
        return "model_int8.tflite"

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def model_info(self) -> dict:
        return {
            "device": self.device,
            "conf_threshold": self.conf,
            "iou_threshold": self.iou,
            "classes": list(ALL_TRAFFIC_CLASS_IDS.values()),
        }
