"""GARUDA Backend — Services layer.

Business-logic modules that are independent of HTTP transport:
  ml_registry       — shared ML pipeline singleton (detector, OCR, classifier…)
  calibration_service — camera calibration resolver
  challan_service   — violation packaging, evidence building, challan dispatch
"""
from .ml_registry import get_ml_registry, MLRegistry  # noqa: F401
