"""GARUDA — _routers.py (DEPRECATED)

This file previously held Cameras, Vehicles, Analytics, Stream, and Debug
routers in one monolithic module.  As of the architectural refactor they have
been split into:

  backend/api/cameras.py   — CamerasRouter
  backend/api/vehicles.py  — VehiclesRouter
  backend/api/analytics.py — AnalyticsRouter
  backend/api/stream.py    — WebSocket stream + broadcast_violation helper
  backend/api/debug.py     — Debug / test injection endpoints

The ML singleton is now the shared registry at:
  backend/services/ml_registry.py  — get_ml_registry()

DO NOT import from this file in new code.
This file exists only to raise a descriptive ImportError if legacy code still
references it, making the migration failure obvious.
"""

raise ImportError(
    "_routers.py has been deprecated and replaced by per-domain router modules. "
    "Import from: cameras.py, vehicles.py, analytics.py, stream.py, debug.py "
    "or backend.services.ml_registry."
)
