from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from typing import List
import os

router = APIRouter(prefix="/evidence")

@router.get("/test-gallery/list", response_model=List[str])
def list_test_gallery_images():
    test_dir = "test"
    if not os.path.exists(test_dir):
        return []
    files = [f for f in os.listdir(test_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
    return sorted(files)


@router.get("/test-gallery/image/{filename}")
def get_test_gallery_image(filename: str):
    filepath = os.path.join("test", filename)
    # Basic directory traversal security check
    normalized_path = os.path.abspath(filepath)
    test_dir_abs = os.path.abspath("test")
    if not normalized_path.startswith(test_dir_abs):
        raise HTTPException(status_code=403, detail="Access denied")
    
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(filepath)
