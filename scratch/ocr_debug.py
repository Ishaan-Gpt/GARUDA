import easyocr
import cv2
import re

reader = easyocr.Reader(['en'], gpu=False)
img_path = r"c:\Ishaan GPT\hackthons\Garuda\test\WhatsApp Image 2026-06-20 at 12.25.15 PM (1).jpeg"
img = cv2.imread(img_path)

if img is None:
    print("Error: Could not read image")
else:
    print("Image dimensions:", img.shape)
    
    # Run OCR on the full image
    print("\n--- Running EasyOCR on full image ---")
    results = reader.readtext(img)
    for bbox, text, conf in results:
        print(f"[{conf:.3f}] '{text}' at {bbox}")
        
    # Also run on bottom strip (often where the plate resides)
    print("\n--- Running EasyOCR on bottom strip ---")
    h, w = img.shape[:2]
    bottom_strip = img[int(h * 0.5):, :]
    results_strip = reader.readtext(bottom_strip)
    for bbox, text, conf in results_strip:
        print(f"[{conf:.3f}] '{text}' at {bbox}")
