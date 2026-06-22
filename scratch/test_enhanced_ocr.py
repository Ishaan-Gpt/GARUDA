import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.pipeline.ocr import PlateOCR

def run_tests():
    print("Initializing PlateOCR...")
    ocr = PlateOCR()
    
    # Test cases for the enhanced sliding-window parser
    test_cases = [
        # (raw ocr text, expected formatted text, expected validity)
        ("INDMH12AB1234", "MH-12-AB-1234", True),
        ("MH12AB1O34", "MH-12-AB-1034", True),      # 'O' correction
        ("KA01ABl234", "KA-01-AB-1234", True),      # 'l'/'L' correction
        ("DL 3C AY 4321", "DL-3C-AY-4321", True),   # spaces and DL formatting
        ("NOISE KA04MX1122", "KA-04-MX-1122", True), # noise prefix
        ("DL1CAB1234NOISE", "DL-1C-AB-1234", True), # noise suffix
        ("HR26DK9988", "HR-26-DK-9988", True),
    ]

    print("\n--- Running Parsing Tests ---")
    all_passed = True
    for raw, expected, exp_valid in test_cases:
        formatted, is_valid = ocr._parse_plate(raw)
        passed = (is_valid == exp_valid) and (formatted == expected or formatted.replace("-", "") == expected.replace("-", ""))
        status = "PASSED" if passed else "FAILED"
        print(f"Raw: '{raw}' | Expected: '{expected}' | Got: '{formatted}' (valid={is_valid}) | Result: {status}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\nAll parser tests PASSED successfully!")
    else:
        print("\nSome parser tests FAILED.")

if __name__ == "__main__":
    run_tests()
