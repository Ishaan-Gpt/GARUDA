import urllib.request
import json
import sys

def test_endpoints():
    base_url = "http://localhost:8000/api/v1"
    endpoints = ["/cameras", "/violations?page_size=5", "/jobs"]
    
    print("Testing connection to backend API...")
    for endpoint in endpoints:
        url = base_url + endpoint
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req) as response:
                if response.status == 200:
                    data = json.loads(response.read().decode('utf-8'))
                    print(f"  [OK] {endpoint} returned 200. Count of items: {len(data) if isinstance(data, list) else len(data.get('violations', []))}")
                else:
                    print(f"  [FAIL] {endpoint} returned status {response.status}")
                    sys.exit(1)
        except Exception as e:
            print(f"  [FAIL] {endpoint} failed to connect: {e}")
            sys.exit(1)
            
    print("All endpoints returned 200 OK.")

if __name__ == "__main__":
    test_endpoints()
