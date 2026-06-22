import urllib.request
import urllib.parse
import mimetypes
import uuid
import os
import time
import json

def submit_file(file_path):
    url = "http://localhost:8000/api/v1/jobs/upload"
    filename = os.path.basename(file_path)
    name = os.path.splitext(filename)[0]

    # Read file bytes
    with open(file_path, "rb") as f:
        file_bytes = f.read()

    # Generate multipart boundary
    boundary = f"----WebKitFormBoundary{uuid.uuid4().hex}"
    
    # Construct multipart request payload
    data = []
    
    # Field: name
    data.append(f"--{boundary}".encode("utf-8"))
    data.append(f'Content-Disposition: form-data; name="name"'.encode("utf-8"))
    data.append(b"")
    data.append(name.encode("utf-8"))
    
    # Field: source_type
    data.append(f"--{boundary}".encode("utf-8"))
    data.append(f'Content-Disposition: form-data; name="source_type"'.encode("utf-8"))
    data.append(b"")
    data.append(b"Image")
    
    # Field: file
    mime_type = mimetypes.guess_type(file_path)[0] or "image/jpeg"
    data.append(f"--{boundary}".encode("utf-8"))
    data.append(f'Content-Disposition: form-data; name="file"; filename="{filename}"'.encode("utf-8"))
    data.append(f"Content-Type: {mime_type}".encode("utf-8"))
    data.append(b"")
    data.append(file_bytes)
    
    data.append(f"--{boundary}--".encode("utf-8"))
    data.append(b"")
    
    body = b"\r\n".join(data)
    
    req = urllib.request.Request(url, data=body)
    req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    req.add_header("Content-Length", str(len(body)))
    
    try:
        with urllib.request.urlopen(req) as response:
            if response.status == 201:
                res_data = json.loads(response.read().decode("utf-8"))
                print(f"  [OK] Submitted {filename} -> JobID: {res_data.get('id')}")
                return res_data.get("id")
            else:
                print(f"  [FAIL] Failed to submit {filename}, status code: {response.status}")
    except Exception as e:
        print(f"  [FAIL] Error submitting {filename}: {e}")
    return None

def main():
    test_dir = "test"
    files = [
        "WhatsApp Image 2026-06-20 at 12.24.39 PM.jpeg",
        "WhatsApp Image 2026-06-20 at 12.24.40 PM.jpeg",
        "WhatsApp Image 2026-06-20 at 12.24.41 PM.jpeg"
    ]
    
    job_ids = []
    print("Submitting 3 test images to backend...")
    for f in files:
        path = os.path.join(test_dir, f)
        if os.path.exists(path):
            jid = submit_file(path)
            if jid:
                job_ids.append(jid)
        else:
            print(f"  [WARN] File not found: {path}")

    # Wait for the jobs to finish
    if not job_ids:
        return
        
    print("\nWaiting for jobs to finish processing...")
    completed_jobs = set()
    for _ in range(30): # Timeout after 60s
        time.sleep(2)
        all_done = True
        for jid in job_ids:
            if jid in completed_jobs:
                continue
            try:
                url = f"http://localhost:8000/api/v1/jobs/{jid}"
                req = urllib.request.Request(url)
                with urllib.request.urlopen(req) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    status = data.get("status")
                    progress = data.get("progress")
                    print(f"  Job {jid}: {status} ({progress}%)")
                    if status == "Completed" or status == "Failed":
                        completed_jobs.add(jid)
                    else:
                        all_done = False
            except Exception as e:
                print(f"  Error checking job {jid}: {e}")
                all_done = False
        if all_done:
            break
            
    print("\nAll submissions complete.")

if __name__ == "__main__":
    main()
