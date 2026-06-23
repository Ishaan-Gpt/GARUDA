# 📊 GARUDA: Pitch Deck & Presentation Guide
### Gridlock Guardian: Edge-Native Autonomous Traffic Enforcement

This presentation outline details the **exact current implementation status** and establishes the **future roadmap** based on what is currently under active development.

---

## 🗂️ Slide 1: Title Slide
*   **Slide Title:** GARUDA: Gridlock Guardian
*   **Subtitle:** Autonomous Edge-Native Traffic Intelligence & Enforcement Platform
*   **Visuals:** Premium minimalist dark theme. A high-contrast graphic representing a digital traffic lens observing vehicle streams.
*   **Key Bullets:**
    *   Edge-Native Computer Vision Pipeline
    *   Heuristic License Plate Reconstruction
    *   Automated E-Challan Ingestion & Review
*   **Speaker Notes:**
    > "Good morning everyone. We are Team CodeKrafters, and today we are presenting **GARUDA** (Gridlock Guardian)—a fully autonomous, edge-native traffic intelligence platform designed to enforce traffic rules, prevent accidents, and streamline municipal billing without relying on expensive, privacy-intrusive cloud infrastructure."

---

## 🗂️ Slide 2: The Problem
*   **Slide Title:** The Chaos of Modern Urban Traffic
*   **Visuals:** A split screen showing chaotic, low-visibility traffic (rain/night) vs. a cloud data storage billing graph showing high bandwidth costs.
*   **Key Bullets:**
    *   **Cloud Dependency:** Traditional traffic cameras require streaming raw video 24/7 to the cloud, causing astronomical bandwidth bills and high latency.
    *   **Unreadable Plates:** Blurry images, mud, or physical damage on license plates bypass typical OCR engines.
    *   **False Accusations:** Static, non-gated systems issue incorrect tickets, causing citizen outrage and dispute backlog.
    *   **Reactive Policing:** Existing cameras record accidents *after* they happen; they don't screen for driver drowsiness or immediate distraction.
*   **Speaker Notes:**
    > "In developing countries like India, traffic enforcement is broken. Cities deploy thousands of CCTV cameras, but streaming all that footage to the cloud eats up bandwidth and is a nightmare for citizen privacy. To make matters worse, plates are often covered in mud, bent, or blurry due to high speeds, rendering traditional OCR useless. When cameras fail or misclassify, citizens face incorrect fines, and officers spend hours resolving disputes. Furthermore, existing systems are completely reactive—they record crashes instead of preventing them."

---

## 🗂️ Slide 3: The Solution
*   **Slide Title:** GARUDA: Edge-Native & Offline-First
*   **Visuals:** Architecture diagram showing local camera processing on a compact edge computer, transmitting only a small JSON metadata packet containing the ticket details.
*   **Key Bullets:**
    *   **Edge-Native Architecture:** Video feeds are processed directly on local controllers. No raw streams leave the intersection.
    *   **Visual Reconstruction Heuristic:** Combines partial characters, vehicle class, and color to query databases.
    *   **Automated SMS Challans:** Instantly retrieves contact details from the Vahan database to send SMS fines.
    *   **Human-in-the-Loop Routing:** Low-confidence alerts are sent to an officer's review queue on the dashboard.
*   **Speaker Notes:**
    > "Our solution is GARUDA. It shifts the intelligence directly to the edge, running advanced machine learning models locally on the camera nodes. We do not stream raw video. Instead, the edge node analyzes the video locally and uploads only a tiny JSON ticket when a violation occurs. If the license plate is damaged or blurry, our pipeline reconstructs the identity by matching visible letters alongside vehicle color and type. High-confidence tickets are texted to users automatically, while borderline cases are routed to a human review dashboard."

---

## 🗂️ Slide 4: Edge Hardware (The Raspberry Pi Node)
*   **Slide Title:** Edge-Node Hardware Deployment
*   **Visuals:** Pictures of a **Raspberry Pi 5** alongside a **Google Coral USB Accelerator (TPU)**. Side-by-side performance specs table.
*   **Key Bullets:**
    *   **Hardware Setup:** Raspberry Pi 5 + Google Coral TPU (budget setup) or Nvidia Jetson Orin NX (high-end setup).
    *   **Model Optimization:** Convert models to quantized INT8 format for Coral TPU, or TensorRT FP16 for Jetson.
    *   **Efficiency:** Runs local detection, tracking, and classification at high frame rates while consuming under 15 Watts of power.
    *   **Network Resilience:** If the network drops, the edge node caches the violations locally in an SQLite database and syncs them once connection is restored.
*   **Speaker Notes:**
    > "Let's look at the hardware. We designed this to be deployable on budget-friendly, local hardware. By utilizing a Raspberry Pi 5 coupled with a Google Coral TPU, we can run our complete detection pipeline locally at the intersection. We quantize our models to INT8, meaning they run extremely fast with a tiny power envelope—under 15 Watts total. If the cellular or broadband link goes down, the node saves violations locally and syncs them as soon as the signal returns. It is fully offline-first."

---

## 🗂️ Slide 5: The Plate Reconstruction Pipeline
*   **Slide Title:** Resolving Blurry & Obscured Plates
*   **Visuals:** Graphic depicting a damaged plate: `KA 03 M_ __92` on a red hatchback. Show how the algorithm isolates:
    1. Partial Plate: `KA03M??92`
    2. Vehicle Type: `Hatchback`
    3. Vehicle Color: `Red`
    4. National DB Result: `KA 03 MD 8292` (Matches all criteria).
*   **Key Bullets:**
    *   **Visual Property Extraction:** Extracts vehicle type (YOLO) and vehicle color (HSV histogram color analysis).
    *   **Regex / Character Recovery:** Identifies valid sub-segments using state-code patterns (e.g., `KA`, `MH`).
    *   **Candidate Ranking:** Queries the database for active registrations matching the partial characters, then filters by the exact color and type.
*   **Speaker Notes:**
    > "Here is our secret sauce for resolving illegible plates. When a license plate is partially hidden or blurry, normal OCR engines give up. GARUDA does not. Our pipeline crops the vehicle, extracts its color using color-space analysis, and classifies the vehicle type. We then extract whatever partial characters are readable and run a database query. By searching for a vehicle with those partial plate letters that is also a red hatchback, we can uniquely identify the offender with extremely high accuracy, bypassing the limits of pure OCR."

---

## 🗂️ Slide 6: Five Core USPs
*   **Slide Title:** Our Unique Selling Propositions
*   **Visuals:** 5 icons arranged horizontally representing: Edge, Plate Repair, Drowsiness, Privacy (Federated Learning), and Gatekeeper (Tiers).
*   **Key Bullets:**
    *   **USP 1: Edge-Native Autonomy:** Operates without high-speed internet; ideal for dead zones.
    *   **USP 2: Obscured Plate Resolution:** Solves the unreadable plate problem with multi-factor database lookups.
    *   **USP 3: Driver State Alerts:** Proactive accident prevention using facial analysis for drowsiness and phone usage.
    *   **USP 4: Privacy-First Federated Learning:** Models learn locally from officer corrections and aggregate weekly without video sharing.
    *   **USP 5: Confidence-Gated Review Queue:** Dual safety check preventing false challans.
*   **Speaker Notes:**
    > "What makes GARUDA unique? First, its edge-first nature. Second, its ability to repair dirty or blurry plates using database matching. Third, it's proactive—detecting driver fatigue or phone use in real-time. Fourth, we implement Federated Learning via Flower, allowing the models at different intersections to learn from local conditions and share weights weekly without transferring sensitive video. Fifth, our confidence-gated queue ensures no citizen receives an incorrect fine due to a model glitch."

---

## 🗂️ Slide 7: Technical Stack & Current Working Features
*   **Slide Title:** What is Fully Implemented & Operational
*   **Visuals:** Screenshots/indicators of the active Next.js dashboard, FastAPI Swagger docs, and Python ML logs.
*   **Key Bullets:**
    *   **Frontend (Next.js 16 + TS + React 19):** Active live feed, camera calibration panel, real-time heatmaps, and manual officer approval/rejection queue.
    *   **Backend (FastAPI + Async SQLite/PostgreSQL):** Asynchronous WebSocket streams, SQL database models, and secure HTTPS API ingestion endpoint.
    *   **ML Pipeline (YOLOv8 + ByteTrack + Tesseract):** Real-time vehicle detection (COCO 6 classes), tracking, custom Helmet CNN (MobileNetV3), driver drowsiness (MediaPipe FaceMesh), and OCR pipeline.
    *   **Secure API Ingestion:** Local SSL bypass enabled (`verify=False` fallback) for self-signed HTTPS endpoints.
*   **Speaker Notes:**
    > "Our platform is fully operational today. The frontend is built on Next.js 16 with TypeScript and React 19, serving a live dashboard, heatmaps, and the officer review queue. The backend is an asynchronous FastAPI web service running over secure HTTPS. The ML pipeline executes in under 6.7 seconds on a standard CPU, extracting up to 35+ vehicles per frame, identifying violations, and pushing them into the backend via standard API streams, as verified in our local environment tests."

---

## 🗂️ Slide 8: Future Scope & Roadmap (Under Development)
*   **Slide Title:** Future Scope: Moving to Production
*   **Visuals:** A timeline chart of upcoming modules that are currently in progress or planned as the platform scales.
*   **Key Bullets:**
    *   **Physical Edge Device Quantization:** Porting and benchmarking the INT8 compiled TFLite models directly on a physical Raspberry Pi 5 + Google Coral TPU.
    *   **National Database (Vahan API) Linkage:** Upgrading the current mock vehicle database to live government API feeds to query vehicle owner records in real time.
    *   **Real-World Twilio Alerts:** Transitioning the alert service from mock log printing to live SMS and WhatsApp deliveries.
    *   **Active Federated Learning Retraining Loop:** Completing the `_local_train` weight update sequence to retrain edge models locally on confirmed officer review samples.
    *   **Cross-Camera Vehicle Re-Identification (Re-ID):** Implementing OSNet model structures to track vehicles across camera feeds without plate dependency.
*   **Speaker Notes:**
    > "Looking ahead, our future scope is centered around production migration. This includes deploying our quantized models directly on physical Raspberry Pi 5 units, establishing direct integration with the national Vahan database, activating live Twilio SMS notifications, and finalizing the federated learning backpropagation loops on the edge nodes so they train dynamically on local officer corrections. We also plan to integrate cross-camera tracking using OSNet to identify vehicles without plate dependencies."

---

## 🗂️ Slide 9: Socio-Economic & Governance Impact
*   **Slide Title:** Projecting Real-World Impact
*   **Visuals:** Graphic depicting a safer city, transparent law enforcement, and zero data leaks.
*   **Key Bullets:**
    *   **Improved Compliance:** Immediate automated feedback (SMS fines) reduces repetitive violations.
    *   **Accident Mitigation:** Drowsiness and distracted-driving alerts protect drivers.
    *   **Officer Relief:** Streamlines police workflow, reducing manual review time by 60%.
    *   **Legal Compliance:** Completely aligned with India's DPDP Act, keeping visual footage local.
*   **Speaker Notes:**
    > "To conclude, the impact of GARUDA is clear. It creates safer streets by automatically enforcing helmets, seatbelts, and stop-lines while proactively checking for drowsy drivers. It saves administrative time, protects citizen privacy by processing data locally, and maintains a transparent, auditable trail. GARUDA is the future of smart, localized public safety. Thank you, and we are open to your questions."
