# 🚗 Real-Time Edge AI Driver Monitoring System (DMS)

<p align="center">
  <img src="https://img.shields.io/badge/Platform-Raspberry%20Pi%204%20%7C%20Linux%20%7C%20Windows-blue" alt="Platform">
  <img src="https://img.shields.io/badge/Python-3.9%20%7C%203.11-brightgreen.svg" alt="Python">
  <img src="https://img.shields.io/badge/OpenCV-Computer%20Vision-orange.svg" alt="OpenCV">
  <img src="https://img.shields.io/badge/dlib-68--Point%20Landmarks-red.svg" alt="dlib">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License">
</p>

An intelligent, computer-vision safety pipeline engineered to monitor driver attentiveness in real time. The system calculates **Eye Aspect Ratio (EAR)** for drowsiness and micro-sleep detection, **Mouth Aspect Ratio (MAR)** for yawning frequency, and **Gaze Vectoring** to identify distracted driving.

Optimized to run on resource-constrained Edge hardware (**Raspberry Pi 4 Model B**) by integrating native hardware-accelerated video streaming (`rpicam-vid`) and high-throughput volatile RAM data transfers (`/dev/shm`).

---

## 🖥️ Live Telemetry & HUD Preview

<div align="center">
  
  <br>
  <em>Figure 1: Real-time driver bounding box isolation, 68-point landmark telemetry, and dynamic HUD overlay.</em>
</div>

<br>

| Telemetry Parameter | Live Value | Operational Threshold | Driver State | Safety Indicator |
| :--- | :---: | :---: | :---: | :---: |
| **Driver Identification** | `Driver (Verified)` | Face Encoding Match | Active & Authorized | 🟢 **SECURE** |
| **Eye Aspect Ratio (EAR)** | `0.31` | Normal: $\ge 0.25$ | Eyes Open / Attentive | 🟢 **NORMAL** |
| **Mouth Aspect Ratio (MAR)** | `0.12` | Normal: $< 0.50$ | Mouth Closed / No Yawn | 🟢 **NORMAL** |
| **Gaze Vector** | `CENTER` | Safe: `CENTER`, `UP` | Focused on Roadway | 🟢 **SAFE** |
| **Gaze Persistence** | `0 frames` | Alert: $\ge 10$ frames off-road | Nominal | 🟢 **SAFE** |

---

## 📸 System Architecture & Pipeline

```text
       [ Native Camera Stream ]
                 │
                 ▼
   [ Frame Downscaling & Color Conversion ]
                 │
                 ├──► [ Face Detection & Driver Isolation ]
                 │           (Calculates Largest Surface Area to Exclude Passengers)
                 │
                 ├──► [ 68-Point Facial Landmark Mapping ]
                 │           │
                 │           ├──► Eye Aspect Ratio (EAR) ──► Ring Buffer (deque) ──► Drowsiness Alert
                 │           └──► Mouth Aspect Ratio (MAR) ─► Ring Buffer (deque) ──► Yawn Alert
                 │
                 └──► [ Gaze Estimation Engine ] 
                             └──► Statistical Mode Consensus Filter ──────────────► Distraction Alert
```
## ✨ Key Technical Features

- **Micro-Sleep Detection (EAR):** Computes vertical-to-horizontal eye coordinate Euclidean distances across 6 landmark points per eye. Continuous low values trigger fatigue warnings.
- **Yawn Detection (MAR):** Analyzes inner and outer lip distances to distinguish conversational talking from prolonged yawning.
- **Off-Road Distraction Tracking:** Monitors directional gaze persistence (left, right, down, center) and alerts when visual focus leaves the driving zone.
- **Dynamic Driver Isolation:** Computes bounding-box pixel surface area ($W \times H$) to lock monitoring algorithms exclusively onto the primary driver, eliminating interference from vehicle passengers.
- **Moving Average Signal Smoothing:** Employs high-speed ring buffers (`collections.deque`) to average the last 7 frames of EAR/MAR data, eliminating camera noise and display flicker.
- **Embedded Edge Hardware Optimizations:**
  - **Native Hardware Pipe:** Uses an asynchronous `rpicam-vid` MJPEG stream, bypassing OpenCV V4L2 incompatibilities on modern Debian Bookworm.
  - **Volatile RAM Routing:** Writes cross-module synchronization frames directly to `/dev/shm` (Linux RAM disk), preventing physical SD card wear and reducing I/O latency.

---

## 📊 Mathematical Formulation

### 1. Eye Aspect Ratio (EAR)
Tracks eye closure dynamically across landmarks $p_1 \dots p_6$:

$$\text{EAR} = \frac{\|p_2 - p_6\| + \|p_3 - p_5\|}{2\|p_1 - p_4\|}$$

### 2. Mouth Aspect Ratio (MAR)
Evaluates mouth vertical opening relative to mouth width across landmarks $p_{49} \dots p_{68}$:

$$\text{MAR} = \frac{\|p_{51} - p_{59}\| + \|p_{53} - p_{57}\|}{2\|p_{49} - p_{55}\|}$$

### 3. Gaze Mode Consensus
Filters out micro-movements by finding the statistical mode over a sliding window:

$$\text{Gaze}_{\text{stable}} = \operatorname{mode}\left(\text{Gaze}_{t-k}, \dots, \text{Gaze}_t\right)$$

---

## 🛠️ Hardware & Software Specifications

- **Compute Platform:** Raspberry Pi 4 Model B (Broadcom BCM2711, Quad-core Cortex-A72 @ 1.8GHz) / Standard PC
- **Camera Module:** Raspberry Pi Camera Module (CSI Ribbon, OV5647) or standard USB UVC Webcam
- **Operating System:** Raspberry Pi OS 64-bit (Debian Bookworm) / Ubuntu 22.04 LTS / Windows 11
- **Core Dependencies:** OpenCV, dlib, face_recognition, NumPy, imutils                             
