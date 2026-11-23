\# 🚗 Real-Time Edge AI Driver Monitoring System (DMS)



\[!\[Platform](https://img.shields.io/badge/Platform-Raspberry%20Pi%204%20%7C%20Linux%20%7C%20Windows-blue)](https://www.raspberrypi.com/)

\[!\[Python](https://img.shields.io/badge/Python-3.9%20%7C%203.11-brightgreen.svg)](https://www.python.org/)

\[!\[OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-orange.svg)](https://opencv.org/)

\[!\[dlib](https://img.shields.io/badge/dlib-68--Point%20Landmarks-red.svg)](http://dlib.net/)

\[!\[License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)



An intelligent, computer-vision-based safety system engineered to monitor driver attentiveness in real-time. The system processes video streams to estimate \*\*Eye Aspect Ratio (EAR)\*\* for micro-sleep/drowsiness detection, \*\*Mouth Aspect Ratio (MAR)\*\* for yawning frequency, and \*\*Gaze Vectoring\*\* to detect off-road distractions.



Optimized to run on resource-constrained Edge hardware (\*\*Raspberry Pi 4 Model B\*\*) using hardware-accelerated camera pipelines and low-latency in-memory data routing.



\---



\## 📸 System Architecture \& Pipeline



```text

&#x20;      \[ Native Camera Stream ]

&#x20;                │

&#x20;                ▼

&#x20;  \[ Frame Downscaling \& Color Conversion ]

&#x20;                │

&#x20;                ├──► \[ Face Detection \& Driver Isolation ]

&#x20;                │           (Targets Largest Bounding Box)

&#x20;                │

&#x20;                ├──► \[ 68-Point Facial Landmark Mapping ]

&#x20;                │           │

&#x20;                │           ├──► Eye Aspect Ratio (EAR) ──► Moving Average Filter ──► Drowsiness Alert

&#x20;                │           └──► Mouth Aspect Ratio (MAR) ─► Moving Average Filter ──► Yawn Alert

&#x20;                │

&#x20;                └──► \[ Gaze Estimation Engine ] 

&#x20;                            └──► Statistical Mode Consensus ──────────────────────► Distraction Alert

