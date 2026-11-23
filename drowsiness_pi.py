import face_recognition
import cv2
import numpy as np
import dlib
import os
import urllib.request
import bz2
import time
import subprocess
import threading
from collections import deque

try:
    import eye_game
except ImportError as e:
    print(f"Failed to load eye_game. REAL ERROR: {e}")
    print("Ensure eye_game.py is in the same directory and all its dependencies are installed in the (env).")
    exit()

# ── CUSTOM RASPBERRY PI NATIVE CAMERA ─────────────────────────────────────────
class RpiCamStream:
    def __init__(self, width=640, height=480, framerate=15):
        self.width = width
        self.height = height
        self.framerate = framerate
        self.frame = None
        self.stopped = False
        self.process = None

    def start(self):
        print("[INFO] Starting rpicam-vid hardware stream...")
        command =[
            "rpicam-vid",
            "-t", "0",
            "--nopreview",
            "--width", str(self.width),
            "--height", str(self.height),
            "--framerate", str(self.framerate),
            "--codec", "mjpeg",
            "-o", "-"
        ]
        self.process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=10**8)
        t = threading.Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        bytes_data = b''
        while not self.stopped:
            try:
                chunk = self.process.stdout.read(4096)
                if not chunk:
                    self.stopped = True
                    break
                bytes_data += chunk
                
                # Look for start and end of JPEG frames
                a = bytes_data.find(b'\xff\xd8')
                b = bytes_data.find(b'\xff\xd9')
                
                if a != -1 and b != -1:
                    if a > b:
                        # Bad frame alignment, drop old data
                        bytes_data = bytes_data[a:]
                        continue
                        
                    jpg = bytes_data[a:b+2]
                    bytes_data = bytes_data[b+2:]
                    
                    # Decode hardware MJPEG frame into OpenCV array
                    img_array = np.frombuffer(jpg, dtype=np.uint8)
                    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                    if frame is not None:
                        self.frame = frame
            except Exception:
                break
        if self.process:
            self.process.kill()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        if self.process:
            self.process.kill()

# ── Smoothing Filters (Mathematical Stability Setup) ──────────────────────────
ear_history  = deque(maxlen=7)  # Retains average across last 7 consecutive frames
mar_history  = deque(maxlen=7)  # Prevents jumping during pixel micro-errors
gaze_history = deque(maxlen=9)  # Takes consensus over odd interval of states

# Variables initializing outputs gracefully
smooth_ear = 0.0
smooth_mar = 0.0
smooth_gaze = "center"

# ── Thresholds ────────────────────────────────────────────────────────────────
EAR_THRESH        = 0.25   # below → eyes closing
EAR_CONSEC_FRAMES = 15     # consecutive frames to trigger alert
MAR_THRESH        = 0.5    # above → yawning
MAR_CONSEC_FRAMES = 15

ear_counter = 0
mar_counter = 0
previous    = "center"
gaze_count  = 0

SAFE_GAZES = {"center", "up"}
MILD_GAZES = {"left", "right", "upper left", "upper right"}
DANGER_GAZES = {"down", "lower left", "lower right"}

# ── dlib 68-landmark model ────────────────────────────────────────────────────
MODEL_PATH    = "shape_predictor_68_face_landmarks.dat"
MODEL_BZ2_URL = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"

if not os.path.exists(MODEL_PATH):
    print("Downloading facial landmark model (~100 MB), please wait...")
    bz2_path = MODEL_PATH + ".bz2"

    def _progress(block, block_size, total):
        done = block * block_size
        if total > 0:
            pct = min(done * 100 // total, 100)
            print(f"\r  {pct}% ({done // 1_048_576} MB / {total // 1_048_576} MB)", end="", flush=True)

    urllib.request.urlretrieve(MODEL_BZ2_URL, bz2_path, reporthook=_progress)
    print("\nExtracting...")
    with bz2.open(bz2_path, "rb") as fin, open(MODEL_PATH, "wb") as fout:
        fout.write(fin.read())
    os.remove(bz2_path)
    print("Model ready.")

landmark_predictor = dlib.shape_predictor(MODEL_PATH)

LEFT_EYE  = list(range(36, 42))
RIGHT_EYE = list(range(42, 48))
MOUTH     = list(range(48, 68))

def eye_aspect_ratio(pts):
    A = np.linalg.norm(pts[1] - pts[5])
    B = np.linalg.norm(pts[2] - pts[4])
    C = np.linalg.norm(pts[0] - pts[3])
    return (A + B) / (2.0 * C) if C > 0 else 0.0

def mouth_aspect_ratio(pts):
    A = np.linalg.norm(pts[2]  - pts[10])   
    B = np.linalg.norm(pts[4]  - pts[8])    
    C = np.linalg.norm(pts[0]  - pts[6])    
    return (A + B) / (2.0 * C) if C > 0 else 0.0

if not os.path.exists("image_data"):
    os.makedirs("image_data")

temp_gaze_file = "/dev/shm/temp_gaze_image.jpg"
REF_IMG        = "image_data/img.jpg"

# ── Auto-capture reference photo ──────────────────────────────────────────────
if not os.path.exists(REF_IMG):
    print("No reference photo found. Opening Native RPi Camera.")
    print("  -> Look at camera and press SPACE to capture, or Q to quit.")
    
    cap = RpiCamStream(width=640, height=480, framerate=15).start()
    time.sleep(2.0)
    
    captured = False
    while True:
        frame_cap = cap.read()
        if frame_cap is None:
            continue
            
        disp = frame_cap.copy()
        cv2.putText(disp, "Press SPACE to capture your face",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Capture Reference Photo", disp)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            cv2.imwrite(REF_IMG, frame_cap)
            print("Saved:", REF_IMG)
            captured = True
            break
        elif key == ord('q'):
            break
            
    cap.stop()
    cv2.destroyAllWindows()
    if not captured:
        exit()

# ── Load known face encoding ──────────────────────────────────────────────────
img_image = face_recognition.load_image_file(REF_IMG)
encodings = face_recognition.face_encodings(img_image)
if len(encodings) == 0:
    print("No face found in", REF_IMG, "- delete it and rerun.")
    exit()

known_face_encodings = [encodings[0]]
known_face_names     = ["Driver"]

face_locations =[]
face_names     =[]
process_this_frame = True

# ── Start Main Video Stream ───────────────────────────────────────────────────
vs = RpiCamStream(width=640, height=480, framerate=15).start()
time.sleep(2.0) 

# ── Main loop ─────────────────────────────────────────────────────────────────
while True:
    frame = vs.read()
    
    if frame is None:
        time.sleep(0.01)
        continue

    small_frame     = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1])
    gray_frame      = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if process_this_frame:
        face_locations   = face_recognition.face_locations(rgb_small_frame, model="hog")
        face_enc_list    = face_recognition.face_encodings(rgb_small_frame, face_locations)
        
        cv2.imwrite(temp_gaze_file, frame)
        face_names =[]

        for fe in face_enc_list:
            matches    = face_recognition.compare_faces(known_face_encodings, fe)
            name       = "Unknown"
            distances  = face_recognition.face_distance(known_face_encodings, fe)
            best       = int(np.argmin(distances))

            if matches[best]:
                name = known_face_names[best]
                
                try:
                    direction = eye_game.get_gaze_direction(temp_gaze_file)
                    if direction:
                        gaze_history.append(direction)
                        
                    if len(gaze_history) > 0:
                        # Identifies true focal vector out of last 'X' tracked arrays stabilizing readout text!
                        smooth_gaze = max(set(gaze_history), key=gaze_history.count)
                    
                    if previous != smooth_gaze:
                        previous = smooth_gaze
                        gaze_count = 0
                    else:
                        gaze_count += 1
                except Exception:
                    pass

            face_names.append(name)

    process_this_frame = not process_this_frame

    # Process Mathematics purely strictly targeted against DRIVERS isolation avoiding passenger jumps
    if len(face_locations) > 0:
        largest_face_idx = 0
        max_area = 0
        for i, (t_box, r_box, b_box, l_box) in enumerate(face_locations):
            area = (b_box - t_box) * (r_box - l_box)
            if area > max_area:
                max_area = area
                largest_face_idx = i

        top, right, bottom, left = face_locations[largest_face_idx]
        t, r, b, l = top * 4, right * 4, bottom * 4, left * 4
        rect  = dlib.rectangle(l, t, r, b)
        shape = landmark_predictor(gray_frame, rect)
        pts   = np.array([[shape.part(i).x, shape.part(i).y] for i in range(68)])

        le  = eye_aspect_ratio(pts[LEFT_EYE])
        re  = eye_aspect_ratio(pts[RIGHT_EYE])
        raw_ear = (le + re) / 2.0
        raw_mar = mouth_aspect_ratio(pts[MOUTH])

        # Safely buffer raw findings in chronological histories to wipe frame micro skips natively
        ear_history.append(raw_ear)
        mar_history.append(raw_mar)
        smooth_ear = sum(ear_history) / len(ear_history)
        smooth_mar = sum(mar_history) / len(mar_history)

        if smooth_ear > 0.01 and smooth_ear < EAR_THRESH:
            ear_counter += 1
            if ear_counter >= EAR_CONSEC_FRAMES:
                cv2.putText(frame, "ALERT: Eyes Closed!", (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
        else:
            ear_counter = 0

        if smooth_mar > MAR_THRESH:
            mar_counter += 1
            if mar_counter >= MAR_CONSEC_FRAMES:
                cv2.putText(frame, "ALERT: Yawning!", (10, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 140, 255), 2)
        else:
            mar_counter = 0

    # ── HUD rendering variables with smooth implementations replacing previous single calls ─────────────
    ear_color = (0, 0, 255) if (smooth_ear > 0.01 and smooth_ear < EAR_THRESH) else (0, 255, 255)
    mar_color = (0, 140, 255) if smooth_mar > MAR_THRESH else (0, 255, 255)

    if smooth_gaze in SAFE_GAZES:
        gaze_color = (0, 220, 80)
    elif smooth_gaze in MILD_GAZES:
        gaze_color = (0, 200, 255)
    else:                                
        gaze_color = (0, 60, 255)

    # Added string `: .2f` formatter trims extra unnecessary unstable floats
    cv2.putText(frame, f"EAR: {smooth_ear:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, ear_color, 2)
    cv2.putText(frame, f"MAR: {smooth_mar:.2f}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, mar_color, 2)
    cv2.putText(frame, f"Gaze: {smooth_gaze.upper()}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.70, gaze_color, 2)
    cv2.putText(frame, f"Gaze fixed: {gaze_count} frames", (10, 118),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 0), 1)

    hint = f"EAR<{EAR_THRESH}=Drowsy  MAR>{MAR_THRESH}=Yawn  Gaze>=10=Alert"
    cv2.putText(frame, hint, (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 160, 160), 1)

    if gaze_count >= 10 and smooth_gaze not in SAFE_GAZES:
        alert_msg = f"ALERT: Gaze Fixed {smooth_gaze.upper()}!"
        cv2.putText(frame, alert_msg, (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 0, 255), 2)
    if smooth_gaze in DANGER_GAZES and gaze_count >= 5:
        cv2.putText(frame, "ALERT: Looking Away from Road!", (10, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4; right *= 4; bottom *= 4; left *= 4
        box_color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (left, top), (right, bottom), box_color, 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), box_color, cv2.FILLED)
        cv2.putText(frame, name, (left + 6, bottom - 6),
                    cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

    cv2.imshow("Driver Drowsiness Monitor", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vs.stop()
cv2.destroyAllWindows()
