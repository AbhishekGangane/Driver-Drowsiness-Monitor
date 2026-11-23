import face_recognition
import cv2
import numpy as np
import dlib
import os
import urllib.request
import bz2

try:
    import eye_game
except ImportError:
    print("eye_game is not installed.")
    exit()

# ── Thresholds ────────────────────────────────────────────────────────────────
EAR_THRESH        = 0.25   # below → eyes closing
EAR_CONSEC_FRAMES = 20     # consecutive frames to trigger alert
MAR_THRESH        = 0.5   # above → yawning
MAR_CONSEC_FRAMES = 15

ear_counter = 0
mar_counter = 0
previous    = "Unknown"
gaze_count  = 0
gaze_dir    = "center"   # live gaze direction label shown on HUD

# Directions considered "safe" (looking at road)
SAFE_GAZES = {"center", "up"}
# Directions that are mildly off (corner glances)
MILD_GAZES = {"left", "right", "upper left", "upper right"}
# Directions that are dangerously off
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
    A = np.linalg.norm(pts[2]  - pts[10])   # 50-58
    B = np.linalg.norm(pts[4]  - pts[8])    # 52-56
    C = np.linalg.norm(pts[0]  - pts[6])    # 48-54
    return (A + B) / (2.0 * C) if C > 0 else 0.0


# ── Folder setup ──────────────────────────────────────────────────────────────
if not os.path.exists("image_data"):
    os.makedirs("image_data")

file    = "image_data/image.jpg"
REF_IMG = "image_data/img.jpg"

# ── Auto-capture reference photo ──────────────────────────────────────────────
if not os.path.exists(REF_IMG):
    print("No reference photo found. Opening webcam.")
    print("  -> Look at camera and press SPACE to capture, or Q to quit.")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam.")
        exit()
    captured = False
    while True:
        ret, frame_cap = cap.read()
        if not ret:
            break
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
    cap.release()
    cv2.destroyAllWindows()
    if not captured:
        exit()

# ── Load known face encoding ──────────────────────────────────────────────────
video_capture = cv2.VideoCapture(0)

img_image = face_recognition.load_image_file(REF_IMG)
encodings = face_recognition.face_encodings(img_image)
if len(encodings) == 0:
    print("No face found in", REF_IMG, "- delete it and rerun.")
    exit()

known_face_encodings = [encodings[0]]
known_face_names     = ["Abhish"]

face_locations = []
face_names     = []
process_this_frame = True

# ── Main loop ─────────────────────────────────────────────────────────────────
while True:
    ret, frame = video_capture.read()
    if not ret:
        print("Failed to read from webcam")
        break

    small_frame     = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1])
    gray_frame      = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ── Face recognition (every other frame) ──────────────────────────────────
    if process_this_frame:
        face_locations   = face_recognition.face_locations(rgb_small_frame)
        face_enc_list    = face_recognition.face_encodings(rgb_small_frame, face_locations)
        cv2.imwrite(file, frame)
        face_names = []

        for fe in face_enc_list:
            matches    = face_recognition.compare_faces(known_face_encodings, fe)
            name       = "Unknown"
            distances  = face_recognition.face_distance(known_face_encodings, fe)
            best       = int(np.argmin(distances))

            if matches[best]:
                name = known_face_names[best]
                try:
                    direction = eye_game.get_gaze_direction(file)
                    gaze_dir  = direction          # update live label
                    if previous != direction:
                        previous   = direction
                        gaze_count = 0
                    else:
                        gaze_count += 1
                except Exception:
                    pass

            face_names.append(name)

    process_this_frame = not process_this_frame

    # ── EAR / MAR ─────────────────────────────────────────────────────────────
    ear = 0.0
    mar = 0.0

    for (top, right, bottom, left) in face_locations:
        t, r, b, l = top * 4, right * 4, bottom * 4, left * 4
        rect  = dlib.rectangle(l, t, r, b)
        shape = landmark_predictor(gray_frame, rect)
        pts   = np.array([[shape.part(i).x, shape.part(i).y] for i in range(68)])

        le  = eye_aspect_ratio(pts[LEFT_EYE])
        re  = eye_aspect_ratio(pts[RIGHT_EYE])
        ear = (le + re) / 2.0
        mar = mouth_aspect_ratio(pts[MOUTH])

        # EAR alert
        if ear < EAR_THRESH:
            ear_counter += 1
            if ear_counter >= EAR_CONSEC_FRAMES:
                cv2.putText(frame, "ALERT: Eyes Closed!", (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
        else:
            ear_counter = 0

        # MAR alert
        if mar > MAR_THRESH:
            mar_counter += 1
            if mar_counter >= MAR_CONSEC_FRAMES:
                cv2.putText(frame, "ALERT: Yawning!", (10, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 140, 255), 2)
        else:
            mar_counter = 0

    # ── HUD ───────────────────────────────────────────────────────────────────
    ear_color = (0, 0, 255) if ear < EAR_THRESH else (0, 255, 255)
    mar_color = (0, 140, 255) if mar > MAR_THRESH else (0, 255, 255)

    # Gaze direction colour: green=safe, yellow=mild, red=danger
    if gaze_dir in SAFE_GAZES:
        gaze_color = (0, 220, 80)        # green
    elif gaze_dir in MILD_GAZES:
        gaze_color = (0, 200, 255)       # amber/yellow
    else:                                # DANGER_GAZES or unknown
        gaze_color = (0, 60, 255)        # red

    cv2.putText(frame, f"EAR: {ear:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, ear_color, 2)
    cv2.putText(frame, f"MAR: {mar:.2f}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, mar_color, 2)
    cv2.putText(frame, f"Gaze: {gaze_dir.upper()}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.70, gaze_color, 2)
    cv2.putText(frame, f"Gaze fixed: {gaze_count} frames", (10, 118),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 0), 1)

    hint = f"EAR<{EAR_THRESH}=Drowsy  MAR>{MAR_THRESH}=Yawn  Gaze>=10=Alert"
    cv2.putText(frame, hint, (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 160, 160), 1)

    # Gaze distraction alerts
    if gaze_count >= 10 and gaze_dir not in SAFE_GAZES:
        alert_msg = f"ALERT: Gaze Fixed {gaze_dir.upper()}!"
        cv2.putText(frame, alert_msg, (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 0, 255), 2)
    if gaze_dir in DANGER_GAZES and gaze_count >= 5:
        cv2.putText(frame, "ALERT: Looking Away from Road!", (10, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

    # ── Face boxes ────────────────────────────────────────────────────────────
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

video_capture.release()
cv2.destroyAllWindows()