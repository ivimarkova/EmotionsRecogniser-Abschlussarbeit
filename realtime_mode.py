"""
================================================
  realtime_mode.py - Real-Time Webcam Analysis
  Emotion Recognition System
  Abschlussarbeit - Ivayla Markova
================================================

Approach 1: Real-Time Emotion Recognition
- Opens the webcam
- Detects faces using Haar Cascade
- Detects eyes and smile using Haar Cascade
- Classifies emotion using rule-based heuristics
- Displays emotion as TEXT on a side panel
  (replaces the old animation images)
- Press Q to quit, S for screenshot
"""

import cv2
import numpy as np
import os

# ── Load Haar Cascades ────────────────────────────────────────────
# Pre-trained XML classifiers provided by OpenCV.
# They detect patterns (face, eyes, smile) in grayscale images.
face_cascade  = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade   = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

# ── Emotion colors (BGR format) ───────────────────────────────────
EMOTION_COLORS = {
    'Happy':     (0,   220,   0),
    'Neutral':   (200, 200, 200),
    'Surprised': (0,   200, 255),
    'Sad':       (200, 100,  50),
    'Angry':     (0,     0, 220),
    'Fear':      (130,   0, 180),
    'Disgust':   (0,   140, 255),
    'No Face':   (80,   80,  80),
}

# Short description shown in the side panel for each emotion
EMOTION_DESCRIPTIONS = {
    'Happy':     ["Smile detected", "Eyes open"],
    'Neutral':   ["No smile detected", "Eyes open"],
    'Surprised': ["Mouth open", "Eyes wide"],
    'Sad':       ["Eyes not visible", "Low brightness"],
    'Fear':      ["Eyes closed", "Mouth open"],
    'Disgust':   ["Mouth tension", "detected"],
    'No Face':   ["No face in", "camera frame"],
}


def detect_emotion(gray_frame, x, y, w, h):
    """
    Rule-based emotion classification using Haar Cascade features.

    Rules applied in order:
      2 eyes + smile detected       -> Happy
      2 eyes + open mouth           -> Surprised
      2 eyes + no smile             -> Neutral
      1 eye visible                 -> Neutral
      0 eyes + open mouth           -> Fear
      0 eyes                        -> Sad

    Parameters:
        gray_frame : full grayscale image
        x, y, w, h : face bounding box

    Returns:
        emotion (str), color (BGR tuple),
        eyes (detections), smiles (detections)
    """
    roi_gray = gray_frame[y:y+h, x:x+w]

    # Detect eyes inside the face region
    eyes = eye_cascade.detectMultiScale(
        roi_gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(20, 20)
    )

    # Detect smile only in the LOWER half of the face
    # (prevents false positives from nose/forehead)
    lower_half = roi_gray[h // 2:, :]
    smiles = smile_cascade.detectMultiScale(
        lower_half,
        scaleFactor=1.8,
        minNeighbors=20,
        minSize=(25, 25)
    )

    # Estimate if mouth is open by checking darkness in the mouth area
    mouth_region = roi_gray[int(h * 0.65):h, int(w * 0.25):int(w * 0.75)]
    mouth_dark_ratio = 0.0
    if mouth_region.size > 0:
        dark_pixels = (mouth_region < 80).sum()
        mouth_dark_ratio = dark_pixels / mouth_region.size

    # Apply heuristic rules
    eye_count     = len(eyes)
    smile_found   = len(smiles) > 0

    if eye_count >= 2:
        if smile_found:
            emotion = 'Happy'
        elif mouth_dark_ratio > 0.15:
            emotion = 'Surprised'
        else:
            emotion = 'Neutral'
    elif eye_count == 1:
        emotion = 'Neutral'
    else:
        emotion = 'Fear' if mouth_dark_ratio > 0.15 else 'Sad'

    color = EMOTION_COLORS.get(emotion, (200, 200, 200))
    return emotion, color, eyes, smiles


def create_text_panel(current_emotion, panel_width, panel_height):
    """
    Create the right-side panel showing the detected emotion as text.
    This replaces the old animation image panel from the original code.
    """
    panel = np.full((panel_height, panel_width, 3), 30, dtype=np.uint8)

    # Header background
    cv2.rectangle(panel, (0, 0), (panel_width, 90), (50, 50, 50), -1)

    cv2.putText(panel, "EMOTION DETECTED:",
                (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (255, 255, 255), 1)

    # Emotion name in large colored text
    emo_color = EMOTION_COLORS.get(current_emotion, (200, 200, 200))
    cv2.putText(panel, current_emotion,
                (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.1, emo_color, 3)

    # Divider
    cv2.line(panel, (10, 100), (panel_width - 10, 100), (80, 80, 80), 1)

    # Description lines
    desc_lines = EMOTION_DESCRIPTIONS.get(current_emotion, [])
    for i, line in enumerate(desc_lines):
        cv2.putText(panel, line,
                    (10, 135 + i * 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (180, 180, 180), 1)

    return panel


def run_realtime():
    """
    Approach 1 — Real-Time Webcam Mode.
    Directly based on the original recogniser.py loop.
    """
    if not os.path.exists('screenshots'):
        os.makedirs('screenshots')

    # Open webcam — 0 = default built-in camera
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] Cannot access webcam!")
        return

    print("\n" + "="*50)
    print("  Approach 1: Real-Time Webcam Mode")
    print("="*50)
    print("  [Q] Quit   [S] Screenshot")
    print("="*50 + "\n")

    frame_count = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            print("[WARN] Skipped frame, retrying...")
            continue

        frame_count += 1

        # Mirror effect (like a selfie camera)
        frame = cv2.flip(frame, 1)
        frame_height, frame_width = frame.shape[:2]

        # Grayscale needed for Haar Cascade
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(100, 100)
        )

        current_emotion = 'No Face'

        if len(faces) > 0:
            for (x, y, w, h) in faces:
                emotion, color, eyes, smiles = detect_emotion(gray, x, y, w, h)
                current_emotion = emotion

                # Face rectangle
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)

                # Eye rectangles (drawn inside face ROI)
                roi_color = frame[y:y+h, x:x+w]
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(roi_color,
                                  (ex, ey), (ex+ew, ey+eh),
                                  (0, 255, 0), 2)

                # Smile rectangles (lower half offset)
                for (sx, sy, sw, sh) in smiles:
                    cv2.rectangle(roi_color,
                                  (sx, h//2 + sy),
                                  (sx+sw, h//2 + sy+sh),
                                  (255, 255, 0), 2)

                # Emotion label above face box
                label_size, _ = cv2.getTextSize(
                    emotion, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
                )
                cv2.rectangle(frame,
                              (x, y - 35),
                              (x + label_size[0] + 10, y),
                              color, -1)
                cv2.putText(frame, emotion,
                            (x + 5, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 0, 0), 2)
        else:
            cv2.putText(frame, "No face detected",
                        (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 0, 255), 2)

        # Build side panel and combine with frame
        panel = create_text_panel(current_emotion, 280, frame_height)
        combined = np.hstack((frame, panel))

        # Bottom info bar
        h_c, w_c = combined.shape[:2]
        cv2.rectangle(combined, (0, h_c - 40), (w_c, h_c), (0, 0, 0), -1)
        cv2.putText(combined,
                    f"Frame: {frame_count}  |  Faces: {len(faces)}  |  [Q] Quit  [S] Screenshot",
                    (10, h_c - 12),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (200, 200, 200), 1)

        cv2.imshow("Emotion Recognition - Real-Time", combined)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\n[INFO] Quitting real-time mode...")
            break
        elif key == ord('s'):
            fname = f"screenshots/realtime_{frame_count:04d}.jpg"
            cv2.imwrite(fname, combined)
            print(f"[INFO] Screenshot saved: {fname}")

    cap.release()
    cv2.destroyAllWindows()
    print(f"[INFO] Total frames processed: {frame_count}\n")