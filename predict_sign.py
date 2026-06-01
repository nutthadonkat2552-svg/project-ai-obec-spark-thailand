import cv2
import numpy as np
import joblib
import os
import time
import traceback
import sys

os.environ["GLOG_minloglevel"] = "3"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"

import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import HolisticLandmarker
from PIL import ImageFont, ImageDraw, Image
import urllib.request

_native_stderr_silenced = False


def silence_native_stderr():
    global _native_stderr_silenced
    if _native_stderr_silenced:
        return

    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull_fd, 2)
    os.close(devnull_fd)
    _native_stderr_silenced = True


THAI_FONT_PATHS = [
    "C:/Windows/Fonts/leelawui.ttf",
    "C:/Windows/Fonts/leelawad.ttf",
    "C:/Windows/Fonts/tahoma.ttf",
    "C:/Windows/Fonts/arialuni.ttf",
    "C:/Windows/Fonts/cordia.ttc",
]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "holistic_landmarker.task")
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "holistic_landmarker/holistic_landmarker/float16/1/holistic_landmarker.task"
)

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading holistic_landmarker.task ...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Download complete.")

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
]

POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
]

EXPECTED_FRAME_FEATURES = 1090
POSE_HAND_START_INDEX = 936


def load_thai_font(size=28):
    for path in THAI_FONT_PATHS:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


_thai_font = load_thai_font(28)


def fix_mojibake_thai(text):
    if not isinstance(text, str) or "à" not in text:
        return text

    try:
        return text.encode("cp1252").decode("utf-8")
    except UnicodeError:
        return text


def draw_text_thai(frame, text, pos=(10, 30), color=(255, 255, 255)):
    img_pil = Image.fromarray(frame)
    draw = ImageDraw.Draw(img_pil)
    x, y = pos

    try:
        bbox = draw.textbbox((x, y), text, font=_thai_font)
        text_size = (bbox[2] - bbox[0], bbox[3] - bbox[1])
    except AttributeError:
        text_size = _thai_font.getsize(text)

    draw.rectangle(
        [x - 5, y - 5, x + text_size[0] + 5, y + text_size[1] + 5],
        fill=(0, 0, 0),
    )
    draw.text(pos, text, font=_thai_font, fill=color)
    return np.array(img_pil)


def prepare_frame_for_mediapipe(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)
    enhanced = cv2.cvtColor(
        cv2.merge((l_channel, a_channel, b_channel)),
        cv2.COLOR_LAB2BGR,
    )

    gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    mean_brightness = float(np.mean(gray))

    if mean_brightness < 95:
        gamma = 0.75
    elif mean_brightness > 175:
        gamma = 1.25
    else:
        gamma = 1.0

    if gamma != 1.0:
        table = np.array([
            ((i / 255.0) ** gamma) * 255
            for i in range(256)
        ], dtype=np.uint8)
        enhanced = cv2.LUT(enhanced, table)

    return enhanced


def draw_landmarks(frame, pose_landmarks, left_landmarks, right_landmarks):
    h, w = frame.shape[:2]

    def to_points(landmarks):
        return [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

    def draw_lines(points, connections, color, thickness=2):
        for a, b in connections:
            if a < len(points) and b < len(points):
                cv2.line(frame, points[a], points[b], color, thickness, cv2.LINE_AA)

    def draw_points(points, color, radius=4):
        for point in points:
            cv2.circle(frame, point, radius, color, -1, cv2.LINE_AA)
            cv2.circle(frame, point, radius, (255, 255, 255), 1, cv2.LINE_AA)

    if pose_landmarks:
        pose_points = to_points(pose_landmarks)
        draw_lines(pose_points, POSE_CONNECTIONS, (0, 200, 255), 2)
        draw_points([pose_points[i] for i in [11, 12, 13, 14, 15, 16, 23, 24] if i < len(pose_points)], (0, 200, 255), 4)

    if left_landmarks:
        left_points = to_points(left_landmarks)
        draw_lines(left_points, HAND_CONNECTIONS, (255, 0, 0), 2)
        draw_points(left_points, (255, 0, 0), 4)

    if right_landmarks:
        right_points = to_points(right_landmarks)
        draw_lines(right_points, HAND_CONNECTIONS, (0, 0, 255), 2)
        draw_points(right_points, (0, 0, 255), 4)

    return frame


def open_webcam():
    backends = [
        ("DirectShow", cv2.CAP_DSHOW),
        ("MSMF", cv2.CAP_MSMF),
        ("Default", cv2.CAP_ANY),
    ]

    for camera_index in range(5):
        for backend_name, backend in backends:
            print(f"Trying webcam index {camera_index} with {backend_name}...")
            cap = cv2.VideoCapture(camera_index, backend)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
            time.sleep(0.5)

            if not cap.isOpened():
                cap.release()
                continue

            for _ in range(10):
                ret, frame = cap.read()
                if ret and frame is not None:
                    print(f"Using webcam index {camera_index} with {backend_name}.")
                    return cap
                time.sleep(0.1)

            cap.release()

    return None


def landmark_to_feature(face_landmarks, pose_landmarks, right_landmarks, left_landmarks):
    features = []

    if face_landmarks:
        face_count = min(len(face_landmarks), 468)
        face_part = face_landmarks[:face_count]
        nose = face_part[1] if len(face_part) > 1 else face_part[0]
        for lm in face_part:
            features.append(lm.x - nose.x)
            features.append(lm.y - nose.y)
        features.extend([0.0] * ((468 - face_count) * 2))
    else:
        features.extend([0.0] * 936)

    if pose_landmarks:
        pose_count = min(len(pose_landmarks), 33)
        pose_part = pose_landmarks[:pose_count]
        nose = pose_part[0]
        for lm in pose_part:
            features.append(lm.x - nose.x)
            features.append(lm.y - nose.y)
        features.extend([0.0] * ((33 - pose_count) * 2))
    else:
        features.extend([0.0] * 66)

    if right_landmarks:
        right_count = min(len(right_landmarks), 21)
        right_part = right_landmarks[:right_count]
        wrist = right_part[0]
        for lm in right_part:
            features.append(lm.x - wrist.x)
            features.append(lm.y - wrist.y)
        features.extend([0.0] * ((21 - right_count) * 2))
    else:
        features.extend([0.0] * 42)

    if left_landmarks:
        left_count = min(len(left_landmarks), 21)
        left_part = left_landmarks[:left_count]
        wrist = left_part[0]
        for lm in left_part:
            features.append(lm.x - wrist.x)
            features.append(lm.y - wrist.y)
        features.extend([0.0] * ((21 - left_count) * 2))
    else:
        features.extend([0.0] * 42)

    if pose_landmarks:
        anchor = pose_landmarks[0]
    elif right_landmarks:
        anchor = right_landmarks[0]
    elif left_landmarks:
        anchor = left_landmarks[0]
    else:
        anchor = None

    for hand_landmarks in (right_landmarks, left_landmarks):
        if hand_landmarks and anchor:
            wrist = hand_landmarks[0]
            features.append(wrist.x - anchor.x)
            features.append(wrist.y - anchor.y)
        else:
            features.extend([0.0, 0.0])

    return features


def summarize_sequence(sequence):
    seq_array = np.array(sequence)
    if seq_array.shape[1] > EXPECTED_FRAME_FEATURES:
        seq_array = seq_array[:, :EXPECTED_FRAME_FEATURES]
    elif seq_array.shape[1] < EXPECTED_FRAME_FEATURES:
        pad_width = EXPECTED_FRAME_FEATURES - seq_array.shape[1]
        seq_array = np.pad(seq_array, ((0, 0), (0, pad_width)), mode="constant")

    seq_array = seq_array[:, POSE_HAND_START_INDEX:]
    thirds = np.array_split(seq_array, 3)
    start_mean = np.mean(thirds[0], axis=0)
    mid_mean = np.mean(thirds[1], axis=0)
    end_mean = np.mean(thirds[2], axis=0)

    return np.concatenate([
        np.mean(seq_array, axis=0),
        np.std(seq_array, axis=0),
        np.max(seq_array, axis=0),
        np.min(seq_array, axis=0),
        start_mean,
        mid_mean,
        end_mean,
        end_mean - start_mean,
        mid_mean - start_mean,
        end_mean - mid_mean,
    ])


def match_feature_size(features, expected_size):
    if expected_size is None or len(features) == expected_size:
        return features
    raise ValueError(
        f"Feature size mismatch. Model expects {expected_size}, "
        f"but current extractor produced {len(features)}."
    )


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    model = joblib.load(os.path.join(base_dir, "sign_language_model.pkl"))
    scaler = joblib.load(os.path.join(base_dir, "scaler.pkl"))
    label_classes = np.array([
        fix_mojibake_thai(label)
        for label in np.load(os.path.join(base_dir, "label_classes.npy"), allow_pickle=True)
    ])

    print("Label classes:", list(label_classes))
    expected_feature_size = getattr(scaler, "n_features_in_", None)
    new_feature_size = (EXPECTED_FRAME_FEATURES - POSE_HAND_START_INDEX) * 10
    model_needs_retrain = expected_feature_size != new_feature_size
    if model_needs_retrain:
        print("WARNING: scaler/model ยังเป็นสูตรเก่า ควรรัน train_model.py ใหม่ก่อนใช้ predict")
        print(f"Expected by scaler: {expected_feature_size} | New feature size: {new_feature_size}")
    print(f"ทดลองใช้ได้ตอนนี้: {', '.join(label_classes)}")

    download_model()
    base_options = mp.tasks.BaseOptions(model_asset_path=MODEL_PATH)
    holistic_options = vision.HolisticLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        min_face_detection_confidence=0.5,
        min_pose_detection_confidence=0.5,
        min_hand_landmarks_confidence=0.5,
        min_face_landmarks_confidence=0.5,
        min_pose_landmarks_confidence=0.5,
    )
    holistic = HolisticLandmarker.create_from_options(holistic_options)
    silence_native_stderr()

    cap = open_webcam()
    if cap is None:
        print("ERROR: Cannot open webcam. Close other camera apps or check Windows camera permissions.")
        return

    sequence = []
    prediction_buffer = []
    window_size = 24
    buffer_size = 5
    min_sequence_len = 10
    reset_when_hand_missing = True
    confidence_threshold = 0.50
    failed_frames = 0

    print("Realtime prediction เริ่มทำงาน: วางมือให้เห็นชัด แล้วระบบจะทำนายอัตโนมัติ")
    print("กด Q เพื่อออก, กด C เพื่อล้าง sequence ปัจจุบัน")

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            failed_frames += 1
            print(f"WARNING: Failed to read frame ({failed_frames}/30)")
            if failed_frames >= 30:
                print("ERROR: Webcam stopped responding.")
                break
            time.sleep(0.05)
            continue

        failed_frames = 0

        mediapipe_frame = prepare_frame_for_mediapipe(frame)
        image = cv2.cvtColor(mediapipe_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        results = holistic.detect(mp_image)

        face_lms = results.face_landmarks if results.face_landmarks else []
        pose_lms = results.pose_landmarks if results.pose_landmarks else []
        left_lms = results.left_hand_landmarks if results.left_hand_landmarks else []
        right_lms = results.right_hand_landmarks if results.right_hand_landmarks else []
        has_hand = bool(left_lms or right_lms)

        if has_hand:
            features = landmark_to_feature(face_lms, pose_lms, right_lms, left_lms)
            sequence.append(features)
            if len(sequence) > window_size:
                sequence = sequence[-window_size:]
        else:
            if reset_when_hand_missing:
                sequence.clear()
                prediction_buffer.clear()

        frame = draw_landmarks(frame, pose_lms, left_lms, right_lms)

        predicted_text = "กรุณาวางมือให้กล้องเห็นชัด" if len(sequence) < min_sequence_len else "กำลังทำนาย..."
        confidence_text = ""

        if model_needs_retrain:
            predicted_text = "ต้องรัน train_model.py ใหม่ก่อน"
            confidence_text = f"โมเดลเก่า: {expected_feature_size} | สูตรใหม่: {new_feature_size}"
        elif len(sequence) >= min_sequence_len:
            pred_features = summarize_sequence(sequence)
            pred_features = match_feature_size(pred_features, expected_feature_size)
            pred_features = scaler.transform([pred_features])

            pred_proba = model.predict_proba(pred_features)[0]
            prediction_buffer.append(pred_proba)
            if len(prediction_buffer) > buffer_size:
                prediction_buffer = prediction_buffer[-buffer_size:]

            avg_proba = np.mean(prediction_buffer, axis=0)
            avg_idx = int(np.argmax(avg_proba))
            avg_label = label_classes[avg_idx]
            avg_confidence = float(avg_proba[avg_idx])

            prob_str = "  ".join([
                f"{label}:{p:.0%}" for label, p in zip(label_classes, avg_proba)
            ])

            if avg_confidence >= confidence_threshold:
                predicted_text = f"ท่าที่ทำนาย: {avg_label}"
            else:
                predicted_text = "ไม่มั่นใจ - ทำท่าใหม่ให้ชัด"
            confidence_text = f"ความมั่นใจ: {avg_confidence:.0%} | {prob_str}"

        frame = draw_text_thai(frame, predicted_text, pos=(10, 10), color=(220, 220, 220))
        if confidence_text:
            frame = draw_text_thai(frame, confidence_text, pos=(10, 46), color=(220, 220, 0))
        frame = draw_text_thai(
            frame,
            f"โหมดลองใช้: {', '.join(label_classes)}",
            pos=(10, 82),
            color=(80, 220, 255),
        )

        cv2.imshow("Sign Language Prediction", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("c"):
            sequence.clear()
            prediction_buffer.clear()
            print("ล้าง sequence ปัจจุบันแล้ว")

    cap.release()
    cv2.destroyAllWindows()
    holistic.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc(file=sys.stdout)
        input("Program crashed. Press Enter to close...")
