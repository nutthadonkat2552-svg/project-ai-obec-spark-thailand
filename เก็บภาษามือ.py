import cv2
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
import json
from PIL import ImageFont, ImageDraw, Image
import numpy as np
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


# Model setup
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


# Configuration
labels = ["สวัสดี", "ขอบคุณ", "ขอโทษ", "ใช่", "ไม่", "ฉัน", "พูดไม่ได้"]
MAX_SEQUENCES = None
output_file = os.path.join(BASE_DIR, "thai_holistic_motion_data.json")
DATA_DIR = os.path.join(BASE_DIR, "pose")
FACE_LANDMARK_COUNT = 468
POSE_LANDMARK_COUNT = 33
HAND_LANDMARK_COUNT = 21
HAND_POSITION_FEATURES = 4
MIN_SAVE_FRAMES = 12
REQUIRE_HAND_FOR_RECORDING = True


# Hand skeleton style
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
]

POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    (11, 12),
    (11, 13), (13, 15),
    (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16),
    (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (24, 26),
    (25, 27), (26, 28),
    (27, 29), (28, 30),
    (29, 31), (30, 32),
    (27, 31), (28, 32),
]

# BGR colors per finger: thumb, index, middle, ring, pinky
FINGER_COLORS = [
    (117, 158, 29),
    (221, 138, 55),
    (74, 75, 226),
    (48, 90, 216),
    (126, 83, 212),
]


def finger_idx(connection_i):
    if connection_i < 4:
        return 0
    if connection_i < 8:
        return 1
    if connection_i < 12:
        return 2
    if connection_i < 16:
        return 3
    return 4


def get_sequence_count(data, label):
    return sum(1 for d in data if d["label"] == label)


def fix_mojibake_thai(text):
    if not isinstance(text, str) or "à" not in text:
        return text

    try:
        return text.encode("cp1252").decode("utf-8")
    except UnicodeError:
        return text


def safe_filename(label):
    invalid_chars = '<>:"/\\|?*'
    name = "".join("_" if ch in invalid_chars else ch for ch in label).strip()
    return f"{name}.json"


def label_data_path(label):
    return os.path.join(DATA_DIR, safe_filename(label))


def load_json_file(path, default):
    if not os.path.exists(path):
        return default

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_label_data(label, sequences):
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(label_data_path(label), "w", encoding="utf-8") as f:
        json.dump(sequences, f, ensure_ascii=False)


def save_combined_data(data_by_label):
    combined = []
    for label in labels:
        combined.extend(data_by_label.get(label, []))

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(combined, f, ensure_ascii=False)


def load_split_data():
    os.makedirs(DATA_DIR, exist_ok=True)
    data_by_label = {label: load_json_file(label_data_path(label), []) for label in labels}

    has_split_data = any(data_by_label[label] for label in labels)
    if has_split_data or not os.path.exists(output_file):
        return data_by_label

    print("พบไฟล์ข้อมูลรวมเก่า กำลังแยกข้อมูลไปที่โฟลเดอร์ pose ...")
    legacy_data = load_json_file(output_file, [])
    for item in legacy_data:
        label = fix_mojibake_thai(item.get("label", ""))
        if label in data_by_label:
            item["label"] = label
            data_by_label[label].append(item)

    for label in labels:
        save_label_data(label, data_by_label[label])
    save_combined_data(data_by_label)

    print("แยกข้อมูลเก่าเสร็จแล้ว ไฟล์รวมเดิมยังถูกเก็บไว้เหมือนเดิม")
    return data_by_label


def build_label_counts(data_by_label):
    return {label: len(data_by_label.get(label, [])) for label in labels}


def append_relative_landmarks(features, landmarks, count, anchor_index):
    if not landmarks:
        features.extend([0.0] * (count * 2))
        return

    anchor = landmarks[anchor_index] if len(landmarks) > anchor_index else landmarks[0]
    for i in range(count):
        if i < len(landmarks):
            lm = landmarks[i]
            features.append(lm.x - anchor.x)
            features.append(lm.y - anchor.y)
        else:
            features.extend([0.0, 0.0])


def append_hand_position_features(features, pose_landmarks, right_landmarks, left_landmarks):
    if pose_landmarks:
        anchor = pose_landmarks[0]
    elif right_landmarks:
        anchor = right_landmarks[0]
    elif left_landmarks:
        anchor = left_landmarks[0]
    else:
        features.extend([0.0] * HAND_POSITION_FEATURES)
        return

    for hand_landmarks in (right_landmarks, left_landmarks):
        if hand_landmarks:
            wrist = hand_landmarks[0]
            features.append(wrist.x - anchor.x)
            features.append(wrist.y - anchor.y)
        else:
            features.extend([0.0, 0.0])


def landmark_to_feature(face_landmarks, pose_landmarks, right_landmarks, left_landmarks):
    features = []

    append_relative_landmarks(features, face_landmarks, FACE_LANDMARK_COUNT, 1)
    append_relative_landmarks(features, pose_landmarks, POSE_LANDMARK_COUNT, 0)
    append_relative_landmarks(features, right_landmarks, HAND_LANDMARK_COUNT, 0)
    append_relative_landmarks(features, left_landmarks, HAND_LANDMARK_COUNT, 0)
    append_hand_position_features(features, pose_landmarks, right_landmarks, left_landmarks)

    return features


def load_thai_font(size=32):
    font_candidates = [
        "C:/Windows/Fonts/arialuni.ttf",
        "C:/Windows/Fonts/tahoma.ttf",
        "C:/Windows/Fonts/leelawad.ttf",
        "C:/Windows/Fonts/leelawui.ttf",
        "C:/Windows/Fonts/cordia.ttc",
        "C:/Windows/Fonts/upcil.ttf",
        "/usr/share/fonts/truetype/tlwg/Garuda.ttf",
    ]
    for path in font_candidates:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


_thai_fonts = {}


def draw_text_thai(frame, text, pos=(10, 30), color=(255, 255, 255), size=28, bg_alpha=95):
    if size not in _thai_fonts:
        _thai_fonts[size] = load_thai_font(size)

    img_pil = Image.fromarray(frame).convert("RGBA")
    overlay = Image.new("RGBA", img_pil.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    x, y = pos

    try:
        bbox = draw.textbbox((x, y), text, font=_thai_fonts[size])
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
    except AttributeError:
        text_w, text_h = _thai_fonts[size].getsize(text)

    padding_x = 5
    padding_y = 3
    draw.rectangle(
        [x - padding_x, y - padding_y, x + text_w + padding_x, y + text_h + padding_y],
        fill=(0, 0, 0, bg_alpha),
    )
    draw.text(pos, text, font=_thai_fonts[size], fill=color + (255,))
    result = np.array(Image.alpha_composite(img_pil, overlay).convert("RGB"))
    np.copyto(frame, result)
    return frame


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


def draw_hand_skeleton(frame, landmarks, label):
    if not landmarks:
        return frame

    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

    for i, (a, b) in enumerate(HAND_CONNECTIONS):
        if a < len(pts) and b < len(pts):
            cv2.line(
                frame,
                pts[a],
                pts[b],
                FINGER_COLORS[finger_idx(i)],
                2,
                cv2.LINE_AA,
            )

    for i, (x, y) in enumerate(pts):
        color = FINGER_COLORS[min(i // 4, 4)]
        radius = 7 if i == 0 else 5
        cv2.circle(frame, (x, y), radius, color, -1, cv2.LINE_AA)
        cv2.circle(frame, (x, y), radius, (255, 255, 255), 1, cv2.LINE_AA)

    wx, wy = pts[0]
    cv2.putText(
        frame,
        label,
        (wx + 10, wy - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return frame


def draw_pose_skeleton(frame, landmarks):
    if not landmarks:
        return frame

    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

    for a, b in POSE_CONNECTIONS:
        if a < len(pts) and b < len(pts):
            cv2.line(frame, pts[a], pts[b], (0, 165, 255), 2, cv2.LINE_AA)

    for x, y in pts:
        cv2.circle(frame, (x, y), 4, (0, 165, 255), -1, cv2.LINE_AA)

    return frame


def draw_face_points(frame, landmarks):
    if not landmarks:
        return frame

    h, w = frame.shape[:2]
    for lm in landmarks:
        x, y = int(lm.x * w), int(lm.y * h)
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1, cv2.LINE_AA)

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


def main():
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

    data_by_label = load_split_data()
    label_counts = build_label_counts(data_by_label)

    current_index = 0
    current_label = labels[current_index]
    recording = False
    motion_sequence = []
    skipped_frames = 0
    last_saved_label = None

    cap = open_webcam()
    if cap is None:
        print("ERROR: Cannot open webcam. Close other camera apps or check Windows camera permissions.")
        return

    failed_frames = 0
    print("กด 1-7 เลือกท่า, A/D เลื่อนท่า, R หรือ Space เริ่มอัด, S หรือ Space เซฟ, C ยกเลิกคลิปนี้, U ลบ sequence ล่าสุด, Q ออก")
    print(f"ข้อมูลจะแยกไฟล์ตามท่าไว้ในโฟลเดอร์: {DATA_DIR}")
    print(f"และจะอัปเดตไฟล์รวมไว้ที่: {output_file}")

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

    # โค้ดประมวลผลภาพต่อจากตรงนี้
        mediapipe_frame = prepare_frame_for_mediapipe(frame)
        image = cv2.cvtColor(mediapipe_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        results = holistic.detect(mp_image)

        face_lms = results.face_landmarks if results.face_landmarks else []
        pose_lms = results.pose_landmarks if results.pose_landmarks else []
        left_lms = results.left_hand_landmarks if results.left_hand_landmarks else []
        right_lms = results.right_hand_landmarks if results.right_hand_landmarks else []
        has_hand = bool(left_lms or right_lms)

        if recording:
            if has_hand or not REQUIRE_HAND_FOR_RECORDING:
                features = landmark_to_feature(face_lms, pose_lms, right_lms, left_lms)
                motion_sequence.append(features)
            else:
                skipped_frames += 1

        count = label_counts.get(current_label, 0)
        status = "กำลังบันทึก" if recording else "ไม่ได้บันทึก"
        hand_status = "เห็นมือ" if has_hand else "ไม่เห็นมือ"
        counter_color = (255, 255, 255)

        line1 = f"ท่า: {current_label}  สถานะ: {status}  เฟรมดี: {len(motion_sequence)}"
        line2 = f"บันทึกแล้ว: {count} sequences | {hand_status} | ข้าม: {skipped_frames}"

        frame = draw_text_thai(frame, line1, pos=(10, 10), color=(255, 255, 255), size=24)
        frame = draw_text_thai(frame, line2, pos=(10, 38), color=counter_color, size=22)

        y = frame.shape[0] - (len(labels) * 26) - 8
        for i, lbl in enumerate(labels):
            c = label_counts.get(lbl, 0)
            frame = draw_text_thai(
                frame,
                f"{i + 1}. {lbl}: {c} sequences",
                pos=(10, y + i * 26),
                color=(255, 255, 255),
                size=20,
            )

        draw_face_points(frame, face_lms)
        draw_pose_skeleton(frame, pose_lms)
        draw_hand_skeleton(frame, left_lms, "Left")
        draw_hand_skeleton(frame, right_lms, "Right")

        cv2.imshow("Collect Thai Holistic Motion Data", frame)

        key = cv2.waitKey(1) & 0xFF
        key_char = chr(key).lower() if key != 255 and key < 128 else ""

        if key_char == "q":
            break
        elif key in (ord("1"), ord("2"), ord("3"), ord("4"), ord("5"), ord("6"), ord("7")):
            current_index = key - ord("1")
            current_label = labels[current_index]
            recording = False
            motion_sequence = []
            skipped_frames = 0
        elif key_char == "a":
            current_index = (current_index - 1) % len(labels)
            current_label = labels[current_index]
            recording = False
            motion_sequence = []
            skipped_frames = 0
        elif key_char == "d":
            current_index = (current_index + 1) % len(labels)
            current_label = labels[current_index]
            recording = False
            motion_sequence = []
            skipped_frames = 0
        elif key_char == "r" or (key == ord(" ") and not recording):
            recording = True
            motion_sequence = []
            skipped_frames = 0
            print(f"เริ่มบันทึก holistic motion สำหรับ: {current_label} ({count + 1} sequences เดิม)")
        elif key_char == "s" or (key == ord(" ") and recording):
            if recording and len(motion_sequence) >= MIN_SAVE_FRAMES:
                data_by_label[current_label].append({"sequence": motion_sequence, "label": current_label})
                label_counts[current_label] = label_counts.get(current_label, 0) + 1
                save_label_data(current_label, data_by_label[current_label])
                save_combined_data(data_by_label)
                last_saved_label = current_label

                count = label_counts.get(current_label, 0)
                print(f"บันทึกสำเร็จ: {current_label} ({count}/{MAX_SEQUENCES}) | frames: {len(motion_sequence)} | skipped: {skipped_frames}")
                recording = False
                motion_sequence = []
                skipped_frames = 0

                if MAX_SEQUENCES and count >= MAX_SEQUENCES:
                    print(f'ท่า "{current_label}" ครบ {MAX_SEQUENCES} sequences แล้ว!')
            elif recording:
                print(f"ยังไม่เซฟ: เฟรมดีมีแค่ {len(motion_sequence)} ต้องมีอย่างน้อย {MIN_SAVE_FRAMES} เฟรม")
        elif key_char == "c":
            if recording or motion_sequence:
                recording = False
                motion_sequence = []
                skipped_frames = 0
                print("ยกเลิก sequence ที่กำลังอัดอยู่")
        elif key_char == "u":
            if recording:
                recording = False
                motion_sequence = []
                skipped_frames = 0
                print("ยกเลิกการอัด sequence ที่กำลังบันทึกอยู่")
            elif data_by_label.get(last_saved_label or current_label, []):
                all_data = data_by_label[last_saved_label or current_label]
                removed = all_data.pop()
                removed_label = removed.get("label", "unknown")
                if removed_label in label_counts:
                    label_counts[removed_label] = max(0, label_counts[removed_label] - 1)
                save_label_data(removed_label, data_by_label[removed_label])
                save_combined_data(data_by_label)
                last_saved_label = None

                removed_frames = len(removed.get("sequence", []))
                count = label_counts.get(removed_label, 0)
                print(f"ลบ sequence ล่าสุดแล้ว: {removed_label} | frames: {removed_frames} | เหลือ: {count} sequences")
            else:
                print("ยังไม่มี sequence ให้ลบ")

    cap.release()
    cv2.destroyAllWindows()
    holistic.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc(file=sys.stdout)
        input("Program crashed. Press Enter to close...")
