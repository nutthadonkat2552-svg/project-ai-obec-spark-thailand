import json
import os
from collections import Counter

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "pose")
DATA_CANDIDATES = [
    os.path.join(BASE_DIR, "thai_holistic_motion_data.json"),
    os.path.join(BASE_DIR, "pose", "thai_holistic_motion_data.json"),
]

EXPECTED_FRAME_FEATURES = 1090
POSE_HAND_START_INDEX = 936
MIN_FRAMES_PER_SEQUENCE = 12
TEST_SIZE = 0.2
RANDOM_STATE = 42


def fix_mojibake_thai(text):
    if not isinstance(text, str) or "à" not in text:
        return text

    try:
        return text.encode("cp1252").decode("utf-8")
    except UnicodeError:
        return text


def find_data_paths():
    if os.path.isdir(DATA_DIR):
        split_files = [
            os.path.join(DATA_DIR, name)
            for name in os.listdir(DATA_DIR)
            if name.lower().endswith(".json") and name != "thai_holistic_motion_data.json"
        ]
        if split_files:
            return sorted(split_files)

    existing = [path for path in DATA_CANDIDATES if os.path.exists(path)]
    if not existing:
        raise FileNotFoundError("ไม่พบไฟล์ข้อมูล JSON ใน pose หรือ thai_holistic_motion_data.json")

    return [max(existing, key=os.path.getmtime)]


def normalize_frame(frame):
    frame = np.asarray(frame, dtype=np.float32).reshape(-1)

    if len(frame) > EXPECTED_FRAME_FEATURES:
        return frame[:EXPECTED_FRAME_FEATURES]
    if len(frame) < EXPECTED_FRAME_FEATURES:
        return np.pad(frame, (0, EXPECTED_FRAME_FEATURES - len(frame)), mode="constant")
    return frame


def sequence_to_features(sequence):
    clean_frames = [normalize_frame(frame) for frame in sequence]
    seq_array = np.asarray(clean_frames, dtype=np.float32)[:, POSE_HAND_START_INDEX:]
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


def load_dataset(data_paths):
    x = []
    y = []
    skipped = 0
    total = 0

    for data_path in data_paths:
        file_label = os.path.splitext(os.path.basename(data_path))[0]
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        total += len(data)
        for item in data:
            label = fix_mojibake_thai(item.get("label") or file_label)
            sequence = item.get("sequence", [])

            if not label or len(sequence) < MIN_FRAMES_PER_SEQUENCE:
                skipped += 1
                continue

            try:
                x.append(sequence_to_features(sequence))
                y.append(label)
            except (TypeError, ValueError):
                skipped += 1

    return np.asarray(x, dtype=np.float32), np.asarray(y), skipped, total


def main():
    print("กำลังโหลดข้อมูล...")
    data_paths = find_data_paths()
    print("ใช้ไฟล์ข้อมูล:")
    for path in data_paths:
        print(f"  {path}")

    x, y, skipped, total = load_dataset(data_paths)
    if len(x) == 0:
        raise RuntimeError("ไม่มี sequence ที่ใช้ train ได้")

    label_counts = Counter(y)
    print(f"โหลดข้อมูลทั้งหมด {total} sequences | ใช้ train {len(x)} | ข้าม {skipped}")
    print("จำนวนข้อมูลต่อท่า:")
    for label, count in sorted(label_counts.items()):
        print(f"  {label}: {count}")

    if len(label_counts) < 2:
        raise RuntimeError("ต้องมีอย่างน้อย 2 ท่าเพื่อ train model")

    too_few = [label for label, count in label_counts.items() if count < 2]
    if too_few:
        raise RuntimeError(f"ท่าที่มีข้อมูลน้อยเกินไปสำหรับแบ่ง train/test: {too_few}")

    print(f"Feature shape: {x.shape}")

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    stratify = y_encoded if min(label_counts.values()) >= 2 else None
    x_train, x_test, y_train, y_test = train_test_split(
        x_scaled,
        y_encoded,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=stratify,
    )

    print("กำลังเทรนโมเดล Random Forest...")
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=1,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)

    train_score = model.score(x_train, y_train)
    test_score = model.score(x_test, y_test)

    print("เทรนโมเดลเสร็จแล้ว")
    print(f"Train Accuracy: {train_score:.2%}")
    print(f"Test Accuracy: {test_score:.2%}")
    print(f"Classes: {label_encoder.classes_}")
    print("Classification report:")
    print(classification_report(y_test, model.predict(x_test), target_names=label_encoder.classes_))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, model.predict(x_test)))

    joblib.dump(model, os.path.join(BASE_DIR, "sign_language_model.pkl"))
    joblib.dump(scaler, os.path.join(BASE_DIR, "scaler.pkl"))
    np.save(os.path.join(BASE_DIR, "label_classes.npy"), label_encoder.classes_)

    print("บันทึกโมเดลเสร็จแล้ว:")
    print(f"  {os.path.join(BASE_DIR, 'sign_language_model.pkl')}")
    print(f"  {os.path.join(BASE_DIR, 'scaler.pkl')}")
    print(f"  {os.path.join(BASE_DIR, 'label_classes.npy')}")


if __name__ == "__main__":
    main()
