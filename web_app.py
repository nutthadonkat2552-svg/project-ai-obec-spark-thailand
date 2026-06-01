import base64
import os
import socket
import threading
import traceback
import argparse

os.environ["GLOG_minloglevel"] = "3"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"

import cv2
import joblib
import mediapipe as mp
import numpy as np
from flask import Flask, jsonify, render_template, request
from werkzeug.serving import make_server

from predict_sign import (
    EXPECTED_FRAME_FEATURES,
    POSE_HAND_START_INDEX,
    fix_mojibake_thai,
    landmark_to_feature,
    match_feature_size,
    silence_native_stderr,
    summarize_sequence,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WINDOW_SIZE = 24
BUFFER_SIZE = 5
MIN_SEQUENCE_LEN = 10
CONFIDENCE_THRESHOLD = 0.40

app = Flask(__name__)

model = joblib.load(os.path.join(BASE_DIR, "sign_language_model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
label_classes = np.array([
    fix_mojibake_thai(label)
    for label in np.load(os.path.join(BASE_DIR, "label_classes.npy"), allow_pickle=True)
])
expected_feature_size = getattr(scaler, "n_features_in_", None)
new_feature_size = (EXPECTED_FRAME_FEATURES - POSE_HAND_START_INDEX) * 10
model_needs_retrain = expected_feature_size != new_feature_size

silence_native_stderr()
holistic = mp.solutions.holistic.Holistic(
    static_image_mode=False,
    model_complexity=0,
    refine_face_landmarks=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

sequence = []
prediction_buffer = []
state_lock = threading.Lock()

# --- เพิ่มตรงนี้: ตัวแปรควบคุมการเปิด/ปิดการแทรกมือ (True = เปิด, False = ปิด) ---
hand_tracking_enabled = True 


def decode_frame(data_url):
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]

    frame_bytes = base64.b64decode(data_url)
    np_buffer = np.frombuffer(frame_bytes, np.uint8)
    frame = cv2.imdecode(np_buffer, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("ไม่สามารถอ่านภาพจากกล้องได้")
    return frame


def reset_state():
    sequence.clear()
    prediction_buffer.clear()


def serialize_landmarks(landmarks):
    return [{"x": float(lm.x), "y": float(lm.y)} for lm in landmarks]


@app.route("/")
def index():
    return render_template("index.html")


# --- เพิ่มตรงนี้: API สำหรับสลับสถานะ เปิด/ปิด หรือเช็คสถานะปัจจุบัน ---
@app.route("/api/toggle_hand", methods=["GET", "POST"])
def toggle_hand():
    global hand_tracking_enabled
    with state_lock:
        if request.method == "POST":
            payload = request.get_json(force=True) or {}
            # สามารถส่ง {"enabled": true/false} มากำหนดได้ หรือถ้าไม่ส่งมาจะเป็นการสลับค่า (Toggle)
            if "enabled" in payload:
                hand_tracking_enabled = bool(payload["enabled"])
            else:
                hand_tracking_enabled = not hand_tracking_enabled
            
            if not hand_tracking_enabled:
                reset_state()  # ล้างข้อมูลค้างเก่าทันทีเมื่อปิด
                
        return jsonify({
            "hand_tracking_enabled": hand_tracking_enabled,
            "status_text": "เปิดใช้งาน" if hand_tracking_enabled else "ปิดใช้งาน"
        })


@app.post("/api/predict")
def predict():
    try:
        payload = request.get_json(force=True)
        frame = decode_frame(payload.get("image", ""))
        include_landmarks = bool(payload.get("includeLandmarks", True))

        # --- เพิ่มตรงนี้: ถ้าสั่งปิดระบบแทรกมือ ให้ข้ามขั้นตอนส่งกลับค่าว่างทันที ---
        if not hand_tracking_enabled:
            with state_lock:
                reset_state()
            return jsonify({
                "label": "ระบบแทรกมือถูกปิดใช้งานอยู่",
                "confidence": 0,
                "status": "disabled",
                "sequenceLength": 0,
                "probabilities": {},
                "labels": label_classes.tolist(),
                "landmarks": {"leftHand": [], "rightHand": [], "pose": []},
            })
        # -------------------------------------------------------------

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = holistic.process(image)

        face_lms = results.face_landmarks.landmark if results.face_landmarks else []
        pose_lms = results.pose_landmarks.landmark if results.pose_landmarks else []
        left_lms = results.left_hand_landmarks.landmark if results.left_hand_landmarks else []
        right_lms = results.right_hand_landmarks.landmark if results.right_hand_landmarks else []
        has_hand = bool(left_lms or right_lms)
        landmarks = {
            "leftHand": serialize_landmarks(left_lms) if include_landmarks else [],
            "rightHand": serialize_landmarks(right_lms) if include_landmarks else [],
            "pose": serialize_landmarks(pose_lms) if include_landmarks else [],
        }

        with state_lock:
            if not has_hand:
                reset_state()
                return jsonify({
                    "label": "กรุณาวางมือให้กล้องเห็นชัด",
                    "confidence": 0,
                    "status": "waiting",
                    "sequenceLength": 0,
                    "probabilities": {},
                    "labels": label_classes.tolist(),
                    "landmarks": landmarks,
                })

            features = landmark_to_feature(face_lms, pose_lms, right_lms, left_lms)
            sequence.append(features)
            if len(sequence) > WINDOW_SIZE:
                del sequence[:-WINDOW_SIZE]

            if model_needs_retrain:
                return jsonify({
                    "label": "ต้องรัน train_model.py ใหม่ก่อน",
                    "confidence": 0,
                    "status": "model_mismatch",
                    "sequenceLength": len(sequence),
                    "message": f"โมเดลเก่า: {expected_feature_size} | สูตรใหม่: {new_feature_size}",
                    "probabilities": {},
                    "labels": label_classes.tolist(),
                    "landmarks": landmarks,
                })

            if len(sequence) < MIN_SEQUENCE_LEN:
                return jsonify({
                    "label": "กำลังทำนาย...",
                    "confidence": 0,
                    "status": "collecting",
                    "sequenceLength": len(sequence),
                    "probabilities": {},
                    "labels": label_classes.tolist(),
                    "landmarks": landmarks,
                })

            pred_features = summarize_sequence(sequence)
            pred_features = match_feature_size(pred_features, expected_feature_size)
            pred_features = scaler.transform([pred_features])

            pred_proba = model.predict_proba(pred_features)[0]
            prediction_buffer.append(pred_proba)
            if len(prediction_buffer) > BUFFER_SIZE:
                del prediction_buffer[:-BUFFER_SIZE]

            avg_proba = np.mean(prediction_buffer, axis=0)
            avg_idx = int(np.argmax(avg_proba))
            avg_label = str(label_classes[avg_idx])
            avg_confidence = float(avg_proba[avg_idx])
            confidence_percent = int(avg_confidence * 100 + 0.5)
            threshold_percent = int(CONFIDENCE_THRESHOLD * 100 + 0.5)
            probabilities = {
                str(label): float(prob)
                for label, prob in zip(label_classes, avg_proba)
            }

        if confidence_percent >= threshold_percent:
            label = avg_label
            status = "ready"
        else:
            label = "ไม่มั่นใจ - ทำท่าใหม่ให้ชัด"
            status = "uncertain"

        return jsonify({
            "label": label,
            "confidence": avg_confidence,
            "status": status,
            "sequenceLength": len(sequence),
            "probabilities": probabilities,
            "labels": label_classes.tolist(),
            "landmarks": landmarks,
        })
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500


@app.post("/api/reset")
def reset():
    with state_lock:
        reset_state()
    return jsonify({"ok": True})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--http-only", action="store_true", help="run only HTTP on port 5000")
    parser.add_argument("--https-only", action="store_true", help="run only HTTPS on port 5443")
    args = parser.parse_args()

    hostname = socket.gethostname()
    try:
        local_ip = socket.gethostbyname(hostname)
    except OSError:
        local_ip = "YOUR_COMPUTER_IP"

    if args.https_only:
        print(f"Open on phone: https://{local_ip}:5443/")
        print("On phone, accept the browser certificate warning once, then allow camera access.")
        app.run(host="0.0.0.0", port=5443, debug=False, threaded=True, ssl_context="adhoc")
    elif args.http_only:
        print("Open on this computer: http://127.0.0.1:5000/")
        print(f"Open on phone without camera HTTPS: http://{local_ip}:5000/")
        app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
    else:
        print("Open on this computer: http://127.0.0.1:5000/")
        print(f"Open on phone: https://{local_ip}:5443/")
        print("On phone, accept the browser certificate warning once, then allow camera access.")

        https_server = make_server(
            "0.0.0.0",
            5443,
            app,
            threaded=True,
            ssl_context="adhoc",
        )
        https_server_thread = threading.Thread(target=https_server.serve_forever, daemon=True)
        https_server_thread.start()

        app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
