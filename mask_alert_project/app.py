from flask import Flask, render_template, Response, jsonify
import cv2, time, os
import face_recognition
import numpy as np
import torch
from ultralytics import YOLO
import pygame

# Flask app
app = Flask(__name__, template_folder="templates", static_folder="static")

# ✅ Load YOLOv8 model
model = YOLO("trained_mask_model.pt")

# ✅ Known people info
known_info = {
    "gokul": {"name": "Gokul Nathan M", "year": "III", "department": "ECE"},
    "pranav": {"name": "Pranav UP", "year": "III", "department": "ECE"},
    "bharath": {"name": "Bharath A", "year": "III", "department": "ECE"},
    "sathik": {"name": "Sathik Mansur B", "year": "III", "department": "MZ"}
}

# ✅ Load known faces (multiple images per person supported)
known_face_encodings = []
known_face_names = []

base_path = "known_faces"
if os.path.exists(base_path):
    for person_name in os.listdir(base_path):
        person_folder = os.path.join(base_path, person_name)
        if os.path.isdir(person_folder):
            for filename in os.listdir(person_folder):
                if filename.lower().endswith((".jpg", ".png", ".jpeg")):
                    img_path = os.path.join(person_folder, filename)
                    image = face_recognition.load_image_file(img_path)
                    encodings = face_recognition.face_encodings(image)
                    if encodings:
                        known_face_encodings.append(encodings[0])
                        known_face_names.append(person_name.lower())
                        print(f"[INFO] Loaded face for {person_name} from {filename}")

# ✅ Alerts + latest status
alerts = []
alert_cooldowns = {}
latest_status = {"name": "None", "status": "Waiting..."}

# ✅ Init camera
camera = cv2.VideoCapture(0)

# ✅ Init sound
pygame.mixer.init()
ALERT_SOUND = "alert.wav"


def classify_mask(frame):
    """
    Run YOLOv8 inference on frame.
    Returns list of (label, confidence, (x1,y1,x2,y2)).
    Assumes: class 0 = Mask, 1 = No Mask
    """
    results = model(frame, verbose=False)
    detections = []
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        label = "Mask" if cls == 0 else "No Mask"
        detections.append((label, conf, (x1, y1, x2, y2)))
    return detections


def generate_frames():
    global latest_status
    while True:
        success, frame = camera.read()
        if not success:
            break

        detections = classify_mask(frame)

        for (label, conf, (x1, y1, x2, y2)) in detections:
            # --- Face Recognition ---
            name_key, display_name, person = "unknown", "Unknown", {}
            face_img = frame[y1:y2, x1:x2]

            try:
                rgb_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                enc = face_recognition.face_encodings(
                    rgb_face, [(0, rgb_face.shape[1], rgb_face.shape[0], 0)]
                )
                if enc and len(known_face_encodings) > 0:
                    distances = face_recognition.face_distance(known_face_encodings, enc[0])
                    if len(distances) > 0 and np.min(distances) < 0.45:
                        idx = int(np.argmin(distances))
                        name_key = known_face_names[idx]
                        person = known_info.get(name_key, {})
                        display_name = person.get("name", name_key.title())
            except Exception as e:
                print("[face_recog error]", e)

            # --- Update status ---
            latest_status["name"] = display_name
            latest_status["status"] = label

            # --- Draw detection ---
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{display_name} | {label}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # --- Alert handling ---
            if label == "No Mask":
                now = time.time()
                if now - alert_cooldowns.get(name_key, 0) >= 60:  # 60s cooldown
                    snap_name = f"{name_key}_{int(now)}.jpg"
                    snap_path = os.path.join("static/screenshots", snap_name)
                    os.makedirs("static/screenshots", exist_ok=True)
                    cv2.imwrite(snap_path, frame)

                    alerts.append({
                        "name": display_name,
                        "year": person.get("year", "N/A") if name_key != "unknown" else "N/A",
                        "dept": person.get("department", "N/A") if name_key != "unknown" else "N/A",
                        "status": "No Mask",
                        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "image": f"screenshots/{snap_name}"
                    })
                    alert_cooldowns[name_key] = now

                    # 🚨 Play alarm
                    try:
                        pygame.mixer.Sound(ALERT_SOUND).play()
                    except Exception as e:
                        print(f"[ERROR] Could not play sound: {e}")

                    # 🚨 Send email
                    try:
                        import email_alert
                        subject = f"🚨 Mask Alert: {display_name if name_key!='unknown' else 'Unknown Person'}"
                        if name_key == "unknown":
                            body = (
                                f"Person: Unknown\n"
                                f"Status: No Mask\n"
                                f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                            )
                        else:
                            body = (
                                f"Name: {display_name}\n"
                                f"Year: {person.get('year', 'N/A')}\n"
                                f"Department: {person.get('department', 'N/A')}\n"
                                f"Status: No Mask\n"
                                f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                            )
                        email_alert.send_email_alert(subject, body, attachment=snap_path)
                        print("[INFO] Email alert sent.")
                    except Exception as e:
                        print(f"[ERROR] Failed to send email: {e}")

        # --- Stream ---
        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')


# ✅ Flask routes
@app.route("/")
@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")


@app.route("/alerts")
def alerts_page():
    return render_template("alerts.html")


@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/get_alerts")
def get_alerts():
    return jsonify(alerts)


@app.route("/get_status")
def get_status():
    return jsonify(latest_status)


if __name__ == "__main__":
    app.run(debug=True)
