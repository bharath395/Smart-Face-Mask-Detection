from flask import Flask, render_template, Response, jsonify
import cv2, time, os
import face_recognition
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Flask app
app = Flask(__name__, template_folder="templates", static_folder="static")

# Load mask detector model
model = load_model("mask_detector_model.h5")

# Known people info
known_info = {
    "gokul": {"name": "Gokul Nathan M", "year": "III", "department": "ECE"},
    "pranav": {"name": "Pranav UP", "year": "III", "department": "ECE"},
    "bharath": {"name": "Bharath A", "year": "III", "department": "ECE"},
    "sathik": {"name": "Sathik Mansur B", "year": "III", "department": "MZ"}
}

# Load known faces
known_face_encodings, known_face_names = [], []
base_path = "known_faces"
if os.path.exists(base_path):
    for folder_name in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder_name)
        if os.path.isdir(folder_path):
            for filename in os.listdir(folder_path):
                if filename.lower().endswith((".jpg", ".png")):
                    img = face_recognition.load_image_file(os.path.join(folder_path, filename))
                    enc = face_recognition.face_encodings(img)
                    if enc:
                        known_face_encodings.append(enc[0])
                        known_face_names.append(folder_name.lower())

# Alerts + latest status
alerts = []
alert_cooldowns = {}
latest_status = {"name": "None", "status": "Waiting..."}

# Camera
camera = cv2.VideoCapture(0)


def classify_mask(frame, box):
    top, right, bottom, left = box
    h, w = frame.shape[:2]
    top, left = max(0, top), max(0, left)
    right, bottom = min(w, right), min(h, bottom)

    if bottom <= top or right <= left:
        return (0.5, 0.5)

    face = frame[top:bottom, left:right]
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = cv2.resize(face, (224, 224))
    face = img_to_array(face)
    face = preprocess_input(face)
    face = np.expand_dims(face, axis=0)
    return model.predict(face, verbose=0)[0]


def generate_frames():
    global latest_status
    while True:
        success, frame = camera.read()
        if not success:
            break

        small = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_small)
        face_encodings = face_recognition.face_encodings(rgb_small, face_locations)

        for (enc, loc) in zip(face_encodings, face_locations):
            distances = face_recognition.face_distance(known_face_encodings, enc)
            min_dist = np.min(distances) if distances.size > 0 else 1.0

            if min_dist < 0.45:
                idx = np.argmin(distances)
                name_key = known_face_names[idx]
                person = known_info.get(name_key, {})
                display_name = person.get("name", name_key.title())
            else:
                name_key, display_name, person = "unknown", "Unknown", {}

            top, right, bottom, left = [v * 4 for v in loc]
            mask, no_mask = classify_mask(frame, (top, right, bottom, left))
            label, color = ("Mask", (0, 255, 0)) if mask > no_mask else ("No Mask", (0, 0, 255))

            # Update latest status
            latest_status["name"] = display_name
            latest_status["status"] = label

            # Draw on frame
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, f"{display_name} | {label}", (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # Save alert
            if label == "No Mask":
                now = time.time()
                if now - alert_cooldowns.get(name_key, 0) >= 60:
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

        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')


# Flask routes
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
