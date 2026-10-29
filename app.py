from flask import Flask, render_template, Response, jsonify
import os
import face_recognition
from ultralytics import YOLO
import cv2
import numpy as np
import threading
import time
import pygame

app = Flask(__name__, template_folder="templates", static_folder="static")

# Load YOLOv8 model
model = YOLO("trained_mask_model.pt")

# Known people info
known_info = {
    "gokul": {"name": "Gokul Nathan M", "year": "III", "department": "ECE"},
    "pranav": {"name": "Pranav UP", "year": "III", "department": "ECE"},
    "bharath": {"name": "Bharath A", "year": "III", "department": "ECE"},
    "sathik": {"name": "Sathik Mansur B", "year": "III", "department": "MZ"},
}

# Load known faces (multiple images per person supported)
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

# Alerts + latest status
alerts = []
alert_cooldowns = {}
latest_status = {"name": "None", "status": "Waiting..."}

# Init camera
camera = cv2.VideoCapture(0)

# Init sound
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
    frame_count = 0
    while True:
        success, frame = camera.read()
        if not success:
            break

        frame_count += 1
        # Skip frames for performance (process every 2nd frame)
        if frame_count % 2 != 0:
            continue

        # Resize frame for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        rgb_small_frame = small_frame[:, :, ::-1]

        # YOLOv8 mask detection
        detections = classify_mask(rgb_small_frame)

        # Face recognition
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
            face_names.append(name)

        # Process each face and associate with mask status
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back up face locations since the frame we detected in was scaled to 0.5
            top *= 2
            right *= 2
            bottom *= 2
            left *= 2

            # Determine mask status for this face by checking overlap with mask_boxes
            mask_status = "No Mask"
            for (label, conf, (x1, y1, x2, y2)) in detections:
                x1, y1, x2, y2 = int(x1*2), int(y1*2), int(x2*2), int(y2*2)
                # Check if face box overlaps with mask box
                if (left < x2 and right > x1 and top < y2 and bottom > y1):
                    mask_status = label
                    break

            # Get person info
            person = known_info.get(name.lower(), {})
            display_name = person.get("name", name.title())

            # Update latest status (for the first detected person)
            if not face_names or face_names[0] == name:
                latest_status["name"] = display_name
                latest_status["status"] = mask_status

            # Set box color based on mask status
            if mask_status == "Mask":
                color = (0, 255, 0)  # Green
            else:
                color = (0, 0, 255)  # Red

            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, f"{display_name} | {mask_status}", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # Trigger alert if no mask and cooldown passed
            if mask_status == "No Mask":
                now = time.time()
                last_alert_time = alert_cooldowns.get(name, 0)
                if now - last_alert_time > 60:
                    alert_cooldowns[name] = now
                    print(f"[ALERT] Triggering alert for {display_name}")

                    # Play alert sound
                    try:
                        pygame.mixer.Sound(ALERT_SOUND).play()
                        print("[INFO] Alert sound played.")
                    except Exception as e:
                        print(f"[ERROR] Could not play sound: {e}")

                    # Save screenshot
                    timestamp = time.strftime("%Y%m%d-%H%M%S")
                    screenshot_filename = f"{name}_{timestamp}.jpg"
                    screenshot_path = os.path.join('static', 'screenshots', screenshot_filename)
                    os.makedirs('static/screenshots', exist_ok=True)
                    cv2.imwrite(screenshot_path, frame)
                    print(f"[INFO] Screenshot saved: {screenshot_path}")

                    # Send email alert
                    try:
                        import email_alert
                        subject = f"🚨 Mask Alert: {display_name if name != 'Unknown' else 'Unknown Person'}"
                        if name == "Unknown":
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
                        email_alert.send_email_alert(subject, body, attachment=screenshot_path)
                        print("[INFO] Email alert sent successfully.")
                    except Exception as e:
                        print(f"[ERROR] Failed to send email: {e}")

                    # Add to alerts list
                    alerts.append({
                        "name": display_name,
                        "year": person.get("year", "N/A") if name != "Unknown" else "N/A",
                        "dept": person.get("department", "N/A") if name != "Unknown" else "N/A",
                        "status": "No Mask",
                        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "image": f"screenshots/{screenshot_filename}"
                    })

        # Encode frame as JPEG
        ret, jpeg = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        frame_bytes = jpeg.tobytes()

        # Yield frame in byte format for streaming
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


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
