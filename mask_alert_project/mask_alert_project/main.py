import cv2
import numpy as np
import face_recognition
import os
import time
import smtplib
import ssl
from email.message import EmailMessage
import simpleaudio as sa  # ✅ Better alternative to playsound
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

# Load trained mask detection model
model = load_model("mask_detector_model.h5")

# Known individuals info
known_info = {
    "gokul": {"name": "Gokul Nathan M", "year": "III", "department": "ECE"},
    "pranav": {"name": "Pranav UP", "year": "III", "department": "ECE"},
    "bharath": {"name": "Bharath A", "year": "III", "department": "ECE"},
    "sathik": {"name": "Sathik Mansur B", "year": "III", "department": "MZ"}
}

# Email setup
EMAIL_SENDER = "fmdetectionbit@gmail.com"
EMAIL_PASSWORD = "iuboltipvixyvelw"
EMAIL_RECEIVER = "pranavup.ec23@bitsathy.ac.in"

# Load known face encodings
known_face_encodings = []
known_face_names = []
path = 'known_faces'

for filename in os.listdir(path):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        img = face_recognition.load_image_file(f"{path}/{filename}")
        encodings = face_recognition.face_encodings(img)
        if encodings:
            known_face_encodings.append(encodings[0])
            name = os.path.splitext(filename)[0]
            known_face_names.append(name)

# Alert cooldown
alert_cooldowns = {}

# Play alert sound (simpleaudio-based)
def play_alert_sound():
    try:
        wave_obj = sa.WaveObject.from_wave_file("alert.wav")
        play_obj = wave_obj.play()
    except Exception as e:
        print("Sound playback failed:", e)

# Email sending function
def send_email_alert(name, image_path, is_unknown=False):
    msg = EmailMessage()
    msg['Subject'] = f"Mask Alert - {'Unknown Person' if is_unknown else name}"
    msg['From'] = EMAIL_SENDER
    msg['To'] = EMAIL_RECEIVER

    if is_unknown:
        msg.set_content("An unknown person without a mask has been detected.")
    else:
        details = known_info.get(name, {})
        msg.set_content(f"{details.get('name')} ({details.get('year')}, {details.get('department')}) was detected without a mask.")

    with open(image_path, 'rb') as f:
        file_data = f.read()
        file_name = os.path.basename(image_path)
        msg.add_attachment(file_data, maintype='image', subtype='jpeg', filename=file_name)

    context = ssl.create_default_context()
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        server.send_message(msg)

# Predict mask from face
def detect_and_classify_face(frame, face_location):
    top, right, bottom, left = face_location
    face = frame[top:bottom, left:right]
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = cv2.resize(face, (224, 224))
    face = img_to_array(face)
    face = preprocess_input(face)
    face = np.expand_dims(face, axis=0)
    prediction = model.predict(face)[0]
    return prediction

# Start video capture
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    # Resize for faster face detection
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Face detection
    face_locations = face_recognition.face_locations(rgb_small)
    face_encodings = face_recognition.face_encodings(rgb_small, face_locations)

    for (face_encoding, face_location) in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "unknown"

        if True in matches:
            match_index = matches.index(True)
            name = known_face_names[match_index]

        top, right, bottom, left = [v * 4 for v in face_location]  # scale back

        prediction = detect_and_classify_face(frame, (top, right, bottom, left))
        (mask, without_mask) = prediction

        label = "Mask" if mask > without_mask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

        # Draw bounding box
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.putText(frame, f"{name.upper()} | {label}", (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

        # Alert if no mask
        if label == "No Mask":
            now = time.time()
            cooldown = alert_cooldowns.get(name, 0)
            if now - cooldown >= 60:
                screenshot_path = f"screenshot_{name}_{int(now)}.jpg"
                cv2.imwrite(screenshot_path, frame)

                play_alert_sound()
                send_email_alert(name, screenshot_path, is_unknown=(name == "unknown"))

                alert_cooldowns[name] = now

    cv2.imshow("Smart Face Mask Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()
