import cv2
import numpy as np
import face_recognition
import os
import time
import smtplib
import ssl
from email.message import EmailMessage
import simpleaudio as sa
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

# Create screenshots folder if not exists
if not os.path.exists("screenshots"):
    os.makedirs("screenshots")

# Load known faces
known_face_encodings = []
known_face_names = []
base_path = 'known_faces'

for folder_name in os.listdir(base_path):
    folder_path = os.path.join(base_path, folder_name)
    if os.path.isdir(folder_path):
        for filename in os.listdir(folder_path):
            if filename.lower().endswith((".jpg", ".png")):
                image_path = os.path.join(folder_path, filename)
                img = face_recognition.load_image_file(image_path)
                encodings = face_recognition.face_encodings(img)
                if encodings:
                    known_face_encodings.append(encodings[0])
                    known_face_names.append(folder_name.lower())  # keep lowercase for lookup

# Alert cooldown
alert_cooldowns = {}

# Play alert sound
def play_alert_sound():
    try:
        wave_obj = sa.WaveObject.from_wave_file("alert.wav")
        wave_obj.play()
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
        msg.set_content(
            f"{details.get('name', name)} ({details.get('year', '')}, {details.get('department', '')}) "
            f"was detected without a mask."
        )

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

    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_small)
    face_encodings = face_recognition.face_encodings(rgb_small, face_locations)

    for (face_encoding, face_location) in zip(face_encodings, face_locations):
        distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        min_distance = np.min(distances) if distances.size > 0 else 1.0

        if min_distance < 0.45:
            match_index = np.argmin(distances)
            name_key = known_face_names[match_index]   # folder name (lowercase)
            details = known_info.get(name_key, {})
            display_name = details.get("name", name_key.title())
        else:
            name_key = "unknown"
            display_name = "Unknown"

        # Scale back face location
        top, right, bottom, left = [v * 4 for v in face_location]
        prediction = detect_and_classify_face(frame, (top, right, bottom, left))
        (mask, without_mask) = prediction

        # Label + Color
        if mask > without_mask:
            label = "Mask"
            color = (0, 255, 0)   # Green for mask
        else:
            label = "No Mask"
            color = (0, 0, 255)   # Red for no mask

        # Display text (fix for Unknown also)
        if name_key == "unknown":
            display_text = f"Unknown | {label}"
        else:
            display_text = f"{display_name} | {label}"

        # Draw bounding box + text
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.putText(frame, display_text, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Handle alerts only for No Mask
        if label == "No Mask":
            now = time.time()
            cooldown = alert_cooldowns.get(name_key, 0)
            if now - cooldown >= 60:
                screenshot_filename = f"screenshots/{name_key}_{int(now)}.jpg"
                cv2.imwrite(screenshot_filename, frame)
                play_alert_sound()
                send_email_alert(name_key, screenshot_filename, is_unknown=(name_key == "unknown"))
                alert_cooldowns[name_key] = now

    cv2.imshow("Smart Face Mask Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()
