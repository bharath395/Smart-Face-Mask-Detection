# TODO for Face Mask Detection and Alert System

## Step 1: Setup Flask app and basic routes
- Create app.py with Flask initialization
- Setup routes for dashboard (/) and alerts (/alerts)
- Setup static and templates folders

## Step 2: Load YOLOv8 model and known faces
- Load trained_mask_model.pt using ultralytics YOLOv8
- Load known faces encodings from known_faces folders using face_recognition
- Cache encodings for performance

## Step 3: Implement real-time webcam capture and processing
- Use OpenCV VideoCapture with threading for non-blocking capture
- For each frame:
  - Detect masks using YOLOv8
  - Detect faces and recognize known individuals
  - Draw bounding boxes (green for mask, red for no mask)
- Optimize for real-time performance (skip frames if needed)

## Step 4: Implement alert system
- Play alert.wav using pygame on no mask detection
- Save screenshot to static/screenshots with timestamp
- Send email alert using email_alert.py with details and screenshot
- Implement 60-second cooldown per person to avoid repeated alerts

## Step 5: Implement email_alert.py
- Setup Gmail SMTP with provided credentials
- Compose and send email alerts for known and unknown persons
- Attach screenshot images
- Handle errors gracefully

## Step 6: Implement dashboard.html
- Display live video feed from Flask route (e.g., MJPEG stream)
- Show current detected person and mask status
- Use AJAX or server-sent events for real-time updates

## Step 7: Implement alerts.html
- Display table of past violations with Name, Year, Department, Time, Screenshot
- Load data dynamically from Flask backend
- Show screenshots as clickable thumbnails

## Step 8: Create requirements.txt
- List all dependencies: Flask, face_recognition, torch, ultralytics, pygame, opencv-python, numpy, etc.

## Step 9: Testing and optimization
- Test real-time performance and alerting
- Optimize frame processing and UI updates to minimize lag
- Ensure cooldown works correctly

## Step 10: Final cleanup and documentation
- Add comments and docstrings
- Provide setup instructions if needed

---

Proceeding with Step 1: Setup Flask app and basic routes.
