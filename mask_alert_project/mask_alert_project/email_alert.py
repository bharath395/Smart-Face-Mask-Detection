import smtplib
import ssl
from email.message import EmailMessage

# Update these with your real email and app password
SENDER_EMAIL = "fmdetectionbit@gmail.com"
SENDER_PASSWORD = "iuboltipvixyvelw"
RECEIVER_EMAIL = "pranavup.ec23@bitsathy.ac.in"

# Known people dictionary
KNOWN_PEOPLE = {
    "gokul": {
        "name": "Gokul Nathan M",
        "year": "III",
        "department": "ECE"
    },
    "pranav": {
        "name": "Pranav UP",
        "year": "III",
        "department": "ECE"
    },
    "bharath": {
        "name": "Bharath A",
        "year": "III",
        "department": "ECE"
    },
    "sathik": {
        "name": "Sathik Mansur B",
        "year": "III",
        "department": "MZ"
    }
}

def send_email_alert(name, screenshot_path=None, is_known=False):
    msg = EmailMessage()

    if is_known:
        person = KNOWN_PEOPLE.get(name, {})
        subject = f"[ALERT] {person.get('name')} Detected Without Mask"
        body = f"""
Name: {person.get('name')}
Year: {person.get('year')}
Department: {person.get('department')}
Status: Detected without mask in real-time
        """
        msg.set_content(body)
    else:
        subject = "[ALERT] Unknown Person Detected Without Mask"
        body = "An unknown individual was detected without a mask. See attached screenshot."
        msg.set_content(body)
        if screenshot_path:
            with open(screenshot_path, 'rb') as f:
                img_data = f.read()
                msg.add_attachment(img_data, maintype='image', subtype='jpeg', filename='alert.jpg')

    msg['Subject'] = subject
    msg['From'] = SENDER_EMAIL
    msg['To'] = RECEIVER_EMAIL

    context = ssl.create_default_context()
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.send_message(msg)
        print("[INFO] Email sent successfully.")
