import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import os

# Email credentials (to be configured)
SENDER_EMAIL = "fmdetectionbit@gmail.com"
SENDER_PASSWORD = "iuboltipvixyvelw"
RECEIVER_EMAIL = "pranavup.ec23@bitsathy.ac.in"

def send_email_alert(name, year, department, time_str, screenshot_path, is_known=True):
    try:
        msg = MIMEMultipart()
        msg['From'] = SENDER_EMAIL
        msg['To'] = RECEIVER_EMAIL
        if is_known:
            msg['Subject'] = f"Mask Violation Alert: {name}"
            body = f"Name: {name}\nYear: {year}\nDepartment: {department}\nTime: {time_str}\n"
        else:
            msg['Subject'] = "⚠️ Unknown Person Detected"
            body = f"An unknown person was detected without a mask at {time_str}."

        msg.attach(MIMEText(body, 'plain'))

        # Attach screenshot
        if os.path.exists(screenshot_path):
            attachment = open(screenshot_path, "rb")
            part = MIMEBase('application', 'octet-stream')
            part.set_payload(attachment.read())
            encoders.encode_base64(part)
            part.add_header('Content-Disposition', f"attachment; filename= {os.path.basename(screenshot_path)}")
            msg.attach(part)
            attachment.close()

        # Setup SMTP server and send email
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        text = msg.as_string()
        server.sendmail(SENDER_EMAIL, RECEIVER_EMAIL, text)
        server.quit()
        print(f"Email alert sent for {name if is_known else 'Unknown Person'}")
    except Exception as e:
        print(f"Failed to send email alert: {e}")
