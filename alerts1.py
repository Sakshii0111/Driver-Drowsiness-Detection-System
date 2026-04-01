# utils/alerts.py
import smtplib
from twilio.rest import Client

# -----------------------------
# Twilio SMS setup
# -----------------------------
def send_sms(message):
    account_sid = 'ACdd2089999911e6f1b0005c59aeb5b5ce'
    auth_token = '6631dd179196820e5648498d14b9f754'
    client = Client(account_sid, auth_token)
    client.messages.create(
        body=message,
        from_='+147XXXXXX',  # Twilio number
        to='+91620XXXXXXX'     # Your phone
    )

# -----------------------------
# Email setup
# -----------------------------
def send_email(message):
    sender = 'youremail@gmail.com'
    password = 'yourpassword'
    receiver = 'receiveremail@gmail.com'

    with smtplib.SMTP('smtp.gmail.com', 587) as server:
        server.starttls()
        server.login(sender, password)
        server.sendmail(sender, receiver, message)