# detect_drowsiness.py
import cv2
import mediapipe as mp
import numpy as np
import pygame
from tensorflow.keras.models import load_model
import threading
from utils.alerts import send_sms, send_email
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from PIL import Image, ImageTk

# -----------------------------
# 🔊 Initialize alarm
# -----------------------------
pygame.mixer.init()
pygame.mixer.music.load("alarm.mp3")

# -----------------------------
# 👁️ Mediapipe setup
# -----------------------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# -----------------------------
# 📏 EAR Calculation
# -----------------------------
def calculate_EAR(eye):
    A = np.linalg.norm(np.array(eye[1]) - np.array(eye[5]))
    B = np.linalg.norm(np.array(eye[2]) - np.array(eye[4]))
    C = np.linalg.norm(np.array(eye[0]) - np.array(eye[3]))
    return (A + B) / (2.0 * C)

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# 🔧 FIXED THRESHOLDS
EAR_THRESHOLD = 0.20
CNN_THRESHOLD = 0.7
DROWSY_FRAMES = 20

frame_count = 0
alarm_on = False

# -----------------------------
# Load CNN model
# -----------------------------
model = load_model("drowsiness_model.keras")
IMG_SIZE = 64

# -----------------------------
# Tkinter GUI
# -----------------------------
root = tk.Tk()
root.title("Driver Drowsiness Dashboard")

video_label = tk.Label(root)
video_label.pack()

# Graph
ear_history = []
fig, ax = plt.subplots(figsize=(5,2))
line, = ax.plot([], [])
ax.set_ylim(0, 0.6)
ax.set_xlim(0, 50)
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack()

# -----------------------------
# Webcam
# -----------------------------
cap = cv2.VideoCapture(0)

def predict_eye(frame, eye_points, w, h):
    x1 = max(min([p[0] for p in eye_points]) - 10, 0)
    y1 = max(min([p[1] for p in eye_points]) - 10, 0)
    x2 = min(max([p[0] for p in eye_points]) + 10, w)
    y2 = min(max([p[1] for p in eye_points]) + 10, h)

    eye_img = frame[y1:y2, x1:x2]
    if eye_img.size == 0:
        return 0

    eye_img = cv2.resize(eye_img, (IMG_SIZE, IMG_SIZE))
    eye_img = eye_img / 255.0
    eye_img = np.expand_dims(eye_img, axis=0)

    pred = model.predict(eye_img, verbose=0)
    return pred[0][0]

def update_frame():
    global frame_count, alarm_on

    ret, frame = cap.read()
    if not ret:
        root.after(10, update_frame)
        return

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:

            left_eye = [(int(face_landmarks.landmark[i].x * w),
                         int(face_landmarks.landmark[i].y * h)) for i in LEFT_EYE]

            right_eye = [(int(face_landmarks.landmark[i].x * w),
                          int(face_landmarks.landmark[i].y * h)) for i in RIGHT_EYE]

            # Draw eye points
            for x, y in left_eye + right_eye:
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

            # EAR
            left_EAR = calculate_EAR(left_eye)
            right_EAR = calculate_EAR(right_eye)
            EAR = (left_EAR + right_EAR) / 2.0

            ear_history.append(EAR)
            if len(ear_history) > 50:
                ear_history.pop(0)

            # CNN prediction
            left_prob = predict_eye(frame, left_eye, w, h)
            right_prob = predict_eye(frame, right_eye, w, h)
            avg_prob = (left_prob + right_prob) / 2.0

            # 📊 Debug text
            cv2.putText(frame, f"EAR: {EAR:.2f}", (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame, f"Prob: {avg_prob:.2f}", (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            # 🚨 Drowsiness logic (fixed)
            if EAR < EAR_THRESHOLD and avg_prob > CNN_THRESHOLD:
                frame_count += 1

                cv2.putText(frame, "DROWSY!", (50, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

                if frame_count >= DROWSY_FRAMES:
                    if not alarm_on:
                        pygame.mixer.music.play(-1)
                        threading.Thread(target=send_sms, args=("Driver is drowsy!",)).start()
                        threading.Thread(target=send_email, args=("Driver is drowsy!",)).start()
                        alarm_on = True
            else:
                frame_count = 0
                if alarm_on:
                    pygame.mixer.music.stop()
                    alarm_on = False

    # Show frame in Tkinter
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)
    imgtk = ImageTk.PhotoImage(image=img)

    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)

    # Update graph
    line.set_ydata(ear_history)
    line.set_xdata(range(len(ear_history)))
    ax.relim()
    ax.autoscale_view()
    canvas.draw()

    root.after(10, update_frame)

# Run
root.after(0, update_frame)
root.mainloop()

cap.release()
pygame.mixer.quit()
cv2.destroyAllWindows()