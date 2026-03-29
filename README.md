Driver Drowsiness Detection System

A real-time AI-powered driver safety system that detects drowsiness using a hybrid approach combining Eye Aspect Ratio (EAR) and a CNN deep learning model. The system triggers alerts and notifications to prevent accidents.

🔥 Features

👁️ Real-time Eye Tracking using MediaPipe

🧠 CNN-based Drowsiness Detection (.keras model)

🔊 Alarm System when drowsiness is detected

📊 Live EAR Graph Visualization

📩 SMS & Email Alerts (emergency notification)

🖥️ GUI Dashboard (Tkinter)

⚡ Hybrid Detection (EAR + Deep Learning) for higher accuracy


🧠 How It Works

Webcam captures live video

MediaPipe detects facial landmarks

Eye Aspect Ratio (EAR) is calculated

Eye images are passed to CNN model

If:

   EAR is below threshold AND
   
   CNN predicts drowsiness
   
   System triggers:
     🔊 Alarm
     📩 SMS
     📧 Email
   


📁 Project Structure

      Driver Drowsiness Detection System/
      │
      ├── dataset/
      │   ├── train/
      │   │   ├── open/
      │   │   └── closed/
      │   └── test/
      │       ├── open/
      │       └── closed/
      │
      ├── utils/
      │   └── alerts.py
      │
      ├── venv/
      ├── alarm.mp3
      ├── detect1.py   # Main file
      ├── train.py         # Model training
      ├── drowsiness_model.keras # Trained model
      |__ requirements.txt


⚙️ Installation

1. Clone repository
   
       git clone https://github.com/your-username/driver-drowsiness-detection.git
  
       cd driver-drowsiness-detection

2. Create virtual environment

       python -m venv venv

       venv\Scripts\activate   # Windows

5. Install dependencies
   
        pip install -r requirements.txt


▶️ Run the Project

      python detect_drowsiness.py


🧪 Model Details

  Input size: 64x64

  Model: CNN trained on eye dataset
  
  Classes: Open / Closed 
  
  Output: Probability of drowsiness


🔧 Configuration

  EAR_THRESHOLD = 0.20

  CNN_THRESHOLD = 0.7
  
  DROWSY_FRAMES = 20

📊 Future Improvements

😴 Yawning detection

🎥 Head tilt detection

📸 Save drowsy snapshots

📈 Accuracy metrics display

🌐 Web/mobile integration

🛠️ Tech Stack

  Python
  
  OpenCV
  
  MediaPipe

  TensorFlow / Keras
  
  NumPy
  
  Tkinter
  
  Pygame

  
⚠️ Note

  Make sure alarm.mp3 is in the root folder
  
  Ensure drowsiness_model.keras is present

  Webcam access is required


📌 Use Case

This system can be used in:

🚗 Driver safety systems

🚛 Transport monitoring

🚘 Smart vehicles

🧠 AI-based surveillance
