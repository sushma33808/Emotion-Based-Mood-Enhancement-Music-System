# Emotion-Based-Mood-Enhancement-Music-System

This is a smart music player that detects human emotions via a webcam and plays uplifting music to enhance mood — especially when a sad emotion is detected.

## 🧠 Features

- 🎭 Emotion detection using webcam
- 🎶 Mood-based music recommendation
- 🎤 Voice interaction & audio feedback
- 🔐 Login/Signup page with Flask backend
- 🛠️ Admin panel for managing users & songs

## 📸 How It Works

1. Detects user emotion using ML model.
2. If user appears sad, it plays happy/cheerful music.

## 💻 Tech Stack

- Python, Flask
- HTML, CSS, JavaScript
- OpenCV, MediaPipe
- SQLite/MySQL (for user & song data)


## 🚀 How to Run

```bash
# Step 1: Clone the repo
git clone https://github.com/your-username/Emotion-Based-Mood-Enhancement-Music-System.git

# Step 2: Navigate into the folder
cd Emotion-Based-Mood-Enhancement-Music-System

# Step 3: Install dependencies
pip install -r requirements.txt

# Step 4: Run the Flask app
python app.py
