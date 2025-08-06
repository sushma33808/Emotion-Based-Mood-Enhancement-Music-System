from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session
from datetime import timedelta
import numpy as np
import cv2
import base64
import os
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
db = SQLAlchemy(app)

# --- Models ---
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(120), nullable=False)

    def __repr__(self):
        return '<User %r>' % self.username

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

# New: Emotion History model
class EmotionHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80))
    emotion = db.Column(db.String(50))
    timestamp = db.Column(db.DateTime, default=db.func.current_timestamp())
class Song(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(200), nullable=False)
    filepath = db.Column(db.String(300), nullable=False)
    emotion = db.Column(db.String(50), nullable=False)
    username = db.Column(db.String(80), nullable=False)
    approved = db.Column(db.Boolean, default=True)  # Admin upload is auto-approved

    def __repr__(self):
        return f'<Song {self.filename}>'

# --- Load model and cascade ---
try:
    model = load_model('model.h5')
except Exception as e:
    print(f"Error loading model.h5: {e}")
    exit()

try:
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    if face_cascade.empty():
        raise IOError("Cascade not loaded correctly.")
except Exception as e:
    print(f"Error loading cascade: {e}")
    exit()

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# --- Routes ---
@app.route('/')
def index():
    if 'username' not in session:
        return redirect(url_for('login'))
    return redirect(url_for('home'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'username' in session:
        return redirect(url_for('home'))
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        if user and user.check_password(password):
            session['username'] = username
            flash("Logged in successfully", "success")
            return redirect(url_for('home'))
        else:
            flash('Invalid username or password.', 'danger')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if 'username' in session:
        return redirect(url_for('home'))
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if User.query.filter_by(username=username).first():
            flash('Username already taken.', 'warning')
        else:
            new_user = User(username=username)
            new_user.set_password(password)
            db.session.add(new_user)
            db.session.commit()
            flash('Account created successfully! Please log in.', 'success')
            return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))

@app.route('/home')
def home():
    if 'username' not in session:
        flash('Please log in to access this page.', 'warning')
        return redirect(url_for('login'))
    return render_template('index.html', username=session['username'])

@app.route('/detect', methods=['POST'])
def detect_emotion():
    if 'username' not in session:
        return jsonify({"error": "Unauthorized"}), 401

    data = request.get_json()
    image_data = data['image'].split(',')[1]
    img_bytes = base64.b64decode(image_data)
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    emotion = "Neutral"
    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        roi = gray[y:y+h, x:x+w]
        roi = cv2.resize(roi, (48, 48))
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        prediction = model.predict(roi)[0]
        emotion = emotion_labels[prediction.argmax()]

    # Save to history
    history = EmotionHistory(username=session['username'], emotion=emotion)
    db.session.add(history)
    db.session.commit()

    return jsonify({"emotion": emotion})
@app.route('/songs/<emotion>')
def get_songs(emotion):
    if 'username' not in session:
        return jsonify({"error": "Unauthorized"}), 401

    base_song_path = 'static/songs'
    folder_map = {
        'Sad': os.path.join(base_song_path, 'sad'),
        'Angry': os.path.join(base_song_path, 'angry'),
        'Happy': os.path.join(base_song_path, 'happy'),
        'Neutral': os.path.join(base_song_path, 'neutral'),
        'Surprise': os.path.join(base_song_path, 'surprise'),  
        'Fear': os.path.join(base_song_path, 'fear')           
    }

    folder = folder_map.get(emotion)
    songs = []
    if folder and os.path.exists(folder):
        songs = [f"/{folder}/{file}" for file in os.listdir(folder) if file.lower().endswith(('.mp3', '.wav'))]
    else:
        print(f"Warning: Song folder not found for emotion '{emotion}' at '{folder}'")

    return jsonify({"songs": songs})


# Optional: View history
@app.route('/history')
def history():
    if 'username' not in session:
        flash('Please log in to view history.', 'warning')
        return redirect(url_for('login'))
    
    records = EmotionHistory.query.filter_by(username=session['username']) \
                                  .order_by(EmotionHistory.timestamp.desc()).all()
    return render_template('history.html', records=records, timedelta=timedelta)
@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if 'username' not in session:
        flash('Please log in first.', 'warning')
        return redirect(url_for('login'))

    if session['username'] != 'admin':
        flash('Only admins can upload songs.', 'danger')
        return redirect(url_for('home'))  # or wherever non-admins are allowed


    if request.method == 'POST':
        song_file = request.files['song_file']
        emotion = request.form['emotion']
        username = session['username']

        if song_file and song_file.filename.lower().endswith(('.mp3', '.wav')):
            filename = song_file.filename
            upload_folder = os.path.join('static', 'songs', emotion)
            os.makedirs(upload_folder, exist_ok=True)

            file_path = os.path.join(upload_folder, filename)
            song_file.save(file_path)

            new_song = Song(
                filename=filename,
                filepath=f"/static/songs/{emotion}/{filename}",
                emotion=emotion,
                username=username,
                approved=True  # admin upload = auto-approved
            )
            db.session.add(new_song)
            db.session.commit()

            flash('Song uploaded and saved successfully!', 'success')
            return redirect(url_for('upload'))
        else:
            flash('Invalid file format. Only MP3 and WAV are allowed.', 'danger')

    return render_template('upload.html')

@app.route('/admin')
def admin_dashboard():
    if 'username' not in session or session['username'] != 'admin':
        flash('Access denied. Admins only.', 'danger')
        return redirect(url_for('login'))
    songs = Song.query.all()  # If you want to show all songs, or remove if not needed

    users = User.query.all()
    user_data = []
    for user in users:
        uploads = Song.query.filter_by(username=user.username).count()
        #wishlists = Wishlist.query.filter_by(user_id=user.id).count()
        user_data.append({
            'username': user.username,
            'upload_count': uploads,
            #'wishlist_count': wishlists
        })

    return render_template('admin.html', users=user_data)


# --- Startup ---
if __name__ == '__main__':
    with app.app_context():
        db.create_all()

    os.makedirs('static/songs/sad', exist_ok=True)
    os.makedirs('static/songs/angry', exist_ok=True)
    os.makedirs('static/songs/happy', exist_ok=True)
    os.makedirs('static/songs/neutral', exist_ok=True)
    os.makedirs('static/images', exist_ok=True)

    app.run(debug=True)
