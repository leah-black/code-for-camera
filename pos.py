import cv2
import mediapipe as mp
import pyttsx3  # Text-to-speech library
import csv
import time
import serial  # Serial communication for Arduino
import random
from datetime import datetime

# ✅ Initialize Serial Communication with Arduino (Update the port)
try:
    arduino = serial.Serial('/dev/tty.usbmodem2101', 9600)  # Change for your system
    time.sleep(2)  # Wait for the connection to stabilize
    arduino.write(b'N')  # Ensure LED is OFF initially
except Exception as e:
    print("⚠️ Could not connect to Arduino. Make sure it's plugged in!")
    arduino = None  # Continue without Arduino support

# ✅ Initialize MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
mp_draw = mp.solutions.drawing_utils

# ✅ Initialize Text-to-Speech Engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Speed of speech
engine.setProperty('volume', 0.9)  # Volume (0.0 to 1.0)

# ✅ Positive Messages 🎉
positive_messages = [
    "Wow! You have a wonderful presence!",
    "Hey there, you're looking great today!",
    "Keep smiling, the world needs your light!",
    "You're doing awesome just by being here!",
    "Your energy is truly inspiring!",
    "You have such a kind and friendly face!",
    "Keep being you! You're amazing!",
    "Wow, you light up the screen!"
]

# ✅ Timer variables
face_detected = False
start_time = None

# ✅ Logging setup
csv_file = "observer_log.csv"
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Timestamp", "Event", "Duration (seconds)"])  # Header row

def speak_message(message):
    """Speak a positive message out loud."""
    print(f"😊 Saying: {message}")
    engine.say(message)
    engine.runAndWait()

# ✅ Webcam capture setup
cap = cv2.VideoCapture(0)

with mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("No frame detected.")
            continue

        # Convert to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process face detection
        face_results = face_mesh.process(frame_rgb)

        # Frame dimensions
        frame_height, frame_width, _ = frame.shape

        with open(csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)

            # ✅ Face Observations
            if face_results.multi_face_landmarks:
                if not face_detected:
                    print("Face detected – Timer started! ⏳")
                    if arduino:
                        arduino.write(b'R')  # Turn LED RED
                    face_detected = True
                    start_time = time.time()  # Start the timer

                    # Speak a positive message when face first detected
                    positive_message = random.choice(positive_messages)
                    speak_message(positive_message)

                for face_landmarks in face_results.multi_face_landmarks:
                    mp_draw.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION)

                # ✅ Display Timer
                elapsed_time = time.time() - start_time
                timer_text = f"Time in Frame: {int(elapsed_time)}s"
                cv2.putText(frame, timer_text, (50, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            else:
                if face_detected:
                    elapsed_time = time.time() - start_time
                    print(f"Face left – Total time in frame: {int(elapsed_time)}s ⏹️")
                    if arduino:
                        arduino.write(b'N')  # Turn OFF LED
                    writer.writerow([datetime.now(), "Face left", int(elapsed_time)])  # Log duration
                    face_detected = False
                    start_time = None  # Reset timer

        # ✅ Display frame
        cv2.imshow('😊 Positive Face Timer', frame)

        # ✅ Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# ✅ Turn off the LED when the program exits
if arduino:
    arduino.write(b'N')  # Turn OFF LED
    arduino.close()

cap.release()
cv2.destroyAllWindows()

print("\nTracking ended. All events are recorded in 'observer_log.csv'.")