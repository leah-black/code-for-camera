import cv2
import mediapipe as mp
import pyttsx3  # Text-to-speech library

# Initialize MediaPipe FaceMesh and Hands
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize Text-to-Speech Engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Speed of speech
engine.setProperty('volume', 0.9)  # Volume (0.0 to 1.0)

# Function to determine if someone is smiling
def is_smiling(face_landmarks, frame_width, frame_height):
    top_lip = face_landmarks.landmark[13]
    bottom_lip = face_landmarks.landmark[14]
    left_corner = face_landmarks.landmark[61]
    right_corner = face_landmarks.landmark[291]

    mouth_width = ((right_corner.x - left_corner.x) * frame_width)
    mouth_height = ((bottom_lip.y - top_lip.y) * frame_height)

    return mouth_height / mouth_width > 0.3

# Function to recognize gestures based on hand landmarks
def recognize_gesture(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[4]
    thumb_ip = hand_landmarks.landmark[3]
    index_tip = hand_landmarks.landmark[8]
    middle_tip = hand_landmarks.landmark[12]
    ring_tip = hand_landmarks.landmark[16]
    pinky_tip = hand_landmarks.landmark[20]

    if thumb_tip.y < thumb_ip.y and all(finger_tip.y > thumb_ip.y for finger_tip in [index_tip, middle_tip, ring_tip, pinky_tip]):
        return "Thumbs Up"
    if all(finger_tip.y < hand_landmarks.landmark[0].y for finger_tip in [thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip]):
        return "Open Hand"
    if (index_tip.y < hand_landmarks.landmark[6].y and middle_tip.y < hand_landmarks.landmark[10].y and
        ring_tip.y > hand_landmarks.landmark[14].y and pinky_tip.y > hand_landmarks.landmark[18].y):
        return "Peace Sign"
    return None

# Function to speak a message
def speak_message(message):
    engine.say(message)
    engine.runAndWait()

# Set up webcam
cap = cv2.VideoCapture(0)
with mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh, \
    mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    
    previous_smile = False
    previous_gesture = None

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Convert frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_results = face_mesh.process(frame_rgb)
        hand_results = hands.process(frame_rgb)
        frame_height, frame_width, _ = frame.shape

        # Face Tracking
        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                mp_draw.draw_landmarks(
                    frame, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())

                # Smiling Detection
                is_user_smiling = is_smiling(face_landmarks, frame_width, frame_height)
                if is_user_smiling:
                    cv2.putText(frame, "You look happy!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    if not previous_smile:
                        speak_message("You look happy. It's great to see you smile!")
                    previous_smile = True
                else:
                    cv2.putText(frame, "Neutral expression", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    previous_smile = False

        # Hand Tracking
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                gesture = recognize_gesture(hand_landmarks)
                if gesture:
                    cv2.putText(frame, gesture, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
                    if gesture != previous_gesture:
                        speak_message(f"{gesture} detected!")
                    previous_gesture = gesture

        # Display the frame
        cv2.imshow('Face and Gesture Detection', frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
