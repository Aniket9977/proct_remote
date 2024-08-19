import cv2
import dlib
import numpy as np
import streamlit as st

# Streamlit page configuration
st.set_page_config(page_title="Remote Proctoring", layout="wide")

# Load pre-trained models for face and eye detection
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def get_aspect_ratio(eye_points):
    A = np.linalg.norm(np.array([eye_points[1].x, eye_points[1].y]) - np.array([eye_points[5].x, eye_points[5].y]))
    B = np.linalg.norm(np.array([eye_points[2].x, eye_points[2].y]) - np.array([eye_points[4].x, eye_points[4].y]))
    C = np.linalg.norm(np.array([eye_points[0].x, eye_points[0].y]) - np.array([eye_points[3].x, eye_points[3].y]))
    aspect_ratio = (A + B) / (2.0 * C)
    return aspect_ratio

def get_mouth_aspect_ratio(mouth_points):
    A = np.linalg.norm(np.array([mouth_points[13].x, mouth_points[13].y]) - np.array([mouth_points[19].x, mouth_points[19].y]))
    B = np.linalg.norm(np.array([mouth_points[14].x, mouth_points[14].y]) - np.array([mouth_points[18].x, mouth_points[18].y]))
    C = np.linalg.norm(np.array([mouth_points[12].x, mouth_points[12].y]) - np.array([mouth_points[16].x, mouth_points[16].y]))
    mar = (A + B) / (2.0 * C)
    return mar

def draw_face_landmarks(frame, landmarks):
    for (x, y) in [(pt.x, pt.y) for pt in landmarks.parts()]:
        cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
    return frame

# Thresholds for EAR and MAR
EAR_THRESHOLD = 0.18  # Adjusted threshold for eye aspect ratio
MAR_THRESHOLD = 0.1   # Adjusted threshold for mouth aspect ratio

# Streamlit application interface
st.title("Remote Proctoring System")
st.write("This application detects suspicious eye and mouth activities, and counts the number of people in the frame.")

# Capture video frames
run = st.checkbox('Run Camera')

if run:
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        st.error("Error: Camera could not be opened")
    else:
        frame_placeholder = st.empty()
        while run:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to grab frame")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)

            # Count the number of faces (people) detected
            num_people = len(faces)

            for face in faces:
                landmarks = predictor(gray, face)

                # Extract eye coordinates
                left_eye_points = [landmarks.part(i) for i in range(36, 42)]
                right_eye_points = [landmarks.part(i) for i in range(42, 48)]

                # Extract mouth coordinates
                mouth_points = [landmarks.part(i) for i in range(48, 68)]

                # Calculate EAR for both eyes
                left_ear = get_aspect_ratio(left_eye_points)
                right_ear = get_aspect_ratio(right_eye_points)

                # Calculate MAR for the mouth
                mar = get_mouth_aspect_ratio(mouth_points)

                # Draw face landmarks
                frame = draw_face_landmarks(frame, landmarks)

                # Check if the EAR is below a certain threshold (for suspicious eye activity)
                if left_ear < EAR_THRESHOLD or right_ear < EAR_THRESHOLD:
                    cv2.putText(frame, "Suspicious Eye Activity!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # Check if the MAR is above a certain threshold (for suspicious mouth activity)
                if mar > MAR_THRESHOLD:
                    cv2.putText(frame, "Suspicious Mouth Activity!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Display the number of people detected on the screen
            cv2.putText(frame, f"People Count: {num_people}", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

            # Convert frame to RGB format (Streamlit expects RGB images)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Display the frame using Streamlit
            frame_placeholder.image(frame, channels="RGB")

        cap.release()
else:
    st.write("Camera is not running. Check the 'Run Camera' checkbox to start.")

