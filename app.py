import cv2
import streamlit as st
from face_detection.detector import get_face_detector
from eye_aspect_ratio.ear_calculator import get_aspect_ratio
from mouth_aspect_ratio.mar_calculator import get_mouth_aspect_ratio
from utils.draw_landmarks import draw_face_landmarks

# Streamlit page configuration
st.set_page_config(page_title="Remote Proctoring", layout="wide")

# Load face detector and shape predictor
detector, predictor = get_face_detector()

# Thresholds for EAR and MAR
EAR_THRESHOLD = 0.18  # Adjusted threshold for eye aspect ratio
MAR_THRESHOLD = 0.1   # Adjusted threshold for mouth aspect ratio

# Streamlit application interface
st.title("Remote Proctoring System")
st.write("This application detects suspicious eye and mouth activities, and counts the number of people in the frame.")

# Capture video frames
run = st.checkbox('Run Camera')

if run:
    cap = cv2.VideoCapture(1)
    
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
                    cv2.putText(frame, "Suspicious Eye Activity!", (25, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                # Check if the MAR is above a certain threshold (for suspicious mouth activity)
                if mar > MAR_THRESHOLD:
                    cv2.putText(frame, "Suspicious Mouth Activity!", (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255),2)

            # Display the number of people detected on the screen
            cv2.putText(frame, f"People Count: {num_people}", (25, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Convert frame to RGB format (Streamlit expects RGB images)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Display the frame using Streamlit
            frame_placeholder.image(frame, channels="RGB")

        cap.release()
else:
    st.write("Camera is not running. Check the 'Run Camera' checkbox to start.")
