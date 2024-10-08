import cv2
import streamlit as st
import logging
from datetime import datetime
from face_detection.detector import get_face_detector
from eye_aspect_ratio.ear_calculator import get_aspect_ratio
from mouth_aspect_ratio.mar_calculator import get_mouth_aspect_ratio
from utils.draw_landmarks import draw_face_landmarks

# Configure logging
logging.basicConfig(filename='proctoring_log.txt', 
                    level=logging.INFO, 
                    format='%(asctime)s - %(message)s', 
                    datefmt='%Y-%m-%d %H:%M:%S')

st.set_page_config(page_title="Remote Proctoring", layout="wide")

detector, predictor = get_face_detector()

EAR_THRESHOLD = 0.14  # Adjusted threshold for eye aspect ratio
MAR_THRESHOLD = 0.1   # Adjusted threshold for mouth aspect ratio

st.title("Remote Proctoring System")
st.write("This application detects suspicious eye and mouth activities, and counts the number of people in the frame.")

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

            num_people = len(faces)

            if num_people >=2:
                logging.info('More than 1 person found')


            for face in faces:
                landmarks = predictor(gray, face)

                left_eye_points = [landmarks.part(i) for i in range(36, 42)]
                right_eye_points = [landmarks.part(i) for i in range(42, 48)]

                mouth_points = [landmarks.part(i) for i in range(48, 68)]

                left_ear = get_aspect_ratio(left_eye_points)
                right_ear = get_aspect_ratio(right_eye_points)

                mar = get_mouth_aspect_ratio(mouth_points)

 
                frame = draw_face_landmarks(frame, landmarks)

     
                if left_ear < EAR_THRESHOLD or right_ear < EAR_THRESHOLD:
                    cv2.putText(frame, "Suspicious Eye Activity!", (25, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    logging.info("Suspicious Eye Activity detected.")

  
                if mar > MAR_THRESHOLD:
                    cv2.putText(frame, "Suspicious Mouth Activity!", (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    logging.info("Suspicious Mouth Activity detected.")

  
            cv2.putText(frame, f"People Count: {num_people}", (25, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            frame_placeholder.image(frame, channels="RGB")

        cap.release()
else:
    st.write("Camera is not running. Check the 'Run Camera' checkbox to start.")
