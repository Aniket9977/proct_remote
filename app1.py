import cv2
import streamlit as st
import logging
from datetime import datetime
from face_detection.detector import get_face_detector
from eye_aspect_ratio.ear_calculator import get_aspect_ratio
from mouth_aspect_ratio.mar_calculator import get_mouth_aspect_ratio
from utils.draw_landmarks import draw_face_landmarks
from utils import process_frame
import torch  # Using PyTorch for YOLO-based detection

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
st.write("This application detects suspicious eye and mouth activities, counts the number of people in the frame, and detects cheating gadgets.")

run = st.checkbox('Run Camera')

# Select which cameras to use
camera_option = st.selectbox('Select Camera Input', ('Camera 1', 'Camera 2', 'Both'))


model = torch.hub.load('yolov5', 'yolov5s', source='local')  # Use 'source=local' to load from the local cloned directory
model.conf = 0.5  # Set confidence threshold to adjust model sensitivity
if run:
    # Setup two camera captures
    cap1 = cv2.VideoCapture(0)
    cap2 = cv2.VideoCapture(1) if camera_option in ('Camera 2', 'Both') else None

    if not cap1.isOpened() or (cap2 and not cap2.isOpened()):
        st.error("Error: One or both cameras could not be opened")
    else:
        frame_placeholder = st.empty()
        while run:
            ret1, frame1 = cap1.read()
            ret2, frame2 = cap2.read() if cap2 else (False, None)
            
            # Handle camera 1
            if ret1:
                frame1 = process_frame(frame1, detector, predictor, model, EAR_THRESHOLD, MAR_THRESHOLD)
                frame_placeholder.image(frame1, channels="RGB")

            # Handle camera 2 if selected
            if camera_option in ('Camera 2', 'Both') and ret2:
                frame2 = process_frame(frame2, detector, predictor, model, EAR_THRESHOLD, MAR_THRESHOLD)
                frame_placeholder.image(frame2, channels="RGB")

        cap1.release()
        if cap2:
            cap2.release()
else:
    st.write("Camera is not running. Check the 'Run Camera' checkbox to start.")

# Function to process frames
