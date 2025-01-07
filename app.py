
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import cv2
import numpy as np
from face_detection.detector import get_face_detector
from eye_aspect_ratio.ear_calculator import get_aspect_ratio
from mouth_aspect_ratio.mar_calculator import get_mouth_aspect_ratio

# Initialize face detector and predictor
detector, predictor = get_face_detector()

# Thresholds
EAR_THRESHOLD = 0.14
MAR_THRESHOLD = 0.1


class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.detector = detector
        self.predictor = predictor

    def transform(self, frame):
        frame = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = self.detector(gray)
        for face in faces:
            landmarks = self.predictor(gray, face)

            # Eye aspect ratio (EAR)
            left_eye_points = [landmarks.part(i) for i in range(36, 42)]
            right_eye_points = [landmarks.part(i) for i in range(42, 48)]
            left_ear = get_aspect_ratio(left_eye_points)
            right_ear = get_aspect_ratio(right_eye_points)

            # Mouth aspect ratio (MAR)
            mouth_points = [landmarks.part(i) for i in range(48, 68)]
            mar = get_mouth_aspect_ratio(mouth_points)

            # Annotate suspicious activities
            if left_ear < EAR_THRESHOLD or right_ear < EAR_THRESHOLD:
                cv2.putText(
                    frame,
                    "Suspicious Eye Activity!",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                )

            if mar > MAR_THRESHOLD:
                cv2.putText(
                    frame,
                    "Suspicious Mouth Activity!",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                )

        return frame


# Streamlit App
st.title("Remote Proctoring System")
st.write(
    """
This application uses face detection to monitor suspicious activities 
such as eye and mouth movements in real-time.
"""
)

webrtc_streamer(
    key="example",
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
)
