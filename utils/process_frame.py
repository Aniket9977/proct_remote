import cv2
import streamlit as st
import logging
from datetime import datetime
from face_detection.detector import get_face_detector
from eye_aspect_ratio.ear_calculator import get_aspect_ratio
from mouth_aspect_ratio.mar_calculator import get_mouth_aspect_ratio
from utils.draw_landmarks import draw_face_landmarks
from utils import try_except_utils
from yolov5.models.common import AutoShape , DetectMultiBackend
import torch  # Using PyTorch for YOLO-based detection 

def process_frame(frame, detector, predictor, model, EAR_THRESHOLD, MAR_THRESHOLD):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    num_people = len(faces)

    if num_people >= 2:
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

        # Detect suspicious eye or mouth activity
        if left_ear < EAR_THRESHOLD or right_ear < EAR_THRESHOLD:
            cv2.putText(frame, "Suspicious Eye Activity!", (25, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            logging.info("Suspicious Eye Activity detected.")

        if mar > MAR_THRESHOLD:
            cv2.putText(frame, "Suspicious Mouth Activity!", (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            logging.info("Suspicious Mouth Activity detected.")

    # Detect gadgets like mobile phones using YOLO
    results = model(frame)
    for result in results.pandas().xyxy[0].itertuples():
        if result.name in ["cell phone", "laptop", "tablet"]:
            x1, y1, x2, y2 = int(result.xmin), int(result.ymin), int(result.xmax), int(result.ymax)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Detected: {result.name}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            logging.info(f"Cheating Gadget Detected: {result.name}")

    cv2.putText(frame, f"People Count: {num_people}", (25, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    return frame
