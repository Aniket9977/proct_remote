from flask import Flask, render_template, Response
import cv2
import logging
from face_detection.detector import get_face_detector
from eye_aspect_ratio.ear_calculator import get_aspect_ratio
from mouth_aspect_ratio.mar_calculator import get_mouth_aspect_ratio
from utils.draw_face_landmarks import draw_face_landmarks

# Configure logging
logging.basicConfig(filename='proctoring_log.txt', 
                    level=logging.INFO, 
                    format='%(asctime)s - %(message)s', 
                    datefmt='%Y-%m-%d %H:%M:%S')

# Initialize Flask app
app = Flask(__name__)

# Load face detector and shape predictor
detector, predictor = get_face_detector()

# Thresholds for EAR and MAR
EAR_THRESHOLD = 0.14  # Adjusted threshold for eye aspect ratio
MAR_THRESHOLD = 0.1   # Adjusted threshold for mouth aspect ratio

# Flask route for the home page
@app.route('/')
def index():
    # Render the HTML template for the front-end
    return render_template('index.html')

# Flask route for video feed
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Function to generate frames from the webcam
def generate_frames():
    cap = cv2.VideoCapture(0)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        # Count the number of faces (people) detected
        num_people = len(faces)

        if num_people >= 2:
            logging.info('More than 1 person found')

        # Process each face detected
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
                logging.info("Suspicious Eye Activity detected.")

            # Check if the MAR is above a certain threshold (for suspicious mouth activity)
            if mar > MAR_THRESHOLD:
                cv2.putText(frame, "Suspicious Mouth Activity!", (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                logging.info("Suspicious Mouth Activity detected.")

        # Display the number of people detected on the screen
        cv2.putText(frame, f"People Count: {num_people}", (25, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Convert frame to JPEG format
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Yield the frame to be used in the video stream
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

if __name__ == '__main__':
    app.run(debug=True)
