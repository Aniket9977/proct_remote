import cv2

def draw_face_landmarks(frame, landmarks):
    for (x, y) in [(pt.x, pt.y) for pt in landmarks.parts()]:
        cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
    return frame
