import numpy as np

def get_aspect_ratio(eye_points):
    A = np.linalg.norm(np.array([eye_points[1].x, eye_points[1].y]) - np.array([eye_points[5].x, eye_points[5].y]))
    B = np.linalg.norm(np.array([eye_points[2].x, eye_points[2].y]) - np.array([eye_points[4].x, eye_points[4].y]))
    C = np.linalg.norm(np.array([eye_points[0].x, eye_points[0].y]) - np.array([eye_points[3].x, eye_points[3].y]))
    aspect_ratio = (A + B) / (2.0 * C)
    return aspect_ratio
