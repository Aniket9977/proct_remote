import numpy as np

def get_mouth_aspect_ratio(mouth_points):
    A = np.linalg.norm(np.array([mouth_points[13].x, mouth_points[13].y]) - np.array([mouth_points[19].x, mouth_points[19].y]))
    B = np.linalg.norm(np.array([mouth_points[14].x, mouth_points[14].y]) - np.array([mouth_points[18].x, mouth_points[18].y]))
    C = np.linalg.norm(np.array([mouth_points[12].x, mouth_points[12].y]) - np.array([mouth_points[16].x, mouth_points[16].y]))
    mar = (A + B) / (2.0 * C)
    return mar
