import dlib

def get_face_detector():
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_model\shape_predictor_68_face_landmarks.dat")
    return detector, predictor
