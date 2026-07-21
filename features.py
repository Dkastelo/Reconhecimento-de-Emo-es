import numpy as np

def extract_features(landmarks):
    # transforma os landmarks em uma matriz (468 x 3)
    coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])

    # garante as distancias sejam normalizadas pela distancias dos olhos
    face_scale = np.linalg.norm(coords[263] - coords[33])
    if face_scale == 0:
        face_scale = 1
    
    # calculo das distancias
    mouth_high = np.linalg.norm(coords[13] - coords[14]) / face_scale
    mouth_width = np.linalg.norm(coords[61] - coords[291]) / face_scale

    left_eye_high = np.linalg.norm(coords[159] - coords[291]) / face_scale
    right_eye_high = np.linalg.norm(coords[386] - coords[374]) / face_scale

    left_eyebrow_eye = np.linalg.norm(coords[55] - coords[159]) / face_scale
    right_eyebrow_eye = np.linalg.norm(coords[285] - coords[386]) / face_scale
    eyebrow_distance = np.linalg.norm(coords[55] - coords[285]) / face_scale

    mouth_proportion = mouth_width / (mouth_high + 1e-6)

    return [
        mouth_high,
        mouth_width,
        mouth_proportion,
        left_eye_high,
        right_eye_high,
        left_eyebrow_eye,
        right_eyebrow_eye,
        eyebrow_distance
    ]
