import numpy as np

def extract_features(landmarks):
    # transforma os landmarks em uma matriz (468 x 3)
    coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])

    # garante as distancias sejam normalizadas pela distancias dos olhos
    face_scale = np.linalg.norm(coords[263] - coords[33])
    if face_scale == 0:
        face_scale = 1

    def dist(idx1, idx2):
        return np.linalg.norm(coords[idx1] - coords[idx2]) / face_scale
    
    # calculo das distancias
    # LÁBIOS E BOCA
    mouth_high          = dist(13, 14)
    mouth_width         = dist(61, 291)
    mouth_proportion    = mouth_width / (mouth_high + 1e-6) # ruido pra prevenir divisao por 0

    # curvatura da boca
    lip_center_y        = coords[13][1]
    left_corner_y       = coords[61][1]
    right_corner_y      = coords[291][1]
    mouth_curve         = ((left_corner_y + right_corner_y) / 2.0 - lip_center_y) / face_scale

    # distancia dos cantos da boca até o queixo
    left_corner_chin    = dist(61, 152)
    right_corner_chin   = dist( 291, 152)
    mouth_chin_mean     = (left_corner_chin + right_corner_chin) / 2

    # OLHOS
    left_eye_high       = dist(159, 145)
    right_eye_high      = dist(386, 374)

    left_eye_width      = dist(33, 133)
    right_eye_width     = dist(362, 263)

    left_eye_prop       = left_eye_high / (left_eye_width + 1e-6)
    right_eye_prop      = right_eye_high / (right_eye_width + 1e-6)

    # SOBRANCELHAS
    left_eyebrow_eye    = dist(55, 159)
    right_eyebrow_eye   = dist(285, 386)
    eyebrow_distance    = dist(55, 285)

    # inclinação das sobrancelhas
    left_eyebrow_angle  = (coords[70][1] - coords[105][1]) / face_scale
    right_eyebrow_angle = (coords[300][1] - coords[334][1]) / face_scale
    

    return [
        mouth_high,
        mouth_width,
        mouth_proportion,
        mouth_curve,
        left_eyebrow_angle,
        right_eyebrow_angle,
        left_eyebrow_eye,
        right_eyebrow_eye,
        eyebrow_distance,
        left_eyebrow_angle,
        right_eyebrow_angle,
        left_corner_chin,
        right_corner_chin,
        mouth_chin_mean
    ]
