import cv2
import mediapipe as mp
import numpy as np
import joblib
from csv_generator import normalize_landmarks

# carrega o modelo treinado, nesse caso o svm
model = joblib.load('SVM/emotion_svm.pkl')

# inicializa o mediapipe e a câmera
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

cap = cv2.VideoCapture(0)

with mp_face_mesh.FaceMesh(
    max_num_faces = 1,
    refine_landmarks = True,
    min_detection_confidence = 0.5,
    min_tracking_confidence = 0.5
) as face_mesh:

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # inverte horizontelmente para efeito de espelho
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:

                norm_coords = normalize_landmarks(face_landmarks.landmark)
                coords_array = norm_coords.reshape(1, -1)

                emotion = model.predict(coords_array)[0]
                probability = np.max(model.predict_proba(coords_array)) * 100

                # desenha a malha do rosto na tela
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                )

                # exibe o resultado na tela
                text = f'{emotion.upper()} ({probability:.1f}%)'
                cv2.putText(frame, text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow('Reconhecimento de Expressoes', frame)

        if cv2.waitKey(5) & 0xFF == 27: # Pressione ESC para sair
            break

cap.release()
cv2.destroyAllWindows()