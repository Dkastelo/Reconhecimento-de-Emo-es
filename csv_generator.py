import os
import cv2
import pandas as pd
import numpy as np
import mediapipe as mp

def normalize_landmarks(landmarks):
    # converte para array numpy de formato (468, 3)
    coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
    
    # centralizando no ponto 1 do MediaPipe (a ponta do nariz)
    center = coords[1]
    coords = coords - center
    
    # normalização usando os pontos 33 (olho esquerdo) e 263 (olho direito)
    # calculamos a distância euclidiana entre os dois olhos
    eye_left = coords[33]
    eye_right = coords[263]
    face_size = np.linalg.norm(eye_right - eye_left)
    
    # evita divisão por zero se a detecção falhar pontualmente
    if face_size > 0:
        coords = coords / face_size
        
    return coords.flatten()  # Retorna de volta como vetor de 1404 posições


# inicializa o face_mesh (uzado pra pegar as coordenadas dos rostos)
mp_face_mesh = mp.solutions.face_mesh

dataset_dir = 'datasetimagens/dataset'
output_csv = 'landmarks.csv'

data = []

# abre o mediapipe
with mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
) as face_mesh:
    classes = [f for f in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, f))]
    print(f'classes encontradas: {classes}')

    for label in classes:
        class_path = os.path.join(dataset_dir, label)
        images = os.listdir(class_path)
        print(f"Processando '{label}' ({len(images)} imagens)...")

        for img_name in images:
            img_path = os.path.join(class_path, img_name)
            image = cv2.imread(img_path)

            if image is None:
                continue
                
            # mediapipe exibe imagem em RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_image)

            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                norm_coords = normalize_landmarks(landmarks)
                data.append(list(norm_coords) + [label])

# cria o dataframe e salva o csv
num_coords = len(data[0]) - 1
columns = [f'coord_{i}' for i in range(num_coords)] + ['label']
df = pd.DataFrame(data, columns=columns)
df.to_csv(output_csv, index=False)

print(f'foram extraidos {len(df)} rostos com sucesso')
