import os
import cv2
import pandas as pd
import numpy as np
import mediapipe as mp
from features import extract_features

# inicializa o face_mesh (uzado pra pegar as coordenadas dos rostos)
mp_face_mesh = mp.solutions.face_mesh

dataset_dir = 'datasetimagens/dataset'
output_csv = 'landmarks.csv'

data = []

# abre o mediapipe
with mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=False,
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

                features = extract_features(landmarks)
                data.append(features + [label])

# cria o dataframe e salva o csv
num_coords = len(data[0]) - 1
columns = [f'coord_{i}' for i in range(num_coords)] + ['label']
df = pd.DataFrame(data, columns=columns)
df.to_csv(output_csv, index=False)

print(f'foram extraidos {len(df)} rostos com sucesso')
