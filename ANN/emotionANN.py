import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score

# inicialização dos dados
df = pd.read_csv('landmarks.csv')
X = df.drop(columns=['label'])
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size = 0.15,
    random_state = 42,
    stratify = y
)

# pipeline
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('ann', MLPClassifier(max_iter=500, random_state=42, early_stopping=True))
])

# gridsearch
param_grid = {
    'ann__hidden_layer_sizes': [(64, 32), (128, 64), (64, 64, 32)],      # estrutura das camadas
    'ann__activation': ['relu', 'tanh'],                                # função de ativação
    'ann__alpha': [0.0001, 0.001, 0.01],                                # regularização l2
    'ann__solver': ['adam']                                             # otimizador
}

print('iniciando o treino da ANN')
grid = GridSearchCV(
    pipe,
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

grid.fit(X_train, y_train)
best_model = grid.best_estimator_

y_pred = best_model.predict(X_test)

print(f'Melhores Hiperparametros: {grid.best_params_}')
print(f'Acurácia da ANN no teste: {accuracy_score(y_test, y_pred)*100:.2f}%')

print(classification_report(y_test, y_pred))

joblib.dump(best_model, 'ANN/emotion_ann.pkl')
print('modelo ANN salvo com sucesso')
