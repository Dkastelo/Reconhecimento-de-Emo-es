import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# ========== inicialização dos dados ==========
df = pd.read_csv("landmarks.csv")

X = df.drop(columns=['label'])
y = df['label']

# dividindo o treino e teste (80% e 20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# define o pipeline
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(kernel='rbf', probability=True))
])

# ========== GridSearch ==========
#define a grade que queremos testar
param_grid = {
    'svm__C'     : [0.1, 1, 10, 100],
    'svm__gamma' : ['scale', 'auto', 0.1, 0.01]
}

# configura o gridsearch
print('inicializando o GridSearch com o SVM')
grid = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=2
)

grid.fit(X_train, y_train)

print('\n melhores hiperparâmetros encontrados:')
print(grid.best_params_)

# ========== avalia e salva ==========
# avalia a precisão do modelo vencedor
best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)

print(f'\n acurácia final no teste: {accuracy_score(y_test, y_pred)*100:.2f}%')
print(classification_report(y_test, y_pred))

# salva o modelo treinado
joblib.dump(best_model, 'emotion_svm.pkl')
print("modelo salvo como 'emotion_svm.pkl'!")
