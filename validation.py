import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ========== inicialização dos dados ==========
df = pd.read_csv('landmarks_val.csv')

X = df.drop(columns=['label'])
y = df['label']

# ========== predição ==========
x = 2
if x == 1:
    model = joblib.load('SVM/emotion_svm.pkl')
elif x == 2:
    model = joblib.load('ANN/emotion_ann.pkl')

y_pred = model.predict(X)

print('=-=' * 15)
print(f'Acurácia na validação: {accuracy_score(y, y_pred) * 100:.2f}%')

print('\n Relatório de Classificação:')
print(classification_report(y, y_pred))

# ========== matriz de confusão ==========
labels = sorted(y.unique())
cm = confusion_matrix(y, y_pred, labels=labels)

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm, 
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=labels, 
    yticklabels=labels
)

plt.title('matriz de confusão da validação')
plt.xlabel('predição do modelo')
plt.ylabel('classe real')
plt.tight_layout()
plt.show()

print('sucesso')