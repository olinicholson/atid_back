
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import resample
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer
import numpy as np

# Cargar datos
tweets_df = pd.read_csv('data/tweets_from_acc.csv')
tweets_df['Created At'] = pd.to_datetime(tweets_df['Created At'], format='%a %b %d %H:%M:%S %z %Y')

# Calcular engagement ratio
tweets_df['engagement_ratio'] = (
    tweets_df['Retweets'] + tweets_df['Likes'] + tweets_df['Quote_count'] + tweets_df['Reply_count']
) / tweets_df['View_count'] * 100

# Definir clases por percentiles
high = tweets_df['engagement_ratio'].quantile(0.75)
low = tweets_df['engagement_ratio'].quantile(0.25)

def categorize(ratio):
    if ratio >= high:
        return 'Alta'
    elif ratio >= low:
        return 'Media'
    else:
        return 'Baja'

tweets_df['target'] = tweets_df['engagement_ratio'].apply(categorize)

# Inicializar modelo de embeddings
embedder = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Generar embeddings para cada tweet
tweets_df['Text'] = tweets_df['Text'].fillna('').astype(str)
embeddings = embedder.encode(tweets_df['Text'].tolist())

# Agregar features adicionales
tweets_df['hour'] = tweets_df['Created At'].dt.hour
tweets_df['strategic_hour'] = tweets_df['hour'].apply(lambda h: 1 if 8 <= h <= 11 or 18 <= h <= 22 else 0)

# Combinar embeddings con features adicionales
X_embed = np.hstack((
    embeddings,
    tweets_df[['hour', 'strategic_hour']].values
))
y = tweets_df['target']

# Crear DataFrame combinado
df_full = pd.DataFrame(X_embed)
df_full['target'] = y.values

# Separar por clase
alta = df_full[df_full['target'] == 'Alta']
media = df_full[df_full['target'] == 'Media']
baja = df_full[df_full['target'] == 'Baja']

# Upsample clases chicas al tamaño de la más grande
max_n = max(len(alta), len(media), len(baja))
alta_up = resample(alta, replace=True, n_samples=max_n, random_state=42)
media_up = resample(media, replace=True, n_samples=max_n, random_state=42)
baja_up = resample(baja, replace=True, n_samples=max_n, random_state=42)

# Combinar y preparar datos
df_bal = pd.concat([alta_up, media_up, baja_up])
X_bal = df_bal.drop(columns='target').values
y_bal = df_bal['target'].values

# Entrenar modelo
X_train, X_test, y_train, y_test = train_test_split(X_bal, y_bal, test_size=0.2, stratify=y_bal, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Guardar modelo
joblib.dump(model, r'predictor\tweet_predictor_balanced_embed_model.pkl')
print("✅ Modelo balanceado con embeddings guardado como 'tweet_predictor_model.pkl'")

# Evaluación
print(f"✔️ Precisión general: {round(model.score(X_test, y_test) * 100, 2)}%")
y_pred = model.predict(X_test)
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred, digits=2))

# Matriz de confusión
print("📉 Matriz de confusión:")
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
df_cm = pd.DataFrame(cm, index=model.classes_, columns=model.classes_)
sns.heatmap(df_cm, annot=True, fmt="d", cmap="Blues")
plt.title("Matriz de Confusión (balanced + embeddings)")
plt.xlabel("Predicho")
plt.ylabel("Real")
plt.tight_layout()
plt.show()
