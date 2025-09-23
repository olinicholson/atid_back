
import sys
import os
import re
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from sentence_transformers import SentenceTransformer

# Cargar modelo entrenado
model = joblib.load(r'predictor\tweet_predictor_balanced_embed_model.pkl')

# Inicializar modelo de embeddings
embedder = SentenceTransformer('paraphrase-MiniLM-L6-v2')

def predict_tweet_performance(text: str, post_datetime: datetime) -> str:
    # Generar embedding
    embedding = embedder.encode([text])  # shape (1, 384)

    # Extra features
    hour = post_datetime.hour
    strategic_hour = 1 if 8 <= hour <= 11 or 18 <= hour <= 22 else 0

    # Combinar embedding + extras
    X = np.hstack((embedding, [[hour, strategic_hour]]))  # shape (1, 386)

    prediction = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    class_labels = model.classes_

    print("\n🔍 Datos analizados:")
    print(f"🕒 Hora: {hour} {'(estratégica)' if strategic_hour else ''}")
    print(f"📤 Vector de entrada: shape {X.shape}")
    print("\n📈 Probabilidad por clase:")
    for label, prob in zip(class_labels, proba):
        print(f"  - {label}: {round(prob * 100, 2)}%")

    return prediction

if __name__ == "__main__":
    text = input("✏️ Escribí el tweet: ")

    while True:
        hora_input = input("⏰ ¿Cuándo lo vas a publicar? (now o YYYY-MM-DD HH:MM): ").strip()
        if hora_input.lower() == "now":
            fecha = datetime.now()
            break
        try:
            fecha = datetime.strptime(hora_input, "%Y-%m-%d %H:%M")
            break
        except ValueError:
            print("⚠️ Formato incorrecto. Usá: YYYY-MM-DD HH:MM")

    resultado = predict_tweet_performance(text, fecha)
    print(f"\n✅ Predicción de performance: {resultado}")
