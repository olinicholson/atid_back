import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA


# ------------------------
# Función de predicción en tiempo real para Ualá
# ------------------------
def predict_tweet(texto, fecha, company="uala", has_image=False, has_offer=False, has_trend=False, model_path="xgboost_retweets_predictor_uala_timeval.pkl", info_path="model_info_uala_timeval.json"):
    # Cargar modelo y metadata
    model = joblib.load(model_path)
    with open(info_path, "r") as f:
        info = json.load(f)

    features = info["features"]
    avg_rt_per_company = info["avg_rt_per_company"]

    # Procesar fecha
    fecha = pd.to_datetime(fecha)
    year, month, day, hour, weekday = fecha.year, fecha.month, fecha.day, fecha.hour, fecha.weekday()
    is_weekend = 1 if weekday >= 5 else 0
    is_golden_hour = 1 if hour in [8,9,10,18,19,20] else 0

    # Longitud del texto
    text_length = len(texto)

    # Embeddings (usar el mismo modelo que entrenamiento)
    emb_model = SentenceTransformer(info["embedding_model"])
    emb_vector = emb_model.encode([texto])
    pca = PCA(n_components=len([c for c in features if c.startswith("text_emb_")]), random_state=42)
    # ⚠️ En producción deberías cargar el PCA entrenado, acá lo simplifico para demo
    pca = joblib.load("pca_embeddings.pkl")
    emb_pca = pca.transform(emb_vector)
    # Construir feature row
    row = {
        "has_offer": int(has_offer),
        "text_length": text_length,
        "day": day,
        "month": month,
        "year": year,
        "hour": hour,
        "is_golden_hour": is_golden_hour,
        "has_trend": int(has_trend),
        "has_image": int(has_image),
        "is_weekend": is_weekend,
        "weekday": weekday,
    }
    row.update(emb_features)

    # Agregar columnas faltantes (clusters, contexto top10, etc.)
    for f in features:
        if f not in row:
            row[f] = 0

    # Crear DataFrame
    X_new = pd.DataFrame([row])[features]

    # Predicción
    pred_log = model.predict(X_new)[0]
    pred_norm = np.expm1(max(pred_log, 0))  # veces baseline
    pred_abs = pred_norm * avg_rt_per_company.get(company, 1)

    return {
        "texto": texto,
        "company": company,
        "pred_norm": round(float(pred_norm), 2),
        "pred_abs": round(float(pred_abs), 2),
        "baseline_company": round(avg_rt_per_company.get(company, 0), 2)
    }

resultado = predict_tweet(
    texto="🎉 Lanzamos nuestra nueva tarjeta con beneficios exclusivos",
    fecha="2025-09-25 20:00:00",
    company="uala",
    has_image=True,
    has_offer=True,
    has_trend=True
)
print(resultado)

ejemplos_idx = [0, 10, 20]  # elegimos 3 índices de prueba
print("\n🔮 Ejemplos de predicción:")
for idx in ejemplos_idx:
    if idx >= len(X_test):
        continue
    X_row = X_test.iloc[[idx]]
    true_val = np.expm1(y_test.iloc[idx])  # valor real (deslog)
    pred_val = np.expm1(xgb_model.predict(X_row)[0])  # predicho (deslog)

    print(f"\nEjemplo {idx}:")
    print(f"Texto: {test_df.iloc[idx]['text'][:100]}...")  # preview del texto
    print(f"Empresa: {test_df.iloc[idx]['company']}")
    print(f"Retweets reales: {true_val:.1f}")
    print(f"Predicción:     {pred_val:.1f}")