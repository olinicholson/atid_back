"""
Predictor multitarget de engagement para Ualá
Predice likes, replies y views por separado con XGBoost
"""

import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.stats import spearmanr
from xgboost import XGBRegressor
import joblib
import warnings
warnings.filterwarnings("ignore")

# -----------------------
# CONFIG
# -----------------------
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "core", "data")

UALA_KEYWORDS = ["brubank"]    # tus datos
TOP10_KEYWORDS = ["top10"]
COMPETENCIA_KEYWORDS = ["uala", "balanz", "supervielle", "galicia", "cocos"]
TARGETS = ["likes", "replies", "views"]

# -----------------------
# FUNCIONES
# -----------------------
def load_all_datasets():
    files = glob.glob(os.path.join(DATA_DIR, "posts_*with_trends*.csv"))
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            df["dataset_name"] = os.path.basename(f).split("_with_trends")[0].replace("posts_", "")
            dfs.append(df)
        except Exception as e:
            print(f"⚠️ Error leyendo {f}: {e}")
    df_all = pd.concat(dfs, ignore_index=True)
    print(f"✅ {len(files)} datasets combinados ({len(df_all)} posteos totales)")
    return df_all

def feature_engineering(df):
    df = df.copy()
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    df = df.dropna(subset=["created_at", "text"])
    df["hour"] = df["created_at"].dt.hour
    df["weekday"] = df["created_at"].dt.weekday
    df["is_weekend"] = (df["weekday"] >= 5).astype(int)
    df["text_length"] = df["text"].astype(str).str.len()
    df["word_count"] = df["text"].astype(str).str.split().str.len()
    df["has_offer"] = df["text"].astype(str).str.contains("oferta|promo|descuento|gratis|%|regalo", case=False, na=False).astype(int)
    df["has_image"] = df["text"].astype(str).str.contains("pic.twitter.com|instagram|https://t.co", case=False, na=False).astype(int)
    df["has_url"] = df["text"].astype(str).str.contains("http", na=False).astype(int)
    df["has_hashtag"] = df["text"].astype(str).str.contains("#", na=False).astype(int)
    df = df.fillna(0)
    return df

def encode_texts(model, texts):
    embeddings = []
    for i in tqdm(range(0, len(texts), 64), desc="🧠 Embeddings"):
        batch = texts[i:i+64].tolist()
        emb = model.encode(batch, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
        embeddings.append(emb)
    return np.vstack(embeddings)

def compute_weights(df):
    weights = np.ones(len(df))
    weights[df["dataset_name"].str.contains("|".join(UALA_KEYWORDS), case=False, na=False)] = 3.0
    weights[df["dataset_name"].str.contains("|".join(TOP10_KEYWORDS), case=False, na=False)] = 2.0
    return weights

# -----------------------
# MAIN ENTRENAMIENTO
# -----------------------
def main():
    print("\n🚀 Entrenando modelos multitarget para Ualá...\n")
    df = load_all_datasets()
    df = feature_engineering(df)

    emb_model = SentenceTransformer(EMBEDDING_MODEL)
    print("🔢 Generando embeddings de texto...")
    X_text = encode_texts(emb_model, df["text"].astype(str))

    tab_features = [
        "hour","weekday","is_weekend","text_length","word_count",
        "has_offer","has_image","has_url","has_hashtag"
    ]
    X_tab = df[tab_features].values
    scaler = StandardScaler()
    X_tab = scaler.fit_transform(X_tab)
    X = np.hstack([X_text, X_tab])
    weights = compute_weights(df)

    results = {}
    for target in TARGETS:
        print(f"\n⚙️ Entrenando modelo para {target.upper()}...")
        y = np.log1p(df[target].fillna(0).values)
        model = XGBRegressor(
            n_estimators=600,
            max_depth=6,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            tree_method="hist",
            random_state=42
        )
        model.fit(X, y, sample_weight=weights)
        preds = model.predict(X)
        r2 = r2_score(y, preds)
        mae = mean_absolute_error(y, preds)
        corr = spearmanr(y, preds).correlation
        print(f"📈 {target.upper()} → R²={r2:.3f}, MAE={mae:.3f}, Spearman={corr:.3f}")
        joblib.dump(model, f"uala_model_{target}.pkl")
        results[target] = {"R²": r2, "MAE": mae, "Spearman": corr}
    joblib.dump(scaler, "uala_scaler.pkl")
    print("\n✅ Modelos guardados: uala_model_likes.pkl / replies / views\n")
    print(pd.DataFrame(results).T.round(3))

# -----------------------
# SIMULADOR DE POST
# -----------------------
def simular_post(text, hour=12, has_image=True, has_hashtag=True, has_url=False, has_offer=False):
    """Usa los modelos entrenados para estimar likes, replies y views"""
    emb_model = SentenceTransformer(EMBEDDING_MODEL)
    scaler = joblib.load("uala_scaler.pkl")

    # cargar modelos individuales
    models = {
        "likes": joblib.load("uala_model_likes.pkl"),
        "replies": joblib.load("uala_model_replies.pkl"),
        "views": joblib.load("uala_model_views.pkl")
    }

    df = pd.DataFrame([{
        "hour": hour,
        "weekday": 3,
        "is_weekend": int(hour in [6,7]),
        "text_length": len(text),
        "word_count": len(text.split()),
        "has_offer": int(has_offer),
        "has_image": int(has_image),
        "has_url": int(has_url),
        "has_hashtag": int(has_hashtag)
    }])

    X_tab = scaler.transform(df.values)
    emb = emb_model.encode([text], show_progress_bar=False, normalize_embeddings=True)
    X = np.hstack([emb, X_tab])

    preds = {}
    for target, model in models.items():
        y_pred = np.expm1(model.predict(X))[0]
        preds[target] = round(float(y_pred), 2)
    return preds

if __name__ == "__main__":
    main()
    # Ejemplo: probar el simulador
    example = simular_post("Nueva promo para ahorrar con tu tarjeta Ualá", hour=13, has_image=True, has_hashtag=True)
    print("\n💬 Simulación ejemplo:\n", example)
