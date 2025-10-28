"""
Predictor personalizado de engagement rate para Ualá
Entrena un modelo global ponderado con datos de competencia + top10
y calcula el rendimiento esperado de Ualá frente al resto.
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
import warnings
warnings.filterwarnings("ignore")

# -----------------------
# CONFIG
# -----------------------
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "core", "data")

UALA_KEYWORDS = ["uala"]             # tus datos
TOP10_KEYWORDS = ["top10"]           # benchmarks
COMPETENCIA_KEYWORDS = ["brubank", "balanz", "supervielle", "galicia", "cocos"]

# -----------------------
# FUNCIONES AUXILIARES
# -----------------------
def load_all_datasets():
    """Combina todos los posts_*with_trends*.csv"""
    files = glob.glob(os.path.join(DATA_DIR, "posts_*with_trends*.csv"))
    if not files:
        raise FileNotFoundError("No se encontraron archivos en core/data")
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
    """Crea features tabulares + engagement_rate_log"""
    df = df.copy()
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    df = df.dropna(subset=["created_at", "text"])

    df["engagement_rate"] = (
        df["likes"].fillna(0) + df["retweets"].fillna(0) + df["replies"].fillna(0)
    ) / (df["views"].fillna(1) + 1)
    df["engagement_rate"] = df["engagement_rate"].clip(upper=df["engagement_rate"].quantile(0.99))
    df["engagement_rate_log"] = np.log1p(df["engagement_rate"])

    df["hour"] = df["created_at"].dt.hour
    df["weekday"] = df["created_at"].dt.weekday
    df["is_weekend"] = (df["weekday"] >= 5).astype(int)
    df["text_length"] = df["text"].astype(str).str.len()
    df["has_offer"] = df["text"].astype(str).str.contains("oferta|promo|descuento|gratis|%|regalo", case=False, na=False).astype(int)
    df["has_image"] = df["text"].astype(str).str.contains("pic.twitter.com|instagram|https://t.co", case=False, na=False).astype(int)
    df["has_url"] = df["text"].astype(str).str.contains("http", na=False).astype(int)
    df["has_hashtag"] = df["text"].astype(str).str.contains("#", na=False).astype(int)
    df["word_count"] = df["text"].astype(str).str.split().str.len()
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
    """Asigna mayor peso a Ualá y top10"""
    weights = np.ones(len(df))
    weights[df["dataset_name"].str.contains("|".join(UALA_KEYWORDS), case=False, na=False)] = 3.0
    weights[df["dataset_name"].str.contains("|".join(TOP10_KEYWORDS), case=False, na=False)] = 2.0
    return weights

# -----------------------
# MAIN
# -----------------------
def main():
    print("\n🚀 Entrenando predictor personalizado para Ualá...\n")
    df = load_all_datasets()
    df = feature_engineering(df)

    print("📥 Cargando modelo de embeddings...")
    emb_model = SentenceTransformer(EMBEDDING_MODEL)

    print("🔢 Generando embeddings de texto...")
    X_text = encode_texts(emb_model, df["text"].astype(str))
    tab_features = [
        "hour","weekday","is_weekend","text_length","has_offer","has_image",
        "has_url","has_hashtag","word_count"
    ]
    X_tab = df[tab_features].values
    scaler = StandardScaler()
    X_tab = scaler.fit_transform(X_tab)

    X = np.hstack([X_text, X_tab])
    y = df["engagement_rate_log"].values
    weights = compute_weights(df)

    print(f"\n📐 Features: {X.shape}, Target mean(log): {y.mean():.3f}")
    print(f"⚖️ Pesos asignados → Ualá:3x, Top10:2x, otros:1x\n")

    print("⚙️ Entrenando modelo XGBoost ponderado...")
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

    # --- Evaluación global ---
    preds = model.predict(X)
    r2 = r2_score(y, preds)
    mae = mean_absolute_error(y, preds)
    corr = spearmanr(y, preds).correlation
    print(f"📈 Evaluación global → R²={r2:.3f}, MAE={mae:.3f}, Spearman={corr:.3f}\n")

    # --- Ranking comparativo ---
    df["pred_log"] = preds
    df["pred_engagement"] = np.expm1(df["pred_log"])

    resumen = []
    for name, group in df.groupby("dataset_name"):
        resumen.append({
            "Cuenta": name,
            "Posts": len(group),
            "Promedio real": np.expm1(group["engagement_rate_log"]).mean(),
            "Promedio predicho": group["pred_engagement"].mean()
        })
    rank = pd.DataFrame(resumen).sort_values(by="Promedio predicho", ascending=False)
    print("🏁 Ranking comparativo de engagement promedio:\n")
    print(rank.round(4).to_string(index=False))

    if any(df["dataset_name"].str.contains("uala", case=False)):
        uala_row = rank[rank["Cuenta"].str.contains("uala", case=False)]
        pos = rank.index.get_loc(uala_row.index[0]) + 1
        print(f"\n🏆 Ualá está en posición #{pos} de {len(rank)} con engagement promedio predicho = {uala_row['Promedio predicho'].values[0]:.4f}\n")

if __name__ == "__main__":
    main()
