import os
import pickle
from typing import Optional, Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

from pathlib import Path
import sys
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from predictor.twitter.predictor_v3_jumpboosted import MODEL_DIR, DATA_DIR

os.makedirs(MODEL_DIR, exist_ok=True)
QUICK_PKL = os.path.join(MODEL_DIR, "quick_models.pkl")


def _featurize_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["text"] = df.get("text", "").astype(str)
    df["text_len"] = df["text"].str.len()
    df["word_count"] = df["text"].str.split().str.len().fillna(0)
    df["hashtag_count"] = df["text"].str.count("#").fillna(0)
    df["mention_count"] = df["text"].str.count("@").fillna(0)
    if "created_at" in df.columns:
        df["created_at_parsed"] = pd.to_datetime(df["created_at"].astype(str), errors="coerce")
        df["hour"] = df["created_at_parsed"].dt.hour.fillna(0).astype(int)
        df["weekday"] = df["created_at_parsed"].dt.weekday.fillna(0).astype(int)
    else:
        df["hour"] = 12
        df["weekday"] = 0
    return df[["text_len", "word_count", "hashtag_count", "mention_count", "hour", "weekday"]]


def train_quick_models(files: Optional[list] = None) -> Dict:
    if files is None:
        files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
        files = [os.path.join(DATA_DIR, f) for f in files]
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            dfs.append(df)
        except Exception:
            continue
    if not dfs:
        raise RuntimeError("No data files found to train quick predictor")
    df_all = pd.concat(dfs, ignore_index=True)
    if "dataset_name" in df_all.columns:
        df = df_all[df_all["dataset_name"].str.contains("uala", case=False, na=False)].copy()
    else:
        df = df_all.copy()

    X = _featurize_df(df)
    models = {}
    targets = [t for t in ["likes", "replies", "views"] if t in df.columns]
    for t in targets:
        y = df[t].fillna(0).astype(float).values
        if len(y) < 20:
            continue
        X_train, X_val, y_train, y_val = train_test_split(X.values, y, test_size=0.2, random_state=42)
        m = GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42)
        m.fit(X_train, y_train)
        models[t] = m

    with open(QUICK_PKL, "wb") as f:
        pickle.dump({"models": models}, f)

    return models


def _load_models():
    if os.path.exists(QUICK_PKL):
        with open(QUICK_PKL, "rb") as f:
            pk = pickle.load(f)
            return pk.get("models", {})
    return {}


def predict_single(text: str, created_at_iso: Optional[str] = None) -> Dict:
    models = _load_models()
    if not models:
        try:
            models = train_quick_models()
        except Exception:
            base = max(1, len(str(text)) // 5)
            return {"predictions": {"likes": {"q01": int(base*0.6), "q50": int(base), "q99": int(base*1.5)}}, "meta": {"note": "heuristic"}}

    row = {"text": text, "created_at": created_at_iso}
    df_row = pd.DataFrame([row])
    X_row = _featurize_df(df_row).values

    out = {}
    for t, m in models.items():
        try:
            pred = float(m.predict(X_row)[0])
        except Exception:
            pred = 0.0
        low = max(0, int(round(pred * 0.7)))
        med = max(0, int(round(pred)))
        high = max(0, int(round(pred * 1.3 + 1)))
        out[t] = {"q01": low, "q50": med, "q99": high}

    return {"predictions": out, "meta": {"model": "quick_gbr"}}


if __name__ == "__main__":
    print(predict_single("Prueba de predictor rapido #test", "2025-11-10T12:00:00"))
