"""Build a features_cache.parquet and feature_meta.json for the Ualá models
without running the heavy training pipeline.

Usage (from repo root, with venv activated):
    python -m predictor.twitter.build_features_cache

This will read CSVs from core/data (posts_*.csv), compute rolling/time/text
features similar to the training pipeline and save the cache and metadata to
models_uala_v3_jumpboosted/.
"""
import json
import os
import glob
import sys
from datetime import datetime
import numpy as np
import pandas as pd

THIS_DIR = os.path.abspath(os.path.dirname(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

MODEL_DIR = os.path.join(THIS_DIR, "models_uala_v3_jumpboosted")
os.makedirs(MODEL_DIR, exist_ok=True)

# Data dir similar to pipeline
DATA_DIR = os.path.join(REPO_ROOT, "core", "data")

TARGETS = ["likes", "replies", "views"]
SEQ_LEN = 8

def read_post_files(data_dir):
    files = glob.glob(os.path.join(data_dir, "posts_*with_trends*.csv"))
    if len(files) == 0:
        files = glob.glob(os.path.join(data_dir, "posts_*.csv"))
    dfs = []
    for f in files:
        try:
            df_temp = pd.read_csv(f)
        except Exception as e:
            print(f"WARN: unable to read {f}: {e}")
            continue
        name = os.path.basename(f).split("_with_trends")[0].replace("posts_", "").replace('.csv','').lower()
        df_temp["dataset_name"] = name
        dfs.append(df_temp)
    if len(dfs) == 0:
        raise RuntimeError(f"No post files found in DATA_DIR={data_dir}")
    df_all = pd.concat(dfs, ignore_index=True)
    # coerce created_at
    df_all["created_at"] = pd.to_datetime(df_all["created_at"], errors="coerce")
    return df_all


def build_features(df):
    df = df.copy()
    # Filter for Ualá (same as training)
    df = df[df["dataset_name"].str.contains("uala", case=False, na=False)].copy()
    df = df.reset_index(drop=True)

    # temporal
    df["month_sin"] = np.sin(2 * np.pi * df["created_at"].dt.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["created_at"].dt.month / 12)
    df["hour"] = df["created_at"].dt.hour
    df["weekday"] = df["created_at"].dt.weekday
    df["is_weekend"] = (df["weekday"] >= 5).astype(int)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["is_morning"] = ((df["hour"] >= 6) & (df["hour"] < 12)).astype(int)
    df["is_afternoon"] = ((df["hour"] >= 12) & (df["hour"] < 18)).astype(int)
    df["is_evening"] = ((df["hour"] >= 18) & (df["hour"] < 22)).astype(int)
    df["is_night"] = ((df["hour"] >= 22) | (df["hour"] < 6)).astype(int)

    # feriados arg (conservative list)
    feriados_arg = [
        "2024-01-01", "2024-02-12", "2024-02-13", "2024-03-24", "2024-03-28",
        "2024-03-29", "2024-04-02", "2024-05-01", "2024-05-25", "2024-06-17",
        "2024-06-20", "2024-07-09", "2024-08-17", "2024-10-12", "2024-11-18",
        "2024-12-08", "2024-12-25", "2025-01-01", "2025-03-03", "2025-03-04",
        "2025-03-24", "2025-04-02", "2025-04-17", "2025-04-18", "2025-05-01",
        "2025-05-25", "2025-06-16", "2025-06-20", "2025-07-09", "2025-08-17",
        "2025-10-12", "2025-11-24", "2025-12-08", "2025-12-25"
    ]
    feriados_set = set(pd.to_datetime(feriados_arg).date)
    df["is_feriado"] = df["created_at"].dt.date.isin(feriados_set).astype(int)
    df["dias_desde_feriado"] = df["created_at"].apply(lambda x: min([abs((x.date() - f).days) for f in feriados_set]) if pd.notna(x) else 365)
    df["dias_hasta_feriado"] = df["created_at"].apply(lambda x: min([((f - x.date()).days) for f in feriados_set if f >= x.date()], default=365) if pd.notna(x) else 365)

    # text features
    df["text_length"] = df["text"].astype(str).str.len()
    df["word_count"] = df["text"].astype(str).str.split().str.len()
    df["has_hashtag"] = df["text"].astype(str).str.contains("#", na=False).astype(int)
    df["has_mention"] = df["text"].astype(str).str.contains("@", na=False).astype(int)
    df["has_excl"] = df["text"].astype(str).str.contains("!", na=False).astype(int)
    df = df.fillna(0)

    # Try to add social/competencia features if helper modules available
    try:
        from predictor.twitter.features_sociales import generar_features_sociales
        from predictor.twitter.features_competencia import generar_features_competencia
        df_top10 = pd.read_csv(os.path.join(DATA_DIR, "posts_top10.csv"))
        df_top10["created_at"] = pd.to_datetime(df_top10["created_at"], errors="coerce")
        df["date"] = df["created_at"]
        df = generar_features_sociales(df_user=df, df_top10=df_top10)

        comp_files = [f for f in glob.glob(os.path.join(DATA_DIR, "posts_*.csv")) if "uala" not in f.lower() and "top10" not in f.lower()]
        comp_dfs = []
        for f in comp_files:
            d = pd.read_csv(f)
            d["created_at"] = pd.to_datetime(d["created_at"], errors="coerce")
            d["dataset_name"] = os.path.basename(f).replace("posts_", "").replace(".csv", "")
            comp_dfs.append(d)
        if len(comp_dfs) > 0:
            df_comp = pd.concat(comp_dfs, ignore_index=True).dropna(subset=["created_at", "text"]) if comp_dfs else pd.DataFrame()
            df = generar_features_competencia(df_uala=df, df_comp=df_comp)
    except Exception:
        # Not critical — proceed without extra social/competencia features
        pass

    # ensure index by created_at for rolling computations
    df = df.sort_values("created_at").reset_index(drop=True)

    for tgt in TARGETS:
        s = df.set_index("created_at")[tgt]
        df[f"{tgt}_rollmed_30d"] = s.rolling("30D", min_periods=3).median().shift(1).reset_index(drop=True)
        df[f"{tgt}_ema_14d"] = s.ewm(span=14, min_periods=3, adjust=False).mean().shift(1).reset_index(drop=True)
        df[f"{tgt}_rel"] = df[tgt] / (1e-3 + df[f"{tgt}_rollmed_30d"])
        df[f"{tgt}_std_7d"] = s.rolling("7D", min_periods=2).std().shift(1).reset_index(drop=True).fillna(0)
        df[f"{tgt}_std_30d"] = s.rolling("30D", min_periods=5).std().shift(1).reset_index(drop=True).fillna(0)
        df[f"{tgt}_cv_30d"] = (df[f"{tgt}_std_30d"] / (df[f"{tgt}_rollmed_30d"] + 1e-3)).fillna(0)
        df[f"{tgt}_diff_1"] = s.diff(1).shift(1).reset_index(drop=True).fillna(0)
        df[f"{tgt}_diff_3"] = s.diff(3).shift(1).reset_index(drop=True).fillna(0)
        df[f"{tgt}_momentum_7d"] = s.rolling("7D", min_periods=2).apply(lambda x: (x[-1] - x[0]) if len(x) > 1 else 0).shift(1).reset_index(drop=True).fillna(0)
        df[f"{tgt}_range_7d"] = (s.rolling("7D", min_periods=2).max() - s.rolling("7D", min_periods=2).min()).shift(1).reset_index(drop=True).fillna(0)
        df[f"{tgt}_iqr_30d"] = (s.rolling("30D", min_periods=5).quantile(0.75) - s.rolling("30D", min_periods=5).quantile(0.25)).shift(1).reset_index(drop=True).fillna(0)

    # replies zscore
    try:
        s_replies = df.set_index("created_at")["replies"]
        roll_std = s_replies.rolling("30D", min_periods=5).std().shift(1).reset_index(drop=True)
        df["replies_zscore_30d"] = ((df["replies"] - df["replies_rollmed_30d"]) / (roll_std + 1e-3)).fillna(0)
    except Exception:
        df["replies_zscore_30d"] = 0

    # uncertainty temporal features
    df["days_since_last"] = df["created_at"].diff().dt.total_seconds() / 86400
    df["days_since_last"] = df["days_since_last"].fillna(df["days_since_last"].median())
    df["posting_irregularity"] = df["days_since_last"].rolling(7, min_periods=1).std().fillna(0)

    # jump intensity and is_jump
    df["jump_intensity"] = np.maximum.reduce([
        df["likes"] / (1e-3 + df["likes_rollmed_30d"]),
        df["replies"] / (1e-3 + df["replies_rollmed_30d"]),
        df["views"] / (1e-3 + df["views_rollmed_30d"])
    ])
    df["jump_intensity"] = np.log1p(df["jump_intensity"])
    df["jump_intensity"] = (df["jump_intensity"] - df["jump_intensity"].mean()) / (df["jump_intensity"].std() + 1e-6)
    df["is_jump"] = (
        (df["likes"] > df["likes_rollmed_30d"] * 2.5) |
        (df["replies"] > df["replies_rollmed_30d"] * 2.5) |
        (df["views"] > df["views_rollmed_30d"] * 2.0)
    ).astype(int)

    return df


def save_cache_and_meta(df, model_dir):
    cache_path = os.path.join(model_dir, "features_cache.parquet")
    meta_path = os.path.join(model_dir, "feature_meta.json")

    # Save cache
    try:
        df.to_parquet(cache_path, index=False)
        print(f"Saved features cache to {cache_path} (shape={df.shape})")
    except Exception as e:
        print(f"Failed to write parquet: {e}")

    # Build feature lists for meta
    exclude_cols = ["created_at", "text", "dataset_name", "hashtags", "date", "best_trend", "username", "is_jump", "jump_intensity"]
    feat_cols_base = [c for c in df.columns if c not in TARGETS + exclude_cols and df[c].dtype.kind in "if" and 'zscore' not in c]
    feat_cols_replies = feat_cols_base + ['replies_zscore_30d']

    feature_meta = {
        "seq_len": SEQ_LEN,
        "targets": {},
        "clf_features": feat_cols_base,
    }
    for tgt in TARGETS:
        if tgt == "replies":
            cols = feat_cols_replies
        else:
            cols = feat_cols_base
        feature_meta["targets"][tgt] = {
            "feat_cols": cols,
            "n_features": len(cols),
            "input_dim": len(cols) + 1,
            "hidden": 256,
            "head_hidden": 128,
        }

    try:
        with open(meta_path, "w", encoding="utf8") as fh:
            json.dump(feature_meta, fh, ensure_ascii=False, indent=2)
        print(f"Saved feature meta to {meta_path}")
    except Exception as e:
        print(f"Failed to write feature_meta.json: {e}")


def main():
    print("Building features cache...")
    df_all = read_post_files(DATA_DIR)
    df_features = build_features(df_all)
    save_cache_and_meta(df_features, MODEL_DIR)


if __name__ == "__main__":
    main()
