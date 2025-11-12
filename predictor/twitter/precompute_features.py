"""Precompute and save engineered features used by the predictor.

This script loads available posts CSVs, runs the same feature engineering as
`predictor_v3_jumpboosted.prepare_uala_dataframe` and saves a parquet file with
the processed DataFrame plus a JSON file listing feature columns. Run this once
after updating your `core/data/posts_*.csv` files to avoid expensive recompute
at prediction time.
"""
import os
import json
from pathlib import Path
import pandas as pd

from predictor.twitter.predictor_v3_jumpboosted import prepare_uala_dataframe, MODEL_DIR


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    print("Building feature cache using prepare_uala_dataframe()... this may take a while")
    # prepare_uala_dataframe will read core/data CSVs by default
    df = prepare_uala_dataframe()

    # determine feature columns used in training (same logic as pipeline)
    exclude_cols = ["created_at", "text", "dataset_name", "hashtags", "date", "best_trend", "username", "is_jump", "jump_intensity"]
    feat_cols_base = [c for c in df.columns if c not in ["likes", "replies", "views"] + exclude_cols
                      and df[c].dtype.name in ("int64", "float64", "float32", "bool") and 'zscore' not in c]

    # Save a compact cache with only the columns needed + meta
    cache_cols = feat_cols_base + ["jump_intensity", "is_jump", "created_at", "text", "likes", "replies", "views"]
    cache_cols = [c for c in cache_cols if c in df.columns]
    df_cache = df[cache_cols].copy()

    cache_path = Path(MODEL_DIR) / "features_cache.parquet"
    df_cache.to_parquet(cache_path, index=False)
    print(f"Saved feature cache to {cache_path}")

    feat_list_path = Path(MODEL_DIR) / "feature_names.json"
    with open(feat_list_path, "w", encoding="utf8") as f:
        json.dump(feat_cols_base, f, ensure_ascii=False, indent=2)
    print(f"Saved feature list to {feat_list_path}")


if __name__ == "__main__":
    main()
