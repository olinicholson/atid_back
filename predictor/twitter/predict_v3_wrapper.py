"""Wrapper script to train v3 models (if needed) and run a forward prediction for a single tweet.

Usage:
  - import predict_v3_wrapper and call `ensure_trained()` to train models if not present
  - call `predict_single(text, created_at_iso)` to get quantile predictions

This reuses the feature engineering from predictor_v3_jumpboosted.prepare_uala_dataframe
and the QuantileLSTM class saved weights.
"""
import os
import pickle
import glob
import json
import numpy as np
import pandas as pd
import torch
import sys

# Ensure the top-level `predictor` package dir is on sys.path so imports like
# `from features_sociales import ...` inside modules in this package resolve.
PREDICTOR_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PREDICTOR_ROOT not in sys.path:
    sys.path.insert(0, PREDICTOR_ROOT)
# Also ensure the current directory (predictor/twitter) is on sys.path so
# local modules like `predictor_v3_jumpboosted.py` can be imported by name
# (legacy imports in this file expect that).
THIS_DIR = os.path.abspath(os.path.dirname(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from predictor_v3_jumpboosted import run_pipeline, MODEL_DIR, DATA_DIR, SEQ_LEN, TARGETS


def ensure_trained(train_if_missing: bool = False):
    """Ensure model artifacts exist.

    By default this function will NOT start the training pipeline; it only
    checks for the presence of required artifacts and returns a boolean.

    If `train_if_missing=True` the heavy `run_pipeline()` will be invoked to
    train and persist models when artifacts are not found (legacy behaviour).
    """
    pkl_path = os.path.join(MODEL_DIR, "jump_models.pkl")
    # LSTM weight file for likes
    lstm_path = os.path.join(MODEL_DIR, "uala_likes_lstm.pt")
    if os.path.exists(pkl_path) and os.path.exists(lstm_path):
        return True

    if not train_if_missing:
        # Do not trigger heavy training as a side-effect of prediction calls.
        return False

    # Legacy: if caller explicitly requests training, run the full pipeline.
    from predictor_v3_jumpboosted import run_pipeline
    run_pipeline()
    return True


def predict_single(text: str, created_at_iso: str, force_is_jump: int = None, last_published_at: str = None, recent_posts: list = None):
    """Predict quantiles for a single tweet using trained v3 models.

    Returns a dict with q01/q50/q99 for likes (and placeholders for replies/views if model missing).
    If models are missing, raises an error.
    """
    # ensure model artifacts exist: jump_models.pkl + per-target LSTM weights
    pkl_path = os.path.join(MODEL_DIR, "jump_models.pkl")
    lstm_files = {
        'likes': os.path.join(MODEL_DIR, "uala_likes_lstm.pt"),
        'replies': os.path.join(MODEL_DIR, "uala_replies_lstm.pt"),
        'views': os.path.join(MODEL_DIR, "uala_views_lstm.pt"),
    }
    missing = []
    if not os.path.exists(pkl_path):
        missing.append('jump_models.pkl')
    for k, p in lstm_files.items():
        if not os.path.exists(p):
            missing.append(os.path.basename(p))
    if missing:
        raise RuntimeError(f"Model artifacts not found (missing: {', '.join(missing)}). Ensure models are trained and present in {MODEL_DIR}.")

    # Prefer using a precomputed feature cache (fast). If not present, fall back
    # to the full (heavier) prepare_uala_dataframe pipeline which computes
    # features from CSVs.
    cache_path = os.path.join(MODEL_DIR, "features_cache.parquet")
    feat_list_path = os.path.join(MODEL_DIR, "feature_names.json")
    extra_row = {"text": text, "created_at": created_at_iso, "likes": 0, "replies": 0, "views": 0}

    # If caller provided last_published_at or recent_posts, use them to
    # compute posting cadence features (days_since_last, posting_irregularity)
    try:
        # recent_posts expected as list of dicts: [{"created_at": iso, "likes": int, "replies": int, "views": int}, ...]
        if recent_posts is None:
            # If last_published_at provided, try to load last 3 from Ualá CSV prior to that timestamp.
            # Otherwise, use the latest 3 records in posts_uala.csv.
            try:
                posts_path = os.path.join(DATA_DIR, "posts_uala.csv")
                if os.path.exists(posts_path):
                    df_posts = pd.read_csv(posts_path)
                    if "created_at" in df_posts.columns:
                        df_posts["created_at"] = pd.to_datetime(df_posts["created_at"], errors="coerce")
                        if last_published_at is not None:
                            try:
                                last_dt = pd.to_datetime(last_published_at)
                                df_candidates = df_posts[df_posts["created_at"] < last_dt].sort_values("created_at", ascending=False)
                            except Exception:
                                df_candidates = df_posts.sort_values("created_at", ascending=False)
                        else:
                            df_candidates = df_posts.sort_values("created_at", ascending=False)
                        recent_df = df_candidates.head(3)
                        if len(recent_df) > 0:
                            recent_posts = []
                            for _, r in recent_df.iterrows():
                                recent_posts.append({
                                    "created_at": r.get("created_at").isoformat() if not pd.isna(r.get("created_at")) else None,
                                    "likes": int(r.get("likes", 0) or 0),
                                    "replies": int(r.get("replies", 0) or 0),
                                    "views": int(r.get("views", 0) or 0),
                                })
            except Exception:
                recent_posts = None

        # If we have recent_posts now, compute days_since_last and posting_irregularity
        if recent_posts and len(recent_posts) > 0:
            try:
                # parse created_at values
                times = []
                for rp in recent_posts:
                    if rp.get("created_at"):
                        times.append(pd.to_datetime(rp.get("created_at")))
                if len(times) > 0:
                    times_sorted = sorted(times)
                    # last published is most recent of these
                    last_pub = times_sorted[-1]
                    created_at_dt = pd.to_datetime(created_at_iso)
                    extra_row["days_since_last"] = (created_at_dt - last_pub).total_seconds() / 86400.0
                    # posting_irregularity: std dev of inter-post days
                    if len(times_sorted) >= 2:
                        deltas = []
                        for i in range(1, len(times_sorted)):
                            deltas.append((times_sorted[i] - times_sorted[i-1]).total_seconds() / 86400.0)
                        extra_row["posting_irregularity"] = float(pd.Series(deltas).std())
                    else:
                        extra_row["posting_irregularity"] = 0.0
                    # provide simple rolling medians as seed values for the backend features
                    likes_vals = [int(rp.get("likes", 0) or 0) for rp in recent_posts]
                    replies_vals = [int(rp.get("replies", 0) or 0) for rp in recent_posts]
                    views_vals = [int(rp.get("views", 0) or 0) for rp in recent_posts]
                    if len(likes_vals) > 0:
                        extra_row["likes_rollmed_30d"] = float(pd.Series(likes_vals).median())
                        extra_row["likes_ema_14d"] = float(pd.Series(likes_vals).mean())
                    if len(replies_vals) > 0:
                        extra_row["replies_rollmed_30d"] = float(pd.Series(replies_vals).median())
                        extra_row["replies_ema_14d"] = float(pd.Series(replies_vals).mean())
                    if len(views_vals) > 0:
                        extra_row["views_rollmed_30d"] = float(pd.Series(views_vals).median())
                        extra_row["views_ema_14d"] = float(pd.Series(views_vals).mean())
                    try:
                        last_rec = recent_posts[-1]
                        extra_row["likes"] = int(last_rec.get("likes", 0) or 0)
                        extra_row["replies"] = int(last_rec.get("replies", 0) or 0)
                        extra_row["views"] = int(last_rec.get("views", 0) or 0)
                    except Exception:
                        pass
            except Exception:
                # non-fatal: fall back to default zeros
                pass
    except Exception:
        pass

    if os.path.exists(cache_path) and os.path.exists(feat_list_path):
        try:
            # Load precomputed cache and feature list
            df_cache = pd.read_parquet(cache_path)
            with open(feat_list_path, "r", encoding="utf8") as f:
                feat_cols_base = json.load(f)

            # Append the new tweet row to cached dataframe so we can compute
            # time-dependent features (rolling medians, EMAs, jump intensity)
            er = extra_row.copy()
            # Ensure all columns exist in er
            for c in df_cache.columns:
                if c not in er:
                    er[c] = 0 if df_cache[c].dtype.kind in "if" else ""
            er["dataset_name"] = "manual"
            # Compute lightweight text-derived features for the appended row so the
            # model sees differences between texts (length, word count, simple flags).
            try:
                txt = str(er.get("text", "") or "")
                er["text_length"] = len(txt)
                er["word_count"] = len(txt.split())
                er["has_hashtag"] = int("#" in txt)
                er["has_mention"] = int("@" in txt)
                er["has_excl"] = int("!" in txt)
            except Exception:
                pass
            df = pd.concat([df_cache, pd.DataFrame([er])], ignore_index=True)

            # Coerce created_at and sort (only needed for rolling computations)
            df["created_at"] = pd.to_datetime(df["created_at"].astype(str), errors="coerce")
            # If parsing yields some NaT values, forward/backfill from neighbors
            if df["created_at"].notna().any():
                df = df.sort_values("created_at").reset_index(drop=True)
            else:
                df = df.reset_index(drop=True)

            # Recompute rolling-based features for targets (only these depend on history)
            for tgt in ["likes", "replies", "views"]:
                if tgt in df.columns:
                    idx = pd.to_datetime(df["created_at"].astype(str), errors="coerce")
                    if idx.isna().any():
                        if idx.notna().any():
                            idx = idx.fillna(method="ffill").fillna(method="bfill")
                        else:
                            idx = pd.date_range(end=pd.Timestamp.now(), periods=len(df), freq="T")
                    s = pd.Series(df[tgt].values, index=idx)
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

            # Recompute jump_intensity and is_jump (depends on rollmed cols)
            if all(c in df.columns for c in ["likes", "replies", "views"]):
                idx = pd.to_datetime(df["created_at"].astype(str), errors="coerce")
                if idx.isna().any():
                    if idx.notna().any():
                        idx = idx.fillna(method="ffill").fillna(method="bfill")
                    else:
                        idx = pd.date_range(end=pd.Timestamp.now(), periods=len(df), freq="T")
                s_replies = pd.Series(df["replies"].values, index=idx)
                roll_std = s_replies.rolling("30D", min_periods=5).std().shift(1).reset_index(drop=True)
                df["replies_zscore_30d"] = ((df["replies"] - df["replies_rollmed_30d"]) / (roll_std + 1e-3)).fillna(0)

                df["jump_intensity"] = np.maximum.reduce([
                    df["likes"] / (1e-3 + df["likes_rollmed_30d"]),
                    df["replies"] / (1e-3 + df["replies_rollmed_30d"]),
                    df["views"] / (1e-3 + df["views_rollmed_30d"])
                ])
                df["jump_intensity"] = np.log1p(df["jump_intensity"])
                # standardize using historical mean/std (including the appended row is fine)
                df["jump_intensity"] = (df["jump_intensity"] - df["jump_intensity"].mean()) / (df["jump_intensity"].std() + 1e-6)
                df["is_jump"] = (
                    (df["likes"] > df["likes_rollmed_30d"] * 2.5) |
                    (df["replies"] > df["replies_rollmed_30d"] * 2.5) |
                    (df["views"] > df["views_rollmed_30d"] * 2.0)
                ).astype(int)

            # Now we have features for historical rows and the appended tweet; proceed
        except Exception:
            # If anything goes wrong reading the cached features, do NOT trigger
            # the training pipeline here. The user requested inference-only
            # behaviour for prediction requests. Instead fall back to a
            # minimal single-row dataframe using the provided tweet and
            # lightweight text-derived features (zeros for all other fields).
            import traceback as _tb
            print("Warning: feature cache read failed; proceeding with minimal fallback (no training)")
            _tb.print_exc()
            er = extra_row.copy()
            # minimal text-derived features so predictions are not blind to text
            try:
                txt = str(er.get("text", "") or "")
                er["text_length"] = len(txt)
                er["word_count"] = len(txt.split())
                er["has_hashtag"] = int("#" in txt)
                er["has_mention"] = int("@" in txt)
                er["has_excl"] = int("!" in txt)
            except Exception:
                pass
            er["dataset_name"] = "manual"
            df = pd.DataFrame([er])
    else:
        # No cache available: do NOT run training here. Build a minimal
        # single-row dataframe populated from the request and lightweight
        # text-derived features. Missing numeric features will be filled with
        # zeros later before inference.
        print("No feature cache found; proceeding with minimal fallback (no training)")
        er = extra_row.copy()
        try:
            txt = str(er.get("text", "") or "")
            er["text_length"] = len(txt)
            er["word_count"] = len(txt.split())
            er["has_hashtag"] = int("#" in txt)
            er["has_mention"] = int("@" in txt)
            er["has_excl"] = int("!" in txt)
        except Exception:
            pass
        er["dataset_name"] = "manual"
        df = pd.DataFrame([er])

    # Prefer saved feature metadata if available (feature ordering + per-target lists)
    meta_path = os.path.join(MODEL_DIR, "feature_meta.json")
    feature_meta = None
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as fh:
                feature_meta = json.load(fh)
        except Exception:
            feature_meta = None

    exclude_cols = ["created_at", "text", "dataset_name", "hashtags", "date", "best_trend", "username", "is_jump", "jump_intensity"]

    if feature_meta and "targets" in feature_meta:
        # Build a full candidate feature list preserving per-target order when possible
        full_feat_cols = []
        for t in ["likes", "replies", "views"]:
            cols = feature_meta["targets"].get(t, {}).get("feat_cols", [])
            for c in cols:
                if c not in full_feat_cols:
                    full_feat_cols.append(c)
        # Fallback: if nothing from meta, derive from df as before
        if len(full_feat_cols) == 0:
            full_feat_cols = [c for c in df.columns if c not in TARGETS + exclude_cols and df[c].dtype.kind in "ifb" and 'zscore' not in c]
    else:
        full_feat_cols = [c for c in df.columns if c not in TARGETS + exclude_cols and df[c].dtype.kind in "ifb" and 'zscore' not in c]

    # Ensure required columns exist
    for c in full_feat_cols + ["jump_intensity"]:
        if c not in df.columns:
            df[c] = 0

    # create full sequences using superset of features; we'll slice per-target later
    X_full_with_jump = df[full_feat_cols + ["jump_intensity"]].fillna(0).values

    def crear_secuencias_full(data, seq_len=SEQ_LEN):
        X_list = []
        for i in range(len(data) - seq_len + 1):
            X_list.append(data[i:i+seq_len])
        return np.array(X_list)

    X_seq_full = crear_secuencias_full(X_full_with_jump, SEQ_LEN)
    # last sequence corresponds to appended tweet
    # We'll slice per-target when loading each model

    # Need device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    full_last_row = X_full_with_jump[-1]
    # Build a minimal feature map for the appended tweet and delegate the
    # forward pass to the lightweight simulator which handles loading models
    # and boosters for us. This avoids duplicating loading logic here.
    try:
        from predictor.twitter.simulate_forward import run_forward
        # Build a feature map from the last dataframe row; include numeric features
        feat_map = {}
        last_row_series = df.iloc[-1]
        for c in full_feat_cols:
            try:
                val = last_row_series.get(c, 0)
                feat_map[c] = float(val) if not pd.isna(val) else 0.0
            except Exception:
                feat_map[c] = 0.0
        # ensure jump and is_jump included
        feat_map["jump_intensity"] = float(last_row_series.get("jump_intensity", 0) or 0)
        feat_map["is_jump"] = int(last_row_series.get("is_jump", 0) or 0)

        out = run_forward(MODEL_DIR, feat_map, seq_len=SEQ_LEN, device=DEVICE, force_is_jump=force_is_jump)
        # run_forward now returns a dict {"predictions": ..., "is_jump": ...}
        if isinstance(out, dict) and out.get('predictions') is not None:
            results = out.get('predictions')
            # prefer classifier decision returned by run_forward, fallback to feat_map
            is_jump = int(out.get('is_jump', int(feat_map.get('is_jump', 0))))
        else:
            # backward compatibility: run_forward returned raw predictions
            results = out
            is_jump = int(feat_map.get("is_jump", 0))
    except Exception as _e:
        # If the simulator import or run fails, fall back to empty results and log
        import traceback
        traceback.print_exc()
        results = {t: None for t in ["likes", "replies", "views"]}
        is_jump = int(df.iloc[-1].get("is_jump", 0))

    if feature_meta and feature_meta.get('clf_features'):
        features_used = len(feature_meta.get('clf_features')) + 1
    else:
        features_used = len(full_feat_cols) + 1
    meta = {"is_jump": is_jump, "features_used": features_used}
    # Emit a compact debug line so the running backend logs show the raw
    # quantiles produced for this request and a short text snippet. This
    # helps verify the server process is actually using the model-backed
    # predictor rather than a stale heuristic.
    try:
        dbg = {
            "source": "predict_v3_wrapper",
            "text_snippet": (str(text)[:120] if 'text' in locals() else ""),
            "predictions": results,
            "meta": meta,
        }
        # Keep ensure_ascii=False to preserve any accented chars in logs
        print("[PREDICT_WRAPPER]", json.dumps(dbg, ensure_ascii=False))
    except Exception:
        # Don't break prediction if logging fails
        pass

    return {"predictions": results, "meta": meta}


if __name__ == "__main__":
    # quick demo - allow demo to explicitly train if artifacts are missing
    ensure_trained(train_if_missing=True)
    out = predict_single("Lanzamos hoy una nueva funcionalidad que mejora la experiencia. ¡Probá y contanos! #lanzamiento", "2025-11-10T14:30:00")
    print(out)
