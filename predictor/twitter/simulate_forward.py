"""Simulator to run a single-forward pass of the trained QuantileLSTM models
using a precomputed feature vector for one tweet.

Usage:
  python simulate_forward.py --model-dir PATH/TO/models_uala_v3_jumpboosted --features tweet_features.json

The features file should be a JSON mapping feature_name -> numeric value for the
set of feature names used by the model (see feature_names.json or feature_meta.json
inside the model dir). The script will repeat the provided row to build a full
sequence of length SEQ_LEN and run the per-target LSTM checkpoints and boosters
if available (jump_models.pkl).
"""
import argparse
import json
import os
import pickle
import sys
import numpy as np
import torch

# Ensure repo root importability similar to the wrapper
THIS_DIR = os.path.abspath(os.path.dirname(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from predictor.twitter.predictor_v3_jumpboosted import QuantileLSTM, SEQ_LEN, TARGETS


def load_feature_order(model_dir):
    # Prefer feature_meta.json targets order, fall back to feature_names.json
    meta_path = os.path.join(model_dir, "feature_meta.json")
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf8") as fh:
                meta = json.load(fh)
            full = []
            for t in TARGETS:
                cols = meta.get("targets", {}).get(t, {}).get("feat_cols", [])
                for c in cols:
                    if c not in full:
                        full.append(c)
            return full
        except Exception:
            pass
    fn = os.path.join(model_dir, "feature_names.json")
    if os.path.exists(fn):
        try:
            return json.load(open(fn, "r", encoding="utf8"))
        except Exception:
            pass
    raise RuntimeError("Unable to determine model feature order. Ensure feature_meta.json or feature_names.json exists in model_dir")


def build_last_row(full_feat_cols, feat_map):
    arr = []
    for c in full_feat_cols:
        v = feat_map.get(c, 0)
        try:
            v = float(v)
        except Exception:
            v = 0.0
        arr.append(v)
    # Append jump_intensity placeholder if not present
    return np.array(arr, dtype=float)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True, help="Path to models_uala_v3_jumpboosted directory")
    p.add_argument("--features", required=False, help="JSON file with feature_name -> value for the tweet row")
    p.add_argument("--text", required=False, help="Tweet text (alternative to --features)")
    p.add_argument("--created-at", dest="created_at", required=False, help="ISO timestamp for the tweet (used to build time features)")
    p.add_argument("--seq-len", type=int, default=None, help="Optional sequence length override (defaults to training SEQ_LEN)")
    args = p.parse_args()

    model_dir = os.path.abspath(args.model_dir)
    if not os.path.isdir(model_dir):
        raise SystemExit(f"model_dir not found: {model_dir}")

    feat_map = {}
    if args.features:
        feat_map = json.load(open(args.features, "r", encoding="utf8"))
    else:
        # If user provided text + created_at, compute a minimal feature map similar
        # to the lightweight features used in the wrapper so no precompute file is needed.
        if not args.text:
            raise SystemExit("Either --features or --text must be provided")
        txt = str(args.text or "")
        created = args.created_at
    # basic text-derived features
    feat_map["text_length"] = len(txt)
    feat_map["word_count"] = len(txt.split())
    # Delegate to run_forward and print (use args.seq_len if provided)
    out = run_forward(model_dir, feat_map, seq_len=args.seq_len)
    print(json.dumps({"predictions": out}, ensure_ascii=False, indent=2))


def run_forward(model_dir, feat_map, seq_len=None, device=None, force_is_jump=None):
    """Run a forward pass given a feature map (feature_name -> value).

    Returns a dict mapping target -> {q01,q50,q99} (or None) and *_adjusted flags.
    """
    seq_len = seq_len or SEQ_LEN
    DEVICE = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # determine feature ordering from model_dir (global fallback)
    full_feat_cols = load_feature_order(model_dir)

    # Try to read per-target metadata (preferred) so each target receives the
    # exact feature ordering used at training time. feature_meta.json is
    # produced by the training pipeline and contains per-target feat_cols.
    feature_meta = None
    meta_path = os.path.join(model_dir, "feature_meta.json")
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf8") as fh:
                feature_meta = json.load(fh)
        except Exception:
            feature_meta = None

    # We'll build per-target last_row and X inside the loop below so each
    # model gets the features it expects (avoids size mismatch between saved
    # state_dict and provided input length).

    # Load boosters + classifier if available
    pkl_path = os.path.join(model_dir, "jump_models.pkl")
    boosters = {}
    clf = None
    scaler = None
    clf_features = None
    if os.path.exists(pkl_path):
        try:
            with open(pkl_path, "rb") as f:
                pk = pickle.load(f)
            boosters = pk.get("boosters", {})
            clf = pk.get("clf")
            scaler = pk.get("scaler")
        except Exception:
            boosters = {}

    # Determine classifier features (if feature_meta provided)
    clf_features = None
    if feature_meta:
        clf_features = feature_meta.get("clf_features")

    # Determine global is_jump using trained classifier if available,
    # otherwise fall back to provided feat_map['is_jump'] (or 0).
    # If `force_is_jump` is provided, use it to override classifier decision
    # (useful for debugging / forcing booster behaviour without retraining).
    global_is_jump = int(bool(feat_map.get('is_jump', 0)))
    if force_is_jump is not None:
        try:
            global_is_jump = int(bool(force_is_jump))
        except Exception:
            global_is_jump = int(bool(feat_map.get('is_jump', 0)))
    if force_is_jump is None and clf is not None and scaler is not None and clf_features:
        try:
            clf_row = np.array([feat_map.get(f, 0.0) for f in clf_features], dtype=float).reshape(1, -1)
            try:
                clf_row_scaled = scaler.transform(clf_row)
            except Exception:
                clf_row_scaled = clf_row
            pred_jump = clf.predict(clf_row_scaled)
            global_is_jump = int(pred_jump[0])
        except Exception:
            # keep fallback
            global_is_jump = int(bool(feat_map.get('is_jump', 0)))

    results = {}
    for target in ["likes", "replies", "views"]:
        lstm_file = os.path.join(model_dir, f"uala_{target}_lstm.pt")
        q01 = q50 = q99 = None
        adjusted = False
        # Build per-target feature vector using feature_meta if available,
        # otherwise fall back to the global feature list.
        if feature_meta and feature_meta.get("targets") and feature_meta.get("targets").get(target):
            per_target_cols = feature_meta.get("targets").get(target).get("feat_cols", full_feat_cols)
        else:
            per_target_cols = full_feat_cols

        last_row_base = build_last_row(per_target_cols, feat_map)
        jump = float(feat_map.get("jump_intensity", 0.0))
        last_row_target = np.concatenate([last_row_base, np.array([jump], dtype=float)], axis=0)
        X = np.tile(last_row_target, (seq_len, 1))[None, :, :]
        if os.path.exists(lstm_file):
            try:
                state = torch.load(lstm_file, map_location=DEVICE)
                # infer input dim
                ih = None
                for k in [k for k in state.keys() if 'lstm.weight_ih_l0' in k]:
                    ih = state[k]
                    break
                if ih is not None:
                    input_dim_chk = ih.shape[1]
                    hidden_chk = ih.shape[0] // 4
                else:
                    input_dim_chk = X.shape[2]
                    hidden_chk = 256

                model = QuantileLSTM(input_dim=input_dim_chk, hidden=hidden_chk, num_targets=1, dropout=0.3).to(DEVICE)
                try:
                    model.load_state_dict(state)
                except Exception:
                    model.load_state_dict(state, strict=False)
                model.eval()

                X_in = torch.FloatTensor(X).to(DEVICE)
                if X_in.shape[-1] != input_dim_chk:
                    padded = torch.zeros((X_in.shape[0], X_in.shape[1], input_dim_chk), dtype=X_in.dtype, device=X_in.device)
                    padded[..., :X_in.shape[-1]] = X_in
                    X_in = padded

                with torch.no_grad():
                    preds = model(X_in).cpu().numpy()[0, 0, :]
                q01, q50, q99 = float(preds[0]), float(preds[1]), float(preds[2])
            except Exception as e:
                print(f"Failed to load/run model for {target}: {e}")

        # boosters
        if boosters and boosters.get(target) and q50 is not None:
            try:
                bjump = boosters[target].get("jump")
                bnorm = boosters[target].get("normal")

                # Determine expected feature ordering for boosters. Preference order:
                # 1) per-booster feature list stored inside the pickle (boosters[target]['feat_cols'] or 'feature_names')
                # 2) feature_meta.json entry: targets[target]['booster_feat_cols']
                # 3) per-target cols (per_target_cols) + ['jump_intensity'] (fallback)
                booster_feat_cols = None
                # try inside pickle
                try:
                    bt = boosters[target]
                    if isinstance(bt, dict):
                        booster_feat_cols = bt.get('feat_cols') or bt.get('feature_names')
                except Exception:
                    booster_feat_cols = None

                # try feature_meta
                if booster_feat_cols is None and feature_meta and feature_meta.get('targets') and feature_meta['targets'].get(target):
                    booster_feat_cols = feature_meta['targets'][target].get('booster_feat_cols')

                # fallback to per-target cols + jump
                if booster_feat_cols is None:
                    booster_feat_cols = list(per_target_cols) + ['jump_intensity']

                # Build last feature vector for booster respecting the ordering
                last_feat_for_booster = np.array([float(feat_map.get(c, 0.0) or 0.0) for c in booster_feat_cols], dtype=float)

                # If booster objects expose n_features_ or similar, try to match that dimension
                expected_dim = None
                for obj in (bnorm, bjump):
                    if obj is None:
                        continue
                    expected_dim = getattr(obj, 'n_features_', expected_dim) or expected_dim

                if expected_dim is not None and expected_dim != last_feat_for_booster.shape[0]:
                    if expected_dim > last_feat_for_booster.shape[0]:
                        last_feat_for_booster = np.concatenate([last_feat_for_booster, np.zeros(expected_dim - last_feat_for_booster.shape[0])], axis=0)
                    else:
                        last_feat_for_booster = last_feat_for_booster[:expected_dim]

                # Use the precomputed global classifier decision
                is_jump = int(global_is_jump)
                if is_jump == 1 and bjump is not None:
                    resid = bjump.predict(last_feat_for_booster.reshape(1, -1))[0]
                    shift = 0.25 * resid
                    q01 += shift
                    q50 += shift
                    q99 += shift
                    adjusted = True
                elif is_jump == 0 and bnorm is not None:
                    resid = bnorm.predict(last_feat_for_booster.reshape(1, -1))[0]
                    shift = 0.20 * resid
                    q01 += shift
                    q50 += shift
                    q99 += shift
                    adjusted = True
            except Exception:
                # Best-effort fallback for older pickles or unexpected failures: try padding/truncating
                import re, warnings
                err = sys.exc_info()[1]
                msg = str(err)
                try:
                    # last resort: try to call boosters with the per-target vector (previous behaviour)
                    last_feat_for_booster = last_row_target
                    if bnorm is not None and last_feat_for_booster.shape[0] != getattr(bnorm, 'n_features_', last_feat_for_booster.shape[0]):
                        expected = getattr(bnorm, 'n_features_', last_feat_for_booster.shape[0])
                        if expected > last_feat_for_booster.shape[0]:
                            last_feat_for_booster = np.concatenate([last_feat_for_booster, np.zeros(expected - last_feat_for_booster.shape[0])], axis=0)
                    is_jump = int(global_is_jump)
                    if is_jump == 1 and bjump is not None:
                        resid = bjump.predict(last_feat_for_booster.reshape(1, -1))[0]
                        shift = 0.25 * resid
                        q01 += shift
                        q50 += shift
                        q99 += shift
                        adjusted = True
                    elif is_jump == 0 and bnorm is not None:
                        resid = bnorm.predict(last_feat_for_booster.reshape(1, -1))[0]
                        shift = 0.20 * resid
                        q01 += shift
                        q50 += shift
                        q99 += shift
                        adjusted = True
                except Exception:
                    warnings.warn(f"Booster prediction failed for {target}: {msg}")

        if q01 is not None:
            # Ensure quantiles remain monotonic after booster adjustments.
            try:
                q_vals = [float(q01), float(q50), float(q99)]
                q_vals_sorted = sorted(q_vals)
                q01_s, q50_s, q99_s = q_vals_sorted
                q01, q50, q99 = q01_s, q50_s, q99_s
            except Exception:
                pass
            results[target] = {"q01": max(0, int(round(q01))), "q50": max(0, int(round(q50))), "q99": max(0, int(round(q99)))}
        else:
            results[target] = None
        results[target + "_adjusted"] = adjusted

    # Return predictions and the classifier decision used
    return {"predictions": results, "is_jump": int(global_is_jump)}


if __name__ == "__main__":
    main()
