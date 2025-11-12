from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from datetime import datetime
import os
import pickle
import random
import math
import json

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# Make repository root importable so we can import the predictor wrapper module
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
import sys
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Try to import the predictor wrapper to provide model-backed predictions.
# Capture and print full tracebacks on import failure so the running server
# logs show the real error (e.g., missing torch or other dependency).
import traceback

PREDICTOR_WRAPPER = None
PREDICTOR_IMPL = None

def _try_import_predictor():
    global PREDICTOR_WRAPPER, PREDICTOR_IMPL
    base = 'predictor.twitter'

    # Try Emma wrapper first (FFN multitarea - más rápido y mejor F1)
    try:
        from predictor.twitter import predict_emma_wrapper as predictor_wrapper
        PREDICTOR_WRAPPER = predictor_wrapper
        PREDICTOR_IMPL = 'emma'
        print('[startup] ✅ Loaded Emma FFN predictor wrapper')
        return
    except Exception as e:
        print('[startup] Failed to import Emma predictor wrapper:')
        traceback.print_exc()

    # Try heavy wrapper second (v3 LSTM quantile)
    try:
        from predictor.twitter import predict_v3_wrapper as predictor_wrapper
        PREDICTOR_WRAPPER = predictor_wrapper
        PREDICTOR_IMPL = 'heavy'
        print('[startup] ✅ Loaded v3 LSTM predictor wrapper')
        return
    except Exception as e:
        print('[startup] Failed to import heavy predictor wrapper:')
        traceback.print_exc()

    # Try quick predictor next
    try:
        from predictor.twitter import quick_predictor as predictor_wrapper
        PREDICTOR_WRAPPER = predictor_wrapper
        PREDICTOR_IMPL = 'quick'
        print('[startup] ✅ Loaded quick predictor wrapper')
        return
    except Exception as e:
        print('[startup] Failed to import quick predictor wrapper:')
        traceback.print_exc()

    # If we get here, no predictor was imported
    PREDICTOR_WRAPPER = None
    PREDICTOR_IMPL = None


# Execute import attempts now so startup logs capture any errors
_try_import_predictor()

# Log which predictor wrapper was imported at module load time for debugging
try:
    if PREDICTOR_WRAPPER is None:
        print("[startup] PREDICTOR_WRAPPER=None (no model-backed predictor loaded)")
    else:
        # show module/name to help debug which implementation is active
        name = getattr(PREDICTOR_WRAPPER, '__name__', None) or type(PREDICTOR_WRAPPER).__name__
        print(f"[startup] PREDICTOR_WRAPPER={name}")
except Exception:
    pass

class TweetRequest(BaseModel):
    text: str = Field(..., example="Este es un tweet de prueba #ejemplo @usuario")
    created_at: Optional[str] = Field(None, example="2025-11-10T12:34:00")
    platform: Optional[str] = Field("twitter", example="twitter")


class Metrics(BaseModel):
    likes: int
    replies: int
    comments: int
    views: int
    retweets: int
    confidence: Dict[str, Any]


app = FastAPI(title="ATID Conexión API", version="0.1")

# CORS: allow frontend running on different origin during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Try to load pre-trained models (if available)
MODELS = {}
_pkl_path = os.path.abspath(os.path.join(BASE_DIR, "..", "predictor", "twitter", "jump_models.pkl"))
# Historically models were sometimes stored inside the predictor/twitter/models_uala_v3_jumpboosted
# directory. Try the top-level path first, then fall back to the MODEL_DIR location used by the
# training pipeline so `models_loaded` correctly reflects what's actually available.
if os.path.exists(_pkl_path):
    try:
        with open(_pkl_path, "rb") as f:
            MODELS = pickle.load(f)
    except Exception:
        MODELS = {}
else:
    # fallback location
    alt_pkl = os.path.abspath(os.path.join(BASE_DIR, "..", "predictor", "twitter", "models_uala_v3_jumpboosted", "jump_models.pkl"))
    if os.path.exists(alt_pkl):
        try:
            with open(alt_pkl, "rb") as f:
                MODELS = pickle.load(f)
        except Exception:
            MODELS = {}
    # Try to load pre-trained models (if available) — still keep the legacy pkl check so other code can inspect it
    MODELS = {}
    _pkl_path = os.path.abspath(os.path.join(BASE_DIR, "..", "predictor", "twitter", "jump_models.pkl"))
    alt_pkl = os.path.abspath(os.path.join(BASE_DIR, "..", "predictor", "twitter", "models_uala_v3_jumpboosted", "jump_models.pkl"))
    for p in [_pkl_path, alt_pkl]:
        if os.path.exists(p):
            try:
                with open(p, "rb") as f:
                    MODELS = pickle.load(f)
                break
            except Exception:
                MODELS = {}


def heuristic_predict(text: str, created_at: Optional[str]) -> Metrics:
    """Fallback lightweight predictor when no trained models are available.

    Uses simple rules based on text length, hashtags and mentions and day of week.
    Returns integer metrics and a small confidence estimate.
    """
    txt = str(text or "")
    text_len = len(txt)
    word_count = len(txt.split())
    hashtags = txt.count("#")
    mentions = txt.count("@")
    exclam = txt.count("!")

    # date features
    is_weekend = False
    if created_at:
        try:
            dt = datetime.fromisoformat(created_at)
            is_weekend = dt.weekday() >= 5
        except Exception:
            is_weekend = False

    # base signals
    base = 5 + text_len * 0.8 + word_count * 1.2
    base += hashtags * 35 + mentions * 18 + exclam * 6
    if is_weekend:
        base *= 0.9

    # non-linear uplift for short catchy texts
    if 3 <= word_count <= 12:
        base *= 1.12

    # random noise (stable per-request)
    noise = random.gauss(0, max(1.0, base * 0.08))

    likes = max(0, int(base + noise))
    replies = max(0, int(likes * 0.08 + mentions * 0.5 + random.randint(0, 3)))
    retweets = max(0, int(likes * 0.28 + hashtags * 2 + random.randint(0, 5)))
    comments = replies  # for Twitter replies ~ comments
    views = max(0, int(max(100, likes * 12 + retweets * 8 + random.randint(0, 50))))

    # simple confidence bands
    q_low = max(0, int(likes * 0.6))
    q_med = likes
    q_high = int(likes * 1.6 + 10)

    return Metrics(
        likes=likes,
        replies=replies,
        comments=comments,
        views=views,
        retweets=retweets,
        confidence={"likes": {"q01": q_low, "q50": q_med, "q99": q_high}}
    )


@app.post("/predict_tweet")
def predict_tweet(payload: TweetRequest):
    """Predict metrics for a tweet (likes, replies, comments, views, retweets).

    This endpoint requires a model-backed predictor. Priority order:
      1. Emma FFN (predict_emma_wrapper) - Fast multitask model with peak detection
      2. V3 LSTM (predict_v3_wrapper) - Quantile regression with boosting
      3. Quick predictor - Lightweight fallback
    
    If no predictor is available the endpoint returns HTTP 503.
    """
    if payload.platform.lower() != "twitter":
        raise HTTPException(status_code=400, detail="This endpoint currently supports only platform='twitter'.")

    # Require a model-backed predictor (no heuristic fallback). Prefer heavy; quick_predictor acceptable.
    if PREDICTOR_WRAPPER is None:
        # No model available — return 503 so frontend knows the model is not ready.
        raise HTTPException(status_code=503, detail="Model artifacts not available. Prediction service is temporarily unavailable.")

    # Call the predictor wrapper; if it raises an unexpected error return 500.
    try:
        # ensure models are trained/available (this is a no-op if artifacts already present)
        try:
            # Ensure models exist but do NOT trigger training as a side-effect of
            # a prediction request. If the caller really wants training to run
            # they must call ensure_trained(train_if_missing=True) explicitly.
            if hasattr(PREDICTOR_WRAPPER, "ensure_trained"):
                try:
                    PREDICTOR_WRAPPER.ensure_trained(train_if_missing=False)
                except TypeError:
                    # Backwards compatibility: older ensure_trained implementations
                    # without the flag will still be called (rare).
                    PREDICTOR_WRAPPER.ensure_trained()
        except Exception:
            # If ensure_trained fails, we still try to predict if artifacts exist; otherwise we escalate below.
            pass

        # Run prediction and log which implementation handled the request.
        pred = PREDICTOR_WRAPPER.predict_single(payload.text, payload.created_at or datetime.now().isoformat())
        try:
            # Lightweight request-level logging to the server stdout so the
            # operator can confirm the running process used the model-backed
            # predictor and inspect the raw quantiles returned.
            impl = getattr(PREDICTOR_WRAPPER, '__name__', None) or getattr(PREDICTOR_WRAPPER, '__module__', None) or 'predictor'
            log_obj = {
                'predictor_impl': impl,
                'text_snippet': (payload.text[:120] if payload.text else ''),
                'predictions': pred.get('predictions') if isinstance(pred, dict) else str(pred)
            }
            print('[PREDICT_API]', json.dumps(log_obj, ensure_ascii=False))
        except Exception:
            pass
        preds = pred.get("predictions", {})
        likes = preds.get("likes") or {"q50": 0}
        replies = preds.get("replies") or {"q50": 0}
        views = preds.get("views") or {"q50": 0}
        
        # Extract jump probability and classification from wrapper response
        is_jump_prob = pred.get("is_jump_prob", {})
        is_jump = pred.get("is_jump", {})

        return JSONResponse(content={
            "metrics": {
                "likes": likes.get("q50", 0),
                "replies": replies.get("q50", 0),
                "comments": replies.get("q50", 0),
                "views": views.get("q50", 0),
                "retweets": 0,
                "confidence": {
                    "likes": likes,
                    "replies": replies,
                    # include 'comments' in confidence to match frontend expectations
                    "comments": replies,
                    "views": views,
                }
            },
            "is_jump_prob": is_jump_prob,
            "is_jump": is_jump
        })
    except HTTPException:
        raise
    except Exception:
        import traceback as _tb
        _tb.print_exc()
        raise HTTPException(status_code=500, detail="Internal error while running model predictor")


@app.get("/health")
def health():
    # Return richer health info including whether a predictor implementation
    # was imported and whether model artifact files exist.
    def _artifact_paths():
        base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "predictor", "twitter"))
        emma_dir = os.path.join(base, "models_uala_multitask_ffn_v4_5_2stage_calib")
        v3_dir = os.path.join(base, "models_uala_v3_jumpboosted")
        candidates = [
            # Emma FFN artifacts
            os.path.join(emma_dir, "likes_best.pt"),
            os.path.join(emma_dir, "likes_scaler.pkl"),
            os.path.join(emma_dir, "likes_calibration.json"),
            # V3 LSTM artifacts
            os.path.join(base, "jump_models.pkl"),
            os.path.join(v3_dir, "jump_models.pkl"),
            os.path.join(v3_dir, "uala_likes_lstm.pt"),
            os.path.join(v3_dir, "feature_meta.json"),
            os.path.join(v3_dir, "features_cache.parquet"),
        ]
        return candidates

    paths = _artifact_paths()
    found = {p: os.path.exists(p) for p in paths}
    return {
        "status": "ok",
        "predictor_impl": PREDICTOR_IMPL,
        "predictor_wrapper_loaded": bool(PREDICTOR_WRAPPER),
        "artifact_files": found,
    }


@app.get("/predict_tweet_demo")
def predict_tweet_demo():
    """Demo endpoint returning a deterministic example JSON for frontend testing."""
    sample = {
        "likes": 164,
        "replies": 15,
        "comments": 15,
        "views": 2390,
        "retweets": 12,
        "confidence": {
            "likes": {"q01": 98, "q50": 164, "q99": 272},
            "comments": {"q01": 11, "q50": 15, "q99": 20},
            "views": {"q01": 1673, "q50": 2390, "q99": 3107}
        },
        "note": "demo-response"
    }
    return JSONResponse(content=sample)
