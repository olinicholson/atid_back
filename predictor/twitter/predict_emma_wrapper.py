"""Wrapper para predicción usando modelos Emma (FFN Multitarea).

Usage:
  - Call `predict_single(text, created_at_iso)` to get predictions for a single tweet

Este wrapper carga los modelos Emma entrenados (FFN multitarea) y genera predicciones
usando solo features flat (sin secuencias LSTM).
"""
import os
import pickle
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import sys

# Ensure imports work
PREDICTOR_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PREDICTOR_ROOT not in sys.path:
    sys.path.insert(0, PREDICTOR_ROOT)
THIS_DIR = os.path.abspath(os.path.dirname(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "core", "data")
MODEL_DIR = os.path.join(THIS_DIR, "models_uala_multitask_ffn_v4_5_2stage_calib")
TARGETS = ["likes", "replies", "views"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ===== ARQUITECTURA (copiar de predictor_emma.py) =====
class MultiTaskFFNShared(nn.Module):
    def __init__(self, input_dim, hidden=(256,128), shared_dim=128, dropout=0.30):
        super().__init__()
        layers, prev = [], input_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(inplace=False), nn.Dropout(dropout)]
            prev = h
        self.backbone = nn.Sequential(*layers)
        self.shared = nn.Linear(prev, shared_dim)

        self.clf_head = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Dropout(dropout),
            nn.Linear(shared_dim, 64),
            nn.ReLU(inplace=False),
            nn.Dropout(dropout/2),
            nn.Linear(64, 1)
        )
        self.reg_head = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Dropout(dropout/2),
            nn.Linear(shared_dim, 1)
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x):
        h = self.backbone(x)
        s = self.shared(h)
        logit = self.clf_head(s)
        yhat = self.reg_head(s)
        return logit, yhat


def compute_features_for_tweet(text: str, created_at_iso: str, recent_posts: list = None):
    """Computa features para un tweet usando pipeline simple de Emma.
    
    Args:
        text: Texto del tweet
        created_at_iso: Timestamp ISO del tweet
        recent_posts: Lista opcional de posts recientes para calcular rolling features
        
    Returns:
        dict con features calculadas
    """
    created_at = pd.to_datetime(created_at_iso)
    
    # Features básicas temporales
    features = {
        "hour": created_at.hour,
        "weekday": created_at.weekday(),
        "is_weekend": int(created_at.weekday() >= 5),
        "hour_sin": np.sin(2 * np.pi * created_at.hour / 24),
        "hour_cos": np.cos(2 * np.pi * created_at.hour / 24),
    }
    
    # Features de texto
    features["text_length"] = len(text)
    features["has_hashtag"] = int("#" in text)
    features["has_mention"] = int("@" in text)
    
    # Features sociales (placeholder - requeriría cargar top10.csv)
    # Por ahora usar valores promedio o cero
    features["followers_rel"] = 0.0
    features["engagement_rel"] = 0.0
    
    # Jump intensity (placeholder - requeriría histórico)
    features["jump_intensity"] = 0.0
    
    # Si hay recent_posts, calcular features de histórico
    if recent_posts and len(recent_posts) > 0:
        try:
            likes_vals = [int(p.get("likes", 0) or 0) for p in recent_posts]
            replies_vals = [int(p.get("replies", 0) or 0) for p in recent_posts]
            views_vals = [int(p.get("views", 0) or 0) for p in recent_posts]
            
            if len(likes_vals) > 0:
                features["likes_rollmed"] = float(np.median(likes_vals))
                features["replies_rollmed"] = float(np.median(replies_vals))
                features["views_rollmed"] = float(np.median(views_vals))
                
                # Jump intensity relativo
                mean_likes = max(1, np.mean(likes_vals))
                features["jump_intensity"] = np.log1p(features.get("likes_rollmed", mean_likes) / mean_likes)
        except Exception:
            pass
    
    return features


def predict_single(text: str, created_at_iso: str, recent_posts: list = None):
    """Predice likes/replies/views para un tweet usando modelos Emma.
    
    Args:
        text: Texto del tweet
        created_at_iso: Timestamp ISO del tweet
        recent_posts: Lista opcional de posts recientes [{"likes": int, "replies": int, "views": int}, ...]
        
    Returns:
        dict con predicciones para cada target: {"likes": float, "replies": float, "views": float, 
                                                  "is_jump_prob": {"likes": float, ...}}
    """
    # Verificar que existan los modelos
    required_files = [
        os.path.join(MODEL_DIR, f"{t}_best.pt") for t in TARGETS
    ] + [
        os.path.join(MODEL_DIR, f"{t}_scaler.pkl") for t in TARGETS
    ] + [
        os.path.join(MODEL_DIR, f"{t}_calibration.json") for t in TARGETS
    ]
    
    missing = [f for f in required_files if not os.path.exists(f)]
    if missing:
        raise RuntimeError(f"Modelos Emma no encontrados. Faltan: {[os.path.basename(f) for f in missing]}. Entrená los modelos primero con predictor_emma.py")
    
    # Computar features
    features = compute_features_for_tweet(text, created_at_iso, recent_posts)
    
    # Features en orden (debe coincidir con el orden de entrenamiento)
    # 8 features base + 1 followers_rel = 9 que espera el scaler
    feat_order = ["hour", "weekday", "is_weekend", "hour_sin", "hour_cos",
                  "text_length", "has_hashtag", "has_mention", "followers_rel"]
    
    # Construir vector de features
    X_vec = np.array([[features.get(f, 0.0) for f in feat_order]], dtype=np.float32)
    
    results = {}
    
    for target in TARGETS:
        # Cargar artefactos
        model_path = os.path.join(MODEL_DIR, f"{target}_best.pt")
        scaler_path = os.path.join(MODEL_DIR, f"{target}_scaler.pkl")
        calib_path = os.path.join(MODEL_DIR, f"{target}_calibration.json")
        
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        
        with open(calib_path, "r") as f:
            calib = json.load(f)
            q90 = calib["q90"]
            thresh_opt = calib.get("threshold_opt", 0.5)
        
        # Escalar features (ya incluye las 9 features que espera el modelo)
        X_scaled = scaler.transform(X_vec)
        
        # NO añadir jump_intensity aquí - ya está incluido en las 9 features
        # El modelo espera exactamente 9 features de entrada
        
        # Cargar modelo
        model = MultiTaskFFNShared(input_dim=X_scaled.shape[1], hidden=(256,128), shared_dim=128, dropout=0.30).to(DEVICE)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval()
        
        # Predicción
        X_t = torch.tensor(X_scaled).float().to(DEVICE)
        with torch.no_grad():
            logit, y_pred_norm = model(X_t)
            prob_jump = torch.sigmoid(logit).cpu().numpy().item()
            y_pred_norm_val = y_pred_norm.cpu().numpy().item()
        
        # Des-normalizar
        y_pred = np.expm1(y_pred_norm_val) * q90
        
        # Calcular IC 95% aproximado usando residual_std (si está disponible en calibración)
        # Por ahora usar un factor conservador del 30% como spread
        spread_factor = 0.3
        q01_approx = max(0, y_pred * (1 - spread_factor))
        q99_approx = y_pred * (1 + spread_factor)
        
        results[target] = {
            "prediction": float(y_pred),
            "q01": float(q01_approx),
            "q50": float(y_pred),
            "q99": float(q99_approx),
            "is_jump_prob": float(prob_jump),
            "is_jump": int(prob_jump >= thresh_opt)
        }
    
    # Formato compatible con API existente (v3 wrapper)
    return {
        "predictions": {
            "likes": {
                "q01": results["likes"]["q01"],
                "q50": results["likes"]["q50"],
                "q99": results["likes"]["q99"]
            },
            "replies": {
                "q01": results["replies"]["q01"],
                "q50": results["replies"]["q50"],
                "q99": results["replies"]["q99"]
            },
            "views": {
                "q01": results["views"]["q01"],
                "q50": results["views"]["q50"],
                "q99": results["views"]["q99"]
            }
        },
        "is_jump_prob": {t: results[t]["is_jump_prob"] for t in TARGETS},
        "is_jump": {t: results[t]["is_jump"] for t in TARGETS},
        "metadata": {
            "model_type": "emma_ffn_multitask",
            "features_used": feat_order,
            "device": DEVICE
        }
    }


def predict_batch(tweets: list):
    """Predice para múltiples tweets.
    
    Args:
        tweets: Lista de dicts con {"text": str, "created_at": str, "recent_posts": list (opcional)}
        
    Returns:
        list de resultados (mismo formato que predict_single)
    """
    results = []
    for tweet in tweets:
        try:
            pred = predict_single(
                text=tweet["text"],
                created_at_iso=tweet["created_at"],
                recent_posts=tweet.get("recent_posts")
            )
            results.append(pred)
        except Exception as e:
            results.append({"error": str(e)})
    return results


if __name__ == "__main__":
    # Ejemplo de uso
    example_text = "¡Nueva función en Ualá! 🚀 Ahora podés invertir desde la app #Fintech"
    example_date = "2024-11-12T14:30:00"
    
    try:
        prediction = predict_single(example_text, example_date)
        print("🎯 Predicción Emma FFN:")
        preds = prediction.get("predictions", {})
        print(f"   Likes: {preds['likes']['q50']:.0f} (IC: {preds['likes']['q01']:.0f} - {preds['likes']['q99']:.0f})")
        print(f"   Replies: {preds['replies']['q50']:.0f} (IC: {preds['replies']['q01']:.0f} - {preds['replies']['q99']:.0f})")
        print(f"   Views: {preds['views']['q50']:.0f} (IC: {preds['views']['q01']:.0f} - {preds['views']['q99']:.0f})")
        print(f"   Probabilidad de pico (likes): {prediction['is_jump_prob']['likes']*100:.1f}%")
        print(f"   Es pico?: {'Sí' if prediction['is_jump']['likes'] else 'No'}")
    except Exception as e:
        print(f"❌ Error: {e}")

