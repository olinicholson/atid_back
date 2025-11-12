import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import pickle
import json
from features_sociales import generar_features_sociales
from sklearn.preprocessing import StandardScaler

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (18, 12)

# ===== CONFIGURACIÓN =====
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "core", "data")
TARGETS = ["likes", "replies", "views"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

VAL_WINDOW_DAYS = 60
FUTURE_GAP_DAYS = 21

# ===== ARQUITECTURA (misma que en predictor_emma.py) =====
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

# ===== CARGAR Y PREPARAR DATOS =====
print("🔄 Cargando y generando features...")

files = glob.glob(os.path.join(DATA_DIR, "posts_*uala*.csv"))
if not files:
    raise FileNotFoundError("No se encontraron archivos posts_*uala*.csv en core/data/")
df = pd.read_csv(files[0])
df["created_at"] = pd.to_datetime(df["created_at"])
print(f"✅ Registros Ualá: {len(df)}")

# ===== Features simples (mismo pipeline que training) =====
df["hour"] = df["created_at"].dt.hour
df["weekday"] = df["created_at"].dt.weekday
df["is_weekend"] = (df["weekday"] >= 5).astype(int)
df["hour_sin"] = np.sin(2*np.pi*df["hour"]/24)
df["hour_cos"] = np.cos(2*np.pi*df["hour"]/24)
df["text_length"] = df["text"].astype(str).str.len()
df["has_hashtag"] = df["text"].astype(str).str.contains("#", na=False).astype(int)
df["has_mention"] = df["text"].astype(str).str.contains("@", na=False).astype(int)

# Sociales
top10 = pd.read_csv(os.path.join(DATA_DIR, "posts_top10.csv"))
top10["created_at"] = pd.to_datetime(top10["created_at"])
top10["date"] = top10["created_at"].dt.date
df["date"] = df["created_at"].dt.date
df = generar_features_sociales(df_user=df, df_top10=top10)

# Jump intensity
df["jump_intensity"] = np.maximum.reduce([
    df["likes"]/(1e-3+df["likes"].mean()),
    df["replies"]/(1e-3+df["replies"].mean()),
    df["views"]/(1e-3+df["views"].mean())
])
df["jump_intensity"] = np.log1p(df["jump_intensity"])
df["jump_intensity"] = (df["jump_intensity"] - df["jump_intensity"].mean()) / (df["jump_intensity"].std() + 1e-6)

# Etiqueta is_jump (misma lógica permisiva)
df["is_jump"] = ((df["likes"]   > df["likes"].mean()   * 1.8) |
                 (df["replies"] > df["replies"].mean() * 1.8) |
                 (df["views"]   > df["views"].mean()   * 1.5)).astype(int)

feat_cols = [
    "hour","weekday","is_weekend","hour_sin","hour_cos",
    "text_length","has_hashtag","has_mention",
    "followers_rel","engagement_rel"
]
feat_cols = [c for c in feat_cols if c in df.columns]
print(f"✅ Features: {len(feat_cols)}")

# Split temporal
cutoff_date = df["created_at"].max() - pd.Timedelta(days=VAL_WINDOW_DAYS + FUTURE_GAP_DAYS)
idx_val = df["created_at"] > cutoff_date

# ===== CARGAR MODELOS Y GENERAR PREDICCIONES =====
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models_uala_multitask_ffn_v4_5_2stage_calib")

fig, axes = plt.subplots(3, 1, figsize=(18, 12))

for idx, target in enumerate(TARGETS):
    print(f"\n🎯 Procesando: {target.upper()}")
    
    # Preparar matrices
    X = df[feat_cols + ["jump_intensity"]].fillna(0).values
    y_raw = df[target].values.astype(float)
    y_jump = df["is_jump"].values.astype(int)
    
    # Validación set
    X_val = X[idx_val]
    y_val = y_raw[idx_val]
    j_val = y_jump[idx_val]
    
    # Cargar artefactos
    model_path = os.path.join(MODEL_DIR, f"{target}_best.pt")
    scaler_path = os.path.join(MODEL_DIR, f"{target}_scaler.pkl")
    calib_path = os.path.join(MODEL_DIR, f"{target}_calibration.json")
    
    if not os.path.exists(model_path):
        print(f"⚠️ Modelo no encontrado: {model_path}")
        continue
    
    # Cargar scaler y q90
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    
    with open(calib_path, "r") as f:
        calib = json.load(f)
        q90 = calib["q90"]
        thresh_opt = calib.get("threshold_opt", 0.5)
    
    # Escalar X
    X_val_scaled = scaler.transform(X_val)
    
    # Cargar modelo
    model = MultiTaskFFNShared(input_dim=X_val_scaled.shape[1], hidden=(256,128), shared_dim=128, dropout=0.30).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    
    # Predicciones
    X_val_t = torch.tensor(X_val_scaled).float().to(DEVICE)
    with torch.no_grad():
        logits, y_pred_norm = model(X_val_t)
        probs = torch.sigmoid(logits).cpu().numpy().ravel()
        y_pred_norm = y_pred_norm.cpu().numpy().ravel()
    
    # Des-normalizar predicciones
    y_pred = np.expm1(y_pred_norm) * q90
    
    # ===== CALCULAR IC 95% =====
    # Estrategia: usar residuales para estimar desviación estándar
    residuals = y_val - y_pred
    residual_std = np.std(residuals)
    
    # IC 95% simétrico (z=1.96 para normal)
    z_95 = 1.96
    ic_lower = y_pred - z_95 * residual_std
    ic_upper = y_pred + z_95 * residual_std
    
    # Asegurar no negativos (likes/replies/views no pueden ser negativos)
    ic_lower = np.maximum(ic_lower, 0)
    
    # Coverage empírico
    coverage = np.mean((y_val >= ic_lower) & (y_val <= ic_upper))
    
    # Clasificación de picos
    is_jump_pred = (probs >= thresh_opt).astype(int)
    
    # Métricas
    mae = np.mean(np.abs(y_val - y_pred))
    
    from sklearn.metrics import f1_score, precision_score, recall_score
    f1 = f1_score(j_val, is_jump_pred, zero_division=0)
    prec = precision_score(j_val, is_jump_pred, zero_division=0)
    rec = recall_score(j_val, is_jump_pred, zero_division=0)
    
    # ===== GRÁFICO =====
    ax = axes[idx]
    x_range = np.arange(len(y_val))
    
    # Banda de confianza IC 95% (primero para que quede debajo)
    ax.fill_between(x_range, ic_lower, ic_upper, alpha=0.15, color='lightcoral', 
                    label=f'IC 95% (Cov: {coverage*100:.1f}%)')
    
    # Líneas principales solamente
    ax.plot(x_range, y_val, '-', linewidth=2.5, 
            label='Real', color='steelblue', alpha=0.9)
    ax.plot(x_range, y_pred, '--', linewidth=2, 
            label='Predicho (FFN Multitarea)', color='coral', alpha=0.9)
    
    # Contadores de picos para stats (sin graficar)
    jump_indices_real = np.where(j_val == 1)[0]
    tp_indices = np.where((is_jump_pred == 1) & (j_val == 1))[0]
    fp_indices = np.where((is_jump_pred == 1) & (j_val == 0))[0]
    
    # Título y etiquetas
    ax.set_title(f'{target.upper()} - FFN Multitarea Two-Stage\n' + 
                 f'MAE: {mae:.2f} | F1: {f1:.3f} (P={prec:.2f}, R={rec:.2f}) | Coverage 95%: {coverage*100:.1f}% | Threshold: {thresh_opt:.2f}',
                 fontsize=13, fontweight='bold', pad=10)
    ax.set_xlabel('Muestra de Validación', fontsize=11)
    ax.set_ylabel(target.capitalize(), fontsize=11)
    ax.legend(loc='best', fontsize=9, framealpha=0.95)
    ax.grid(True, alpha=0.2, linestyle=':')
    
    # Línea promedio
    ax.axhline(y=np.mean(y_val), color='gray', linestyle='--', alpha=0.3, linewidth=1, label='Media')
    
    print(f"   MAE: {mae:.2f} | F1: {f1:.3f} (P={prec:.2f}, R={rec:.2f}) | Coverage: {coverage*100:.1f}% | Picos: {len(jump_indices_real)} | TP: {len(tp_indices)} | FP: {len(fp_indices)}")

plt.suptitle('🚀 Engagement EMMA - Predicción FFN Multitarea (Two-Stage + Calibración)', 
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()

output_path = os.path.join(os.path.dirname(__file__), 'engagement_emma.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\n✅ Gráfico guardado: {output_path}")

plt.show()
