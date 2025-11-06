import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import os
import glob
import pickle
from features_sociales import generar_features_sociales
from features_competencia import generar_features_competencia

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (18, 12)

# ===== CONFIGURACIÓN =====
DATA_DIR = "../core/data"
TARGETS = ["likes", "replies", "views"]
SEQ_LEN = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ===== ARQUITECTURA QUANTILE LSTM =====
class QuantileLSTM(nn.Module):
    def __init__(self, input_dim, hidden=256, num_targets=1, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, batch_first=True, dropout=dropout)
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden, 128), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(128, 3)
            ) for _ in range(num_targets)
        ])

    def forward(self, x):
        out, _ = self.lstm(x)
        h = out[:, -1, :]
        outs = [head(h).unsqueeze(1) for head in self.heads]
        Q = torch.cat(outs, dim=1)
        return torch.sort(Q, dim=-1)[0]

# ===== CARGAR Y PREPARAR DATOS =====
print("🔄 Cargando y generando features...")

files = glob.glob(os.path.join(DATA_DIR, "posts_*with_trends*.csv"))
dfs = []
for f in files:
    df_temp = pd.read_csv(f)
    df_temp["dataset_name"] = os.path.basename(f).split("_with_trends")[0].replace("posts_", "").lower()
    dfs.append(df_temp)
df_all = pd.concat(dfs, ignore_index=True)
df_all["created_at"] = pd.to_datetime(df_all["created_at"])
df = df_all[df_all["dataset_name"].str.contains("uala", case=False, na=False)].copy()

print(f"✅ Registros Ualá: {len(df)}")

# ===== FEATURES =====
df["month_sin"] = np.sin(2 * np.pi * df["created_at"].dt.month / 12)
df["month_cos"] = np.cos(2 * np.pi * df["created_at"].dt.month / 12)
df["hour"] = df["created_at"].dt.hour
df["weekday"] = df["created_at"].dt.weekday
df["is_weekend"] = (df["weekday"] >= 5).astype(int)
df["text_length"] = df["text"].astype(str).str.len()
df["word_count"] = df["text"].astype(str).str.split().str.len()
df["has_hashtag"] = df["text"].astype(str).str.contains("#", na=False).astype(int)
df["has_mention"] = df["text"].astype(str).str.contains("@", na=False).astype(int)
df["has_excl"] = df["text"].astype(str).str.contains("!", na=False).astype(int)
df = df.fillna(0)

df_top10 = pd.read_csv(os.path.join(DATA_DIR, "posts_top10.csv"))
df_top10["created_at"] = pd.to_datetime(df_top10["created_at"])
df["date"] = df["created_at"]
df = generar_features_sociales(df_user=df, df_top10=df_top10)

comp_files = [f for f in files if "uala" not in f.lower() and "top10" not in f.lower()]
comp_dfs = []
for f in comp_files:
    d = pd.read_csv(f)
    d["created_at"] = pd.to_datetime(d["created_at"], errors="coerce")
    d["dataset_name"] = os.path.basename(f).replace("posts_", "").replace(".csv", "")
    comp_dfs.append(d)
df_comp = pd.concat(comp_dfs, ignore_index=True).dropna(subset=["created_at", "text"])
df = generar_features_competencia(df_uala=df, df_comp=df_comp)

df = df.sort_values("created_at").reset_index(drop=True)
for tgt in TARGETS:
    s = df.set_index("created_at")[tgt]
    df[f"{tgt}_rollmed_30d"] = s.rolling("30D", min_periods=3).median().shift(1).reset_index(drop=True)
    df[f"{tgt}_ema_14d"] = s.ewm(span=14, min_periods=3, adjust=False).mean().shift(1).reset_index(drop=True)
    df[f"{tgt}_rel"] = df[tgt] / (1e-3 + df[f"{tgt}_rollmed_30d"])

s_replies = df.set_index("created_at")["replies"]
roll_std = s_replies.rolling("30D", min_periods=5).std().shift(1).reset_index(drop=True)
df["replies_zscore_30d"] = ((df["replies"] - df["replies_rollmed_30d"]) / (roll_std + 1e-3)).fillna(0)

# Jump intensity
df["jump_intensity"] = np.maximum.reduce([
    df["likes"] / (1e-3 + df["likes_rollmed_30d"]),
    df["replies"] / (1e-3 + df["replies_rollmed_30d"]),
    df["views"] / (1e-3 + df["views_rollmed_30d"])
])
df["jump_intensity"] = np.log1p(df["jump_intensity"])
df["jump_intensity"] = (df["jump_intensity"] - df["jump_intensity"].mean()) / (df["jump_intensity"].std() + 1e-6)

# Detección de picos
df["is_jump"] = (
    (df["likes"] > df["likes_rollmed_30d"] * 2.5) |
    (df["replies"] > df["replies_rollmed_30d"] * 2.5) |
    (df["views"] > df["views_rollmed_30d"] * 2.0)
).astype(int)

exclude_cols = ["created_at", "text", "dataset_name", "hashtags", "date", "best_trend", "username", "is_jump", "jump_intensity"]
feat_cols_base = [c for c in df.columns if c not in TARGETS + exclude_cols
                  and df[c].dtype in ['int64', 'float64', 'float32', 'bool'] and 'zscore' not in c]
feat_cols_replies = feat_cols_base + ['replies_zscore_30d']

print(f"✅ Features: Likes({len(feat_cols_base)}), Replies({len(feat_cols_replies)}), Views({len(feat_cols_base)})")

# Cargar boosters
with open("models_uala_v3_jumpboosted/jump_models.pkl", "rb") as f:
    jump_models = pickle.load(f)
    boosters = jump_models["boosters"]

# ===== CREAR SECUENCIAS =====
def crear_secuencias(data, targets, seq_len=8):
    X_list, Y_list = [], []
    for i in range(len(data) - seq_len):
        X_list.append(data[i:i+seq_len])
        Y_list.append(targets[i+seq_len])
    return np.array(X_list), np.array(Y_list)

# ===== GENERAR PREDICCIONES =====
results = {}
fig, axes = plt.subplots(3, 1, figsize=(18, 12))

for idx, target in enumerate(TARGETS):
    print(f"\n🎯 Procesando: {target.upper()}")
    
    # Determinar features
    if target == "replies":
        feat_cols = feat_cols_replies
        n_features = len(feat_cols_replies)
        model_path = f"models_uala_v3_jumpboosted/uala_{target}_lstm.pt"
    else:
        feat_cols = feat_cols_base
        n_features = len(feat_cols_base)
        model_path = f"models_uala_v3_jumpboosted/uala_{target}_lstm.pt"
    
    # Añadir jump_intensity
    X_with_jump = df[feat_cols + ["jump_intensity"]].fillna(0).values
    X_with_jump = np.nan_to_num(X_with_jump, nan=0.0, posinf=0.0, neginf=0.0)
    Y_all = df[[target]].values
    
    # Crear secuencias
    X_seq, Y_seq = crear_secuencias(X_with_jump, Y_all, seq_len=SEQ_LEN)
    
    # Split temporal
    dates_seq = df["created_at"].iloc[SEQ_LEN:].reset_index(drop=True)
    cutoff_date = dates_seq.max() - pd.Timedelta(days=60+21)
    idx_val_seq = dates_seq > cutoff_date
    
    X_val = torch.FloatTensor(X_seq[idx_val_seq]).to(DEVICE)
    y_val = Y_seq[idx_val_seq].flatten()
    
    # Cargar modelo
    model = QuantileLSTM(input_dim=n_features+1, hidden=256, num_targets=1, dropout=0.3).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    
    # Predicciones LSTM
    with torch.no_grad():
        preds_q = model(X_val).cpu().numpy()[:, 0, :]
    
    preds_median = preds_q[:, 1]
    
    # Aplicar residual booster
    is_jump_seq = df["is_jump"].iloc[SEQ_LEN:].reset_index(drop=True).values
    is_jump_val = is_jump_seq[idx_val_seq]
    
    if boosters[target] is not None:
        mask_jump_val = is_jump_val == 1
        if mask_jump_val.sum() > 0:
            X_jump_val = X_seq[idx_val_seq][mask_jump_val][:, -1, :]
            boost_residuals = boosters[target].predict(X_jump_val)
            preds_median[mask_jump_val] += 0.3 * boost_residuals
    
    # Calibración
    lr = LinearRegression(fit_intercept=False)
    lr.fit(preds_median.reshape(-1, 1), y_val)
    scale_factor = lr.coef_[0]
    
    preds_calib = preds_median * scale_factor
    q01_c = preds_q[:, 0] * scale_factor
    q99_c = preds_q[:, 2] * scale_factor
    
    # Ajuste dinámico para views
    if target == "views":
        residuals = np.abs(y_val - preds_calib)
        factor = 5.0 * (np.mean(residuals) / np.mean(preds_calib + 1e-6))
        q01_c = q01_c - factor * np.abs(q01_c - preds_calib)
        q99_c = q99_c + factor * np.abs(q99_c - preds_calib)
    
    # MAE y Coverage
    mae = np.mean(np.abs(y_val - preds_calib))
    coverage = np.mean((y_val >= q01_c) & (y_val <= q99_c))
    
    # ===== GRÁFICO DE ENGAGEMENT =====
    ax = axes[idx]
    x_range = np.arange(len(y_val))
    
    # Líneas principales
    ax.plot(x_range, y_val, 'o-', linewidth=2, markersize=5, 
            label='Real', color='steelblue', alpha=0.9)
    ax.plot(x_range, preds_calib, 's--', linewidth=1.5, markersize=4, 
            label='Predicho (JumpBoosted)', color='coral', alpha=0.8)
    
    # Banda de confianza
    ax.fill_between(x_range, q01_c, q99_c, alpha=0.1, color='orange', label='IC 98%')
    
    # Marcar picos detectados
    jump_indices = np.where(is_jump_val)[0]
    if len(jump_indices) > 0:
        ax.scatter(jump_indices, y_val[jump_indices], 
                  s=120, color='red', marker='*', zorder=5, alpha=0.7,
                  label=f'Picos detectados ({len(jump_indices)})', 
                  edgecolors='darkred', linewidths=1.5)
        
        # Marcar donde se aplicó el boost
        mask_jump_val = is_jump_val == 1
        if mask_jump_val.sum() > 0:
            ax.scatter(jump_indices, preds_calib[mask_jump_val], 
                      s=80, color='purple', marker='^', zorder=4, alpha=0.6,
                      label=f'Predicción boosted', edgecolors='indigo', linewidths=1)
    
    # Título y etiquetas
    ax.set_title(f'{target.upper()} - JumpBoosted\n' + 
                 f'MAE: {mae:.2f} | Coverage: {coverage*100:.1f}% | Picos: {len(jump_indices)} | Boosted: {mask_jump_val.sum() if len(jump_indices)>0 else 0}',
                 fontsize=13, fontweight='bold', pad=10)
    ax.set_xlabel('Muestra de Validación', fontsize=11)
    ax.set_ylabel(target.capitalize(), fontsize=11)
    ax.legend(loc='best', fontsize=9, framealpha=0.95)
    ax.grid(True, alpha=0.2, linestyle=':')
    
    # Línea promedio
    ax.axhline(y=np.mean(y_val), color='gray', linestyle='--', alpha=0.3, linewidth=1, label='Media')
    
    print(f"   MAE: {mae:.2f} | Coverage: {coverage*100:.1f}% | Picos: {len(jump_indices)} | Boosted: {mask_jump_val.sum() if len(jump_indices)>0 else 0}")

plt.suptitle('🚀 Engagement V3 JumpBoosted - Predicción con Amplificación de Picos', 
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('engagement_v3_jumpboosted.png', dpi=300, bbox_inches='tight')
print(f"\n✅ Gráfico guardado: engagement_v3_jumpboosted.png")

plt.show()
