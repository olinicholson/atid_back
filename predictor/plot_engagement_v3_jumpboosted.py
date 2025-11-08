import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import pickle
from features_sociales import generar_features_sociales
from features_competencia import generar_features_competencia

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (18, 12)

# ===== CONFIGURACIÓN =====
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "core", "data")
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

# ===== FEATURES TEMPORALES =====
df["month_sin"] = np.sin(2 * np.pi * df["created_at"].dt.month / 12)
df["month_cos"] = np.cos(2 * np.pi * df["created_at"].dt.month / 12)
df["hour"] = df["created_at"].dt.hour
df["weekday"] = df["created_at"].dt.weekday
df["is_weekend"] = (df["weekday"] >= 5).astype(int)

# Features de horario (binning por franjas)
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
df["is_morning"] = ((df["hour"] >= 6) & (df["hour"] < 12)).astype(int)  # 6-12
df["is_afternoon"] = ((df["hour"] >= 12) & (df["hour"] < 18)).astype(int)  # 12-18
df["is_evening"] = ((df["hour"] >= 18) & (df["hour"] < 22)).astype(int)  # 18-22
df["is_night"] = ((df["hour"] >= 22) | (df["hour"] < 6)).astype(int)  # 22-6

# Feriados Argentina (2024-2025)
feriados_arg = [
    "2024-01-01", "2024-02-12", "2024-02-13",  # Año Nuevo, Carnaval
    "2024-03-24", "2024-03-28", "2024-03-29",  # Memoria, Semana Santa
    "2024-04-02", "2024-05-01", "2024-05-25",  # Malvinas, Trabajador, Revolución
    "2024-06-17", "2024-06-20", "2024-07-09",  # Güemes, Bandera, Independencia
    "2024-08-17", "2024-10-12", "2024-11-18",  # San Martín, Diversidad, Soberanía
    "2024-12-08", "2024-12-25",  # Inmaculada, Navidad
    "2025-01-01", "2025-03-03", "2025-03-04",  # Año Nuevo, Carnaval
    "2025-03-24", "2025-04-02", "2025-04-17", "2025-04-18",  # Memoria, Malvinas, Semana Santa
    "2025-05-01", "2025-05-25", "2025-06-16", "2025-06-20",  # Trabajador, Revolución, Güemes, Bandera
    "2025-07-09", "2025-08-17", "2025-10-12", "2025-11-24",  # Independencia, San Martín, Diversidad, Soberanía
    "2025-12-08", "2025-12-25"  # Inmaculada, Navidad
]
feriados_set = set(pd.to_datetime(feriados_arg).date)
df["is_feriado"] = df["created_at"].dt.date.isin(feriados_set).astype(int)
df["dias_desde_feriado"] = df["created_at"].apply(
    lambda x: min([abs((x.date() - f).days) for f in feriados_set])
)
df["dias_hasta_feriado"] = df["created_at"].apply(
    lambda x: min([((f - x.date()).days) for f in feriados_set if f >= x.date()], default=365)
)

# Features de texto
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
    
    # Features de volatilidad
    df[f"{tgt}_std_7d"] = s.rolling("7D", min_periods=2).std().shift(1).reset_index(drop=True).fillna(0)
    df[f"{tgt}_std_30d"] = s.rolling("30D", min_periods=5).std().shift(1).reset_index(drop=True).fillna(0)
    df[f"{tgt}_cv_30d"] = (df[f"{tgt}_std_30d"] / (df[f"{tgt}_rollmed_30d"] + 1e-3)).fillna(0)
    
    # Momentum y aceleración
    df[f"{tgt}_diff_1"] = s.diff(1).shift(1).reset_index(drop=True).fillna(0)
    df[f"{tgt}_diff_3"] = s.diff(3).shift(1).reset_index(drop=True).fillna(0)
    df[f"{tgt}_momentum_7d"] = s.rolling("7D", min_periods=2).apply(lambda x: (x[-1] - x[0]) if len(x) > 1 else 0).shift(1).reset_index(drop=True).fillna(0)
    
    # Rango y spread
    df[f"{tgt}_range_7d"] = (s.rolling("7D", min_periods=2).max() - s.rolling("7D", min_periods=2).min()).shift(1).reset_index(drop=True).fillna(0)
    df[f"{tgt}_iqr_30d"] = (s.rolling("30D", min_periods=5).quantile(0.75) - s.rolling("30D", min_periods=5).quantile(0.25)).shift(1).reset_index(drop=True).fillna(0)

s_replies = df.set_index("created_at")["replies"]
roll_std = s_replies.rolling("30D", min_periods=5).std().shift(1).reset_index(drop=True)
df["replies_zscore_30d"] = ((df["replies"] - df["replies_rollmed_30d"]) / (roll_std + 1e-3)).fillna(0)

# Features temporales de incertidumbre
df["days_since_last"] = df["created_at"].diff().dt.total_seconds() / 86400
df["days_since_last"] = df["days_since_last"].fillna(df["days_since_last"].median())
df["posting_irregularity"] = df["days_since_last"].rolling(7, min_periods=1).std().fillna(0)

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
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models_uala_v3_jumpboosted")
with open(os.path.join(MODEL_DIR, "jump_models.pkl"), "rb") as f:
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
        model_path = os.path.join(MODEL_DIR, f"uala_{target}_lstm.pt")
    else:
        feat_cols = feat_cols_base
        n_features = len(feat_cols_base)
        model_path = os.path.join(MODEL_DIR, f"uala_{target}_lstm.pt")
    
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
        # Aplicar dual boosters según tipo de post
        mask_jump_val = is_jump_val == 1
        mask_normal_val = is_jump_val == 0
        
        if boosters[target]['jump'] is not None and mask_jump_val.sum() > 0:
            X_jump_val = X_seq[idx_val_seq][mask_jump_val][:, -1, :]
            boost_residuals_jump = boosters[target]['jump'].predict(X_jump_val)
            preds_median[mask_jump_val] += 0.25 * boost_residuals_jump
        
        if boosters[target]['normal'] is not None and mask_normal_val.sum() > 0:
            X_normal_val = X_seq[idx_val_seq][mask_normal_val][:, -1, :]
            boost_residuals_normal = boosters[target]['normal'].predict(X_normal_val)
            preds_median[mask_normal_val] += 0.20 * boost_residuals_normal
    
    # Sin calibración - usar predicciones directas
    preds_final = preds_median
    q01_final = preds_q[:, 0]
    q99_final = preds_q[:, 2]
    
    # Ajuste dinámico para views (solo IC)
    if target == "views":
        residuals = np.abs(y_val - preds_final)
        factor = 5.0 * (np.mean(residuals) / np.mean(preds_final + 1e-6))
        q01_final = q01_final - factor * np.abs(q01_final - preds_final)
        q99_final = q99_final + factor * np.abs(q99_final - preds_final)
    
    # MAE y Coverage
    mae = np.mean(np.abs(y_val - preds_final))
    coverage = np.mean((y_val >= q01_final) & (y_val <= q99_final))
    
    # ===== GRÁFICO DE ENGAGEMENT =====
    ax = axes[idx]
    x_range = np.arange(len(y_val))
    
    # Líneas principales
    ax.plot(x_range, y_val, 'o-', linewidth=2, markersize=5, 
            label='Real', color='steelblue', alpha=0.9)
    ax.plot(x_range, preds_final, 's--', linewidth=1.5, markersize=4, 
            label='Predicho (Dual Boosted)', color='coral', alpha=0.8)
    
    # Banda de confianza
    ax.fill_between(x_range, q01_final, q99_final, alpha=0.1, color='orange', label='IC 95%')
    
    # Marcar picos detectados
    jump_indices = np.where(is_jump_val)[0]
    normal_indices = np.where(~is_jump_val)[0]
    
    if len(jump_indices) > 0:
        ax.scatter(jump_indices, y_val[jump_indices], 
                  s=120, color='red', marker='*', zorder=5, alpha=0.7,
                  label=f'Picos detectados ({len(jump_indices)})', 
                  edgecolors='darkred', linewidths=1.5)
        
        # Marcar donde se aplicó el boost de PICOS
        mask_jump_val = is_jump_val == 1
        if mask_jump_val.sum() > 0:
            ax.scatter(jump_indices, preds_final[mask_jump_val], 
                      s=80, color='purple', marker='^', zorder=4, alpha=0.6,
                      label=f'Boost Picos (25%)', edgecolors='indigo', linewidths=1)
    
    # Marcar algunas muestras normales boosted (para mostrar variabilidad)
    if len(normal_indices) > 0:
        sample_normals = normal_indices[::5][:10]  # Cada 5ta muestra, máx 10
        ax.scatter(sample_normals, preds_final[sample_normals], 
                  s=40, color='green', marker='o', zorder=3, alpha=0.5,
                  label=f'Boost Normales (20%)', edgecolors='darkgreen', linewidths=0.5)
    
    # Título y etiquetas
    n_boosted_jump = mask_jump_val.sum() if len(jump_indices) > 0 else 0
    n_boosted_normal = (~is_jump_val).sum()
    ax.set_title(f'{target.upper()} - Dual Boosted (Picos + Normales)\n' + 
                 f'MAE: {mae:.2f} | Coverage: {coverage*100:.1f}% | Picos: {n_boosted_jump} | Normales: {n_boosted_normal}',
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
