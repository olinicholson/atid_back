import os
import glob
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from features_sociales import generar_features_sociales
from features_competencia import generar_features_competencia
import pickle
import random

warnings.filterwarnings("ignore")

# Fijar seeds para reproducibilidad
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ============================== CONFIG ==============================
# Project root (two levels above predictor/twitter)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "core", "data")
TARGETS = ["likes", "replies", "views"]
QUANTILES = [0.01, 0.5, 0.99]  # 98% teórico
N_Q = len(QUANTILES)

SEQ_LEN = 8
BATCH_SIZE = 32
EPOCHS = 200
LR = 5e-4
CLIP_NORM = 1.0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

VAL_WINDOW_DAYS = 60
FUTURE_GAP_DAYS = 21
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models_uala_v3_jumpboosted")
os.makedirs(MODEL_DIR, exist_ok=True)

# ============================== ARQUITECTURA LSTM ==============================
class QuantileLSTM(nn.Module):
    def __init__(self, input_dim, hidden=512, num_targets=1, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, batch_first=True, dropout=dropout)
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden, 128), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(128, N_Q)
            ) for _ in range(num_targets)
        ])

    def forward(self, x):
        out, _ = self.lstm(x)
        h = out[:, -1, :]
        outs = [head(h).unsqueeze(1) for head in self.heads]
        Q = torch.cat(outs, dim=1)
        return torch.sort(Q, dim=-1)[0]

# ============================== LOSS CON PESOS Y COVERAGE ==============================
def quantile_loss_with_coverage(preds, target, quantiles, alpha=0.5, weights=None):
    """
    Pérdida cuantil con pesos para picos + penalización de coverage suave
    alpha=0.5: Prioriza MAE sobre coverage (configuración original)
    """
    losses = []
    for i, q in enumerate(quantiles):
        errors = target - preds[:, :, i]
        loss_q = torch.max((q - 1) * errors, q * errors)
        if weights is not None:
            loss_q = loss_q * weights.unsqueeze(1)
        losses.append(loss_q)
    
    base_loss = torch.mean(torch.stack(losses))
    
    # Penalización de coverage suave (no forzar coverage alto)
    q_low = preds[:, :, 0]
    q_high = preds[:, :, -1]
    coverage = torch.mean(((target >= q_low) & (target <= q_high)).float())
    coverage_penalty = alpha * torch.relu(0.98 - coverage)
    
    return base_loss + coverage_penalty

# ============================== TRAINING LOOP ==============================
def train_one_epoch(model, opt, loader, quantiles):
    model.train()
    losses = []
    for X_batch, Y_batch in loader:
        opt.zero_grad()
        
        # Extraer jump_intensity del último feature
        jump_intensity = X_batch[:, -1, -1]
        
        # Pesos más suaves: 1.0 base + hasta 2.0 adicional para picos muy intensos
        weights = 1.0 + 2.0 * torch.sigmoid(jump_intensity * 0.5)
        
        preds = model(X_batch)
        loss = quantile_loss_with_coverage(preds, Y_batch, quantiles, weights=weights)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
        opt.step()
        losses.append(loss.item())
    return np.mean(losses)

def eval_model(model, loader):
    model.eval()
    all_preds, all_Y = [], []
    with torch.no_grad():
        for X_batch, Y_batch in loader:
            preds = model(X_batch)
            all_preds.append(preds[:, :, 1].cpu().numpy())
            all_Y.append(Y_batch.cpu().numpy())
    preds_concat = np.concatenate(all_preds, axis=0)
    Y_concat = np.concatenate(all_Y, axis=0)
    mae = mean_absolute_error(Y_concat[:, 0], preds_concat[:, 0])
    return mae

# ============================== PIPELINE PRINCIPAL ==============================
def run_pipeline():
    print("🚀 Iniciando pipeline Ualá v3 JumpBoosted (picos amplificados)")

    # Cargar datasets
    # Prefer the 'with_trends' files, but fall back to any posts_*.csv if needed.
    files = glob.glob(os.path.join(DATA_DIR, "posts_*with_trends*.csv"))
    if len(files) == 0:
        files = glob.glob(os.path.join(DATA_DIR, "posts_*.csv"))

    dfs = []
    for f in files:
        try:
            df_temp = pd.read_csv(f)
        except Exception as e:
            print(f"WARN: unable to read {f}: {e}")
            continue
        # normalize dataset_name as before
        name = os.path.basename(f).split("_with_trends")[0].replace("posts_", "").replace('.csv','').lower()
        df_temp["dataset_name"] = name
        dfs.append(df_temp)

    if len(dfs) == 0:
        raise RuntimeError(f"No post files found in DATA_DIR={DATA_DIR}. Looked for posts_*with_trends*.csv and posts_*.csv.\nPlease ensure the core/data files are present or adjust DATA_DIR.")

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

    # Sociales y competencia
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

    # Rolling + EMA
    df = df.sort_values("created_at").reset_index(drop=True)
    for tgt in TARGETS:
        s = df.set_index("created_at")[tgt]
        df[f"{tgt}_rollmed_30d"] = s.rolling("30D", min_periods=3).median().shift(1).reset_index(drop=True)
        df[f"{tgt}_ema_14d"] = s.ewm(span=14, min_periods=3, adjust=False).mean().shift(1).reset_index(drop=True)
        df[f"{tgt}_rel"] = df[tgt] / (1e-3 + df[f"{tgt}_rollmed_30d"])
        
        # 🆕 FEATURES DE VOLATILIDAD para IC dinámico
        df[f"{tgt}_std_7d"] = s.rolling("7D", min_periods=2).std().shift(1).reset_index(drop=True).fillna(0)
        df[f"{tgt}_std_30d"] = s.rolling("30D", min_periods=5).std().shift(1).reset_index(drop=True).fillna(0)
        df[f"{tgt}_cv_30d"] = (df[f"{tgt}_std_30d"] / (df[f"{tgt}_rollmed_30d"] + 1e-3)).fillna(0)  # Coeficiente variación
        
        # 🆕 MOMENTUM y ACELERACIÓN
        df[f"{tgt}_diff_1"] = s.diff(1).shift(1).reset_index(drop=True).fillna(0)  # Cambio último post
        df[f"{tgt}_diff_3"] = s.diff(3).shift(1).reset_index(drop=True).fillna(0)  # Cambio últimos 3
        df[f"{tgt}_momentum_7d"] = s.rolling("7D", min_periods=2).apply(lambda x: (x[-1] - x[0]) if len(x) > 1 else 0).shift(1).reset_index(drop=True).fillna(0)
        
        # 🆕 RANGO y SPREAD
        df[f"{tgt}_range_7d"] = (s.rolling("7D", min_periods=2).max() - s.rolling("7D", min_periods=2).min()).shift(1).reset_index(drop=True).fillna(0)
        df[f"{tgt}_iqr_30d"] = (s.rolling("30D", min_periods=5).quantile(0.75) - s.rolling("30D", min_periods=5).quantile(0.25)).shift(1).reset_index(drop=True).fillna(0)

    s_replies = df.set_index("created_at")["replies"]
    roll_std = s_replies.rolling("30D", min_periods=5).std().shift(1).reset_index(drop=True)
    df["replies_zscore_30d"] = ((df["replies"] - df["replies_rollmed_30d"]) / (roll_std + 1e-3)).fillna(0)
    
    # 🆕 FEATURES TEMPORALES DE INCERTIDUMBRE
    df["days_since_last"] = df["created_at"].diff().dt.total_seconds() / 86400
    df["days_since_last"] = df["days_since_last"].fillna(df["days_since_last"].median())
    df["posting_irregularity"] = df["days_since_last"].rolling(7, min_periods=1).std().fillna(0)
    
    print(f"✅ Features de volatilidad agregadas (std, cv, momentum, range, iqr)")

    # ===== 1️⃣ INTENSIDAD DE PICO (continua) =====
    print("\n🔍 Calculando intensidad de picos...")
    df["jump_intensity"] = np.maximum.reduce([
        df["likes"] / (1e-3 + df["likes_rollmed_30d"]),
        df["replies"] / (1e-3 + df["replies_rollmed_30d"]),
        df["views"] / (1e-3 + df["views_rollmed_30d"])
    ])
    
    # Escalar logarítmicamente
    df["jump_intensity"] = np.log1p(df["jump_intensity"])
    df["jump_intensity"] = (df["jump_intensity"] - df["jump_intensity"].mean()) / (df["jump_intensity"].std() + 1e-6)
    
    # Detectar picos binarios (para clasificador y stats)
    df["is_jump"] = (
        (df["likes"] > df["likes_rollmed_30d"] * 2.5) |
        (df["replies"] > df["replies_rollmed_30d"] * 2.5) |
        (df["views"] > df["views_rollmed_30d"] * 2.0)
    ).astype(int)
    
    n_jumps = df["is_jump"].sum()
    print(f"   📊 Picos detectados: {n_jumps} ({n_jumps/len(df)*100:.1f}%)")
    print(f"   📈 Jump intensity stats: mean={df['jump_intensity'].mean():.2f}, std={df['jump_intensity'].std():.2f}")

    # Features base
    exclude_cols = ["created_at", "text", "dataset_name", "hashtags", "date", "best_trend", "username", "is_jump", "jump_intensity"]
    feat_cols_base = [c for c in df.columns if c not in TARGETS + exclude_cols
                      and df[c].dtype in ['int64', 'float64', 'float32', 'bool'] and 'zscore' not in c]
    feat_cols_replies = feat_cols_base + ['replies_zscore_30d']

    # ===== 2️⃣ ENTRENAR CLASIFICADOR DE PICOS =====
    print("\n🎯 Entrenando clasificador de picos...")
    
    # Split temporal para clasificador
    cutoff_clf = df["created_at"].max() - pd.Timedelta(days=VAL_WINDOW_DAYS + FUTURE_GAP_DAYS)
    idx_train_clf = df["created_at"] <= cutoff_clf
    idx_val_clf = df["created_at"] > cutoff_clf
    
    X_clf = df[feat_cols_base].fillna(0).values
    X_clf = np.nan_to_num(X_clf, nan=0.0, posinf=0.0, neginf=0.0)
    y_clf = df["is_jump"].values
    
    # Train/Val split
    X_clf_train = X_clf[idx_train_clf]
    y_clf_train = y_clf[idx_train_clf]
    X_clf_val = X_clf[idx_val_clf]
    y_clf_val = y_clf[idx_val_clf]
    
    # Escalar
    scaler_clf = StandardScaler()
    X_clf_train_scaled = scaler_clf.fit_transform(X_clf_train)
    X_clf_val_scaled = scaler_clf.transform(X_clf_val)
    X_clf_scaled = scaler_clf.transform(X_clf)  # Para usar después
    
    # Entrenar solo en train
    clf_jump = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42)
    clf_jump.fit(X_clf_train_scaled, y_clf_train)
    
    # Evaluar en val (no en train!)
    y_pred_clf_val = clf_jump.predict(X_clf_val_scaled)
    print("\n📋 Reporte clasificador (validation set):")
    print(classification_report(y_clf_val, y_pred_clf_val, target_names=["Normal", "Pico"]))

    # ===== 3️⃣ ENTRENAR MODELOS LSTM + RESIDUAL BOOSTERS =====
    print("\n" + "="*60)
    
    def crear_secuencias(data, targets, seq_len=8):
        X_list, Y_list = [], []
        for i in range(len(data) - seq_len):
            X_list.append(data[i:i+seq_len])
            Y_list.append(targets[i+seq_len])
        return np.array(X_list), np.array(Y_list)

    # Split temporal
    cutoff_date = df["created_at"].max() - pd.Timedelta(days=VAL_WINDOW_DAYS + FUTURE_GAP_DAYS)
    idx_train = df["created_at"] <= cutoff_date
    idx_val = df["created_at"] > cutoff_date

    results = {}
    residual_boosters = {}
    
    for target_name in TARGETS:
        print(f"\n{'='*60}")
        print(f"🎯 Entrenando modelo para: {target_name.upper()}")
        print(f"{'='*60}")
        
        # Determinar features
        if target_name == "replies":
            feat_cols = feat_cols_replies
        else:
            feat_cols = feat_cols_base
        
        n_features = len(feat_cols)
        print(f"📊 Features: {n_features} + jump_intensity")
        
        # Añadir jump_intensity como feature continua
        X_with_jump = df[feat_cols + ["jump_intensity"]].fillna(0).values
        X_with_jump = np.nan_to_num(X_with_jump, nan=0.0, posinf=0.0, neginf=0.0)
        Y_target = df[[target_name]].values
        
        # Crear secuencias
        X_seq, Y_seq = crear_secuencias(X_with_jump, Y_target, seq_len=SEQ_LEN)
        
        # Índices después de crear secuencias
        dates_seq = df["created_at"].iloc[SEQ_LEN:].reset_index(drop=True)
        idx_train_seq = dates_seq <= cutoff_date
        idx_val_seq = dates_seq > cutoff_date
        
        X_train = torch.FloatTensor(X_seq[idx_train_seq]).to(DEVICE)
        Y_train = torch.FloatTensor(Y_seq[idx_train_seq]).to(DEVICE)
        X_val = torch.FloatTensor(X_seq[idx_val_seq]).to(DEVICE)
        Y_val = torch.FloatTensor(Y_seq[idx_val_seq]).to(DEVICE)
        
        print(f"📊 Train: {len(X_train)}, Val: {len(X_val)}")
        
        # DataLoaders
        train_dataset = TensorDataset(X_train, Y_train)
        val_dataset = TensorDataset(X_val, Y_val)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        # Modelo LSTM
        model = QuantileLSTM(input_dim=n_features+1, hidden=256, num_targets=1, dropout=0.3).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        
        # Early stopping
        best_mae = float('inf')
        patience_counter = 0
        patience = 30
        
        for epoch in range(1, EPOCHS + 1):
            train_loss = train_one_epoch(model, optimizer, train_loader, QUANTILES)
            val_mae = eval_model(model, val_loader)
            
            if epoch % 10 == 0:
                print(f"Ep{epoch:03d} train={train_loss:.3f} val={val_mae:.3f}")
            
            if val_mae < best_mae:
                best_mae = val_mae
                patience_counter = 0
                torch.save(model.state_dict(), 
                          os.path.join(MODEL_DIR, f"uala_{target_name}_lstm.pt"))
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping!")
                    break
        
        # Cargar mejor modelo
        model.load_state_dict(torch.load(os.path.join(MODEL_DIR, f"uala_{target_name}_lstm.pt")))
        model.eval()
        
        # ===== 4️⃣ ENTRENAR DUAL RESIDUAL BOOSTERS =====
        print(f"\n🔧 Entrenando residual boosters (picos + normales) para {target_name}...")
        
        with torch.no_grad():
            preds_train = model(X_train).cpu().numpy()[:, 0, 1]  # Mediana
        
        Y_train_np = Y_train.cpu().numpy()[:, 0]
        residuals_train = Y_train_np - preds_train
        
        # Obtener is_jump para train sequences
        is_jump_seq = df["is_jump"].iloc[SEQ_LEN:].reset_index(drop=True).values
        is_jump_train = is_jump_seq[idx_train_seq]
        
        # 4A: Entrenar booster para PICOS
        mask_jump_train = is_jump_train == 1
        booster_jump = None
        if mask_jump_train.sum() > 10:
            X_jump_train = X_seq[idx_train_seq][mask_jump_train][:, -1, :]
            residuals_jump = residuals_train[mask_jump_train]
            
            booster_jump = GradientBoostingRegressor(n_estimators=150, learning_rate=0.05, max_depth=4, random_state=42)
            booster_jump.fit(X_jump_train, residuals_jump)
            print(f"   ✅ Booster PICOS entrenado con {mask_jump_train.sum()} muestras")
        else:
            print(f"   ⚠️ No hay suficientes picos para entrenar booster")
        
        # 4B: Entrenar booster para NORMALES
        mask_normal_train = is_jump_train == 0
        booster_normal = None
        if mask_normal_train.sum() > 20:
            X_normal_train = X_seq[idx_train_seq][mask_normal_train][:, -1, :]
            residuals_normal = residuals_train[mask_normal_train]
            
            booster_normal = GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42)
            booster_normal.fit(X_normal_train, residuals_normal)
            print(f"   ✅ Booster NORMALES entrenado con {mask_normal_train.sum()} muestras")
        else:
            print(f"   ⚠️ No hay suficientes normales para entrenar booster")
        
        # Guardar ambos boosters (incluyendo la lista de features usada por el booster)
        # Guardamos explícitamente 'feat_cols' = per-target feature cols + ['jump_intensity']
        residual_boosters[target_name] = {
            'jump': booster_jump,
            'normal': booster_normal,
            'feat_cols': list(feat_cols) + ['jump_intensity']
        }
        
        # Evaluación final con boosters duales
        with torch.no_grad():
            preds_q = model(X_val).cpu().numpy()
        
        Y_val_np = Y_val.cpu().numpy()[:, 0]
        preds_median = preds_q[:, 0, 1].copy()
        
        # Aplicar boosters según tipo de post
        is_jump_val = is_jump_seq[idx_val_seq]
        n_boosted_jump = 0
        n_boosted_normal = 0
        
        # Aplicar booster de PICOS
        if booster_jump is not None:
            mask_jump_val = is_jump_val == 1
            if mask_jump_val.sum() > 0:
                X_jump_val = X_seq[idx_val_seq][mask_jump_val][:, -1, :]
                boost_residuals_jump = booster_jump.predict(X_jump_val)
                preds_median[mask_jump_val] += 0.25 * boost_residuals_jump
                n_boosted_jump = mask_jump_val.sum()
        
        # Aplicar booster de NORMALES
        if booster_normal is not None:
            mask_normal_val = is_jump_val == 0
            if mask_normal_val.sum() > 0:
                X_normal_val = X_seq[idx_val_seq][mask_normal_val][:, -1, :]
                boost_residuals_normal = booster_normal.predict(X_normal_val)
                preds_median[mask_normal_val] += 0.20 * boost_residuals_normal
                n_boosted_normal = mask_normal_val.sum()
        
        print(f"   📈 Boosted: {n_boosted_jump} picos (25%) + {n_boosted_normal} normales (20%) [ÓPTIMO]")
        
        mae_final = mean_absolute_error(Y_val_np, preds_median)
        
        # Coverage
        q01 = preds_q[:, 0, 0]
        q99 = preds_q[:, 0, 2]
        coverage = np.mean((Y_val_np >= q01) & (Y_val_np <= q99))
        
        print(f"✅ {target_name}: MAE={mae_final:.3f} | Coverage 98%={coverage*100:.1f}%")
        
        results[target_name] = {
            "mae": mae_final,
            "coverage": coverage,
            "n_features": n_features + 1
        }
    
    # Guardar clasificador y boosters
    with open(os.path.join(MODEL_DIR, "jump_models.pkl"), "wb") as f:
        pickle.dump({
            "clf": clf_jump, 
            "scaler": scaler_clf,
            "boosters": residual_boosters
        }, f)
    print(f"\n✅ Clasificador y boosters guardados")
    # Guardar metadata útil para loaders/plotters: features + model dims
    try:
        import json
        feature_meta = {
            "seq_len": SEQ_LEN,
            "targets": {},
                "clf_features": feat_cols_base
        }
        for tgt in TARGETS:
            # Use the same naming we used during training
            if tgt == "replies":
                cols = feat_cols_replies
            else:
                cols = feat_cols_base
            # also expose booster-specific feature ordering so inference can exactly
            # reconstruct the vector used to train residual boosters
            feature_meta["targets"][tgt] = {
                "feat_cols": cols,
                "n_features": len(cols),
                "input_dim": len(cols) + 1,
                "hidden": 256,
                "head_hidden": 128,
                "booster_feat_cols": list(cols) + ['jump_intensity']
            }
        with open(os.path.join(MODEL_DIR, "feature_meta.json"), "w", encoding="utf-8") as fh:
            json.dump(feature_meta, fh, ensure_ascii=False, indent=2)
        print(f"✅ Feature metadata saved to {os.path.join(MODEL_DIR, 'feature_meta.json')}")
    except Exception as e:
        print(f"WARN: failed to write feature_meta.json: {e}")
    
    # ===== RESUMEN FINAL =====
    print("\n" + "="*60)
    print("📊 RESULTADOS FINALES (val set con boosting):")
    print("="*60)
    for tgt, res in results.items():
        print(f"{tgt:8s}: MAE={res['mae']:.3f}")
    print("="*60)

if __name__ == "__main__":
    run_pipeline()
