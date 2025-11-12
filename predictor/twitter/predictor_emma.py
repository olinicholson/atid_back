# predictor_multitask_ffn_v4_5_balanced_calibrated.py
import os, glob, warnings, json, random, pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import mean_absolute_error, f1_score, precision_score, recall_score, classification_report
from sklearn.preprocessing import StandardScaler
from features_sociales import generar_features_sociales

warnings.filterwarnings("ignore")

# ============================== SEEDS ==============================
SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ============================== CONFIG ==============================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "core", "data")
TARGETS = ["likes", "replies", "views"]

BATCH_SIZE = 64
LR = 8e-4
WEIGHT_DECAY = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

VAL_WINDOW_DAYS = 60
FUTURE_GAP_DAYS = 21

# Fases
EPOCHS_STAGE1 = 15   # Clasificación pura
EPOCHS_STAGE2 = 45   # Multitarea

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models_uala_multitask_ffn_v4_5_2stage_calib")
os.makedirs(MODEL_DIR, exist_ok=True)

# ============================== MODELOS ==============================
class MultiTaskFFNShared(nn.Module):
    """
    Backbone MLP -> shared -> dos cabezas (clf/reg).
    Cabeza de clasificación más profunda para ayudar a F1.
    """
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

class FocalBCEWithLogitsLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=3.5, pos_weight=None, label_smoothing=0.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight  # tensor([w+]) en device
        self.smooth = label_smoothing

    def forward(self, logits, targets):
        if targets.dim() == 1: targets = targets.unsqueeze(1)
        if self.smooth > 0:
            targets = targets*(1 - self.smooth) + 0.5*self.smooth

        p = torch.sigmoid(logits).clamp(1e-6, 1-1e-6)
        pw = 1.0 if self.pos_weight is None else self.pos_weight
        loss_pos = - self.alpha * pw * targets * torch.log(p)
        loss_neg = - (1 - self.alpha) * (1 - targets) * torch.log(1 - p)
        bce = loss_pos + loss_neg
        mod = (1 - torch.abs(targets - p)) ** self.gamma
        return (mod * bce).mean()

class MultiTaskLoss(nn.Module):
    def __init__(self, focal_alpha=0.75, focal_gamma=3.5, delta=1.0, pos_weight=None, label_smoothing=0.0):
        super().__init__()
        self.focal = FocalBCEWithLogitsLoss(focal_alpha, focal_gamma, pos_weight, label_smoothing)
        self.delta = delta
    def forward(self, logits, y_jump, y_pred_norm, y_true_norm, alpha, beta):
        l_clf = self.focal(logits, y_jump)
        l_reg = F.huber_loss(y_pred_norm, y_true_norm, delta=self.delta, reduction="mean")
        return alpha*l_clf + beta*l_reg, l_clf.item(), l_reg.item()

# ============================== EVAL & THRESH CALIB ==============================
@torch.no_grad()
def eval_epoch(model, loader, q90_train, thresh=0.5):
    model.eval()
    p_all, y_jump, y_pred_n, y_true_n = [], [], [], []
    for Xb, Yb, Jb in loader:
        logit, yhat = model(Xb)
        p_all.append(torch.sigmoid(logit).cpu().numpy().ravel())
        y_jump.append(Jb.cpu().numpy().ravel())
        y_pred_n.append(yhat.cpu().numpy().ravel())
        y_true_n.append(Yb.cpu().numpy().ravel())

    p = np.concatenate(p_all)
    y_true_jump = np.concatenate(y_jump)
    y_pred_jump = (p >= thresh).astype(int)

    y_pred = np.expm1(np.concatenate(y_pred_n)) * q90_train
    y_true = np.expm1(np.concatenate(y_true_n)) * q90_train

    return {
        "f1": f1_score(y_true_jump, y_pred_jump, zero_division=0),
        "prec": precision_score(y_true_jump, y_pred_jump, zero_division=0),
        "rec": recall_score(y_true_jump, y_pred_jump, zero_division=0),
        "mae": mean_absolute_error(y_true, y_pred),
        "p": p, "y_cls": y_true_jump, "y_hat_cls": y_pred_jump,
        "y_true": y_true, "y_pred": y_pred
    }

def calibrate_threshold(p, y_true, grid=np.linspace(0.1, 0.5, 9)):
    best_thr, best_f1 = 0.5, -1
    for thr in grid:
        preds = (p >= thr).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_thr, best_f1 = float(thr), float(f1)
    return best_thr, best_f1

# ============================== SAMPLER ==============================
def make_weighted_sampler(labels, pos_factor=3.0):
    weights = np.where(labels == 1, pos_factor, 1.0).astype(np.float32)
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

# ============================== TRAIN ==============================
def train_two_stage_target(target_name, df, feat_cols, cutoff_date):
    print(f"\n{'='*60}\n🎯 Entrenando v4.5 Two-Stage Calibrated: {target_name.upper()}\n{'='*60}")

    # Matrices
    X = df[feat_cols + ["jump_intensity"]].fillna(0).values
    y_raw = df[target_name].values.astype(float)
    y_jump = df["is_jump"].values.astype(int)

    # Split temporal
    idx_tr = df["created_at"] <= cutoff_date
    idx_va = ~idx_tr
    Xtr, Xva = X[idx_tr], X[idx_va]
    ytr, yva = y_raw[idx_tr], y_raw[idx_va]
    jtr, jva = y_jump[idx_tr], y_jump[idx_va]

    # Escala robusta (train only)
    q90 = np.quantile(ytr, 0.90) + 1e-6
    ytr_n = np.log1p(ytr / q90)
    yva_n = np.log1p(yva / q90)

    scaler = StandardScaler()
    Xtr = scaler.fit_transform(Xtr)
    Xva = scaler.transform(Xva)

    # Tensores
    Xtr_t = torch.tensor(Xtr).float().to(DEVICE)
    Xva_t = torch.tensor(Xva).float().to(DEVICE)
    Ytr_t = torch.tensor(ytr_n).float().unsqueeze(1).to(DEVICE)
    Yva_t = torch.tensor(yva_n).float().unsqueeze(1).to(DEVICE)
    Jtr_t = torch.tensor(jtr).float().to(DEVICE)
    Jva_t = torch.tensor(jva).float().to(DEVICE)

    # DataLoaders (oversampling x3 en train)
    sampler = make_weighted_sampler(jtr, pos_factor=3.0)
    train_dl = DataLoader(TensorDataset(Xtr_t, Ytr_t, Jtr_t), batch_size=BATCH_SIZE, sampler=sampler)
    val_dl   = DataLoader(TensorDataset(Xva_t, Yva_t, Jva_t), batch_size=BATCH_SIZE*2, shuffle=False)

    # pos_weight reforzado: mínimo 5% de tasa efectiva
    pos_rate = max(1e-6, float(jtr.mean()))
    pos_rate_eff = max(pos_rate, 0.05)
    pos_weight = torch.tensor([(1 - pos_rate_eff) / pos_rate_eff], device=DEVICE)

    model = MultiTaskFFNShared(input_dim=Xtr.shape[1], hidden=(256,128), shared_dim=128, dropout=0.30).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # -------- Stage 1: Clasificación pura --------
    stage1_loss = FocalBCEWithLogitsLoss(alpha=0.75, gamma=3.5, pos_weight=pos_weight, label_smoothing=0.05)
    best_f1_s1, wait_s1, patience_s1 = -1, 0, 8
    for ep in range(1, EPOCHS_STAGE1 + 1):
        model.train()
        losses = []
        for Xb, Yb, Jb in train_dl:
            optimizer.zero_grad()
            logit, _ = model(Xb)
            loss = stage1_loss(logit, Jb)  # β = 0 → solo clasificación
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())

        # eval temporal con thr 0.5 (solo para ver tendencia)
        pack = eval_epoch(model, val_dl, q90, thresh=0.5)
        if pack["f1"] > best_f1_s1:
            best_f1_s1, wait_s1 = pack["f1"], 0
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, f"{target_name}_stage1_best.pt"))
        else:
            wait_s1 += 1
            if wait_s1 >= patience_s1:
                print("Early stopping (Stage 1)"); break

        if ep % 5 == 0 or ep == 1:
            print(f"[{target_name} S1 Ep{ep:03d}] clf_loss={np.mean(losses):.3f} | F1={pack['f1']:.3f} (P={pack['prec']:.2f},R={pack['rec']:.2f})")

    # Cargar mejor Stage 1
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, f"{target_name}_stage1_best.pt")))

    # -------- Stage 2: Multitarea combinada --------
    loss_fn = MultiTaskLoss(focal_alpha=0.75, focal_gamma=3.5, delta=1.0,
                            pos_weight=pos_weight, label_smoothing=0.05)

    alpha, beta = 0.7, 0.3
    best_combo, wait, patience = -1e9, 0, 12
    mae_ref = np.mean(yva) + 1e-6
    thr_eval = 0.5  # se recalibra al final

    for ep in range(1, EPOCHS_STAGE2 + 1):
        model.train()
        losses, lclf, lreg = [], [], []
        for Xb, Yb, Jb in train_dl:
            optimizer.zero_grad()
            logit, yhat = model(Xb)
            loss, lc, lr = loss_fn(logit, Jb, yhat, Yb, alpha, beta)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item()); lclf.append(lc); lreg.append(lr)

        pack = eval_epoch(model, val_dl, q90, thresh=thr_eval)

        mae_clamped = min(pack["mae"], mae_ref * 3.0)
        mae_norm = mae_clamped / mae_ref

        # α dinámico: mezcla de error de regresión y F1
        alpha_new = 0.6 * (1 - mae_norm) + 0.4 * pack["f1"]
        alpha = 0.6 * alpha + 0.4 * np.clip(alpha_new, 0.4, 0.8)
        alpha = float(np.clip(alpha, 0.4, 0.8))
        beta = 1.0 - alpha

        combo = pack["f1"] + (1 - mae_norm)

        if combo > best_combo:
            best_combo, wait = combo, 0
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, f"{target_name}_best.pt"))
            with open(os.path.join(MODEL_DIR, f"{target_name}_scaler.pkl"), "wb") as f:
                pickle.dump(scaler, f)
            with open(os.path.join(MODEL_DIR, f"{target_name}_q90.json"), "w") as f:
                json.dump({"q90": float(q90)}, f)
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping (Stage 2)"); break

        if ep % 5 == 0 or ep == 1:
            print(f"[{target_name} S2 Ep{ep:03d}] "
                  f"train={np.mean(losses):.3f} | clf={np.mean(lclf):.3f} | reg={np.mean(lreg):.3f} | "
                  f"F1={pack['f1']:.3f} (P={pack['prec']:.2f},R={pack['rec']:.2f}) | "
                  f"MAE={pack['mae']:.2f} | α={alpha:.2f} β={beta:.2f} | thr_eval={thr_eval:.2f}")

    # Cargar mejor modelo y calibrar threshold en validación
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, f"{target_name}_best.pt")))
    tmp = eval_epoch(model, val_dl, q90, thresh=0.5)
    best_thr, best_f1 = calibrate_threshold(tmp["p"], tmp["y_cls"], grid=np.linspace(0.1, 0.5, 9))
    pack = eval_epoch(model, val_dl, q90, thresh=best_thr)

    print(f"\n🔧 Threshold óptimo (valid): {best_thr:.2f} → F1={best_f1:.3f}")
    print("\n📋 Clasificador (validation set) @thr_opt:")
    print(classification_report(pack["y_cls"], pack["y_hat_cls"], target_names=["Normal","Pico"], zero_division=0))
    print(f"✅ {target_name}: MAE={pack['mae']:.3f} | F1_jump={pack['f1']:.3f}")

    preds = pd.DataFrame({
        "y_true": pack["y_true"], "y_pred": pack["y_pred"],
        "is_jump_true": pack["y_cls"], "is_jump_pred": pack["y_hat_cls"], "p_jump": pack["p"]
    })
    preds.to_csv(os.path.join(MODEL_DIR, f"preds_{target_name}.csv"), index=False)

    # Guardar metadatos de calibración
    with open(os.path.join(MODEL_DIR, f"{target_name}_calibration.json"), "w") as fh:
        json.dump({"threshold_opt": best_thr, "f1_opt": best_f1, "q90": float(q90)}, fh, indent=2)

    return {"mae": pack["mae"], "f1": pack["f1"], "thr": best_thr}

# ============================== PIPELINE ==============================
def run_pipeline_two_stage():
    print("🚀 Iniciando pipeline Ualá — Multitask FFN v4.5 (Two-Stage + Oversampling x3 + Calibración)")
    # Cargar dataset simple (misma lógica que usabas en simple)
    files = glob.glob(os.path.join(DATA_DIR, "posts_*uala*.csv"))
    if not files:
        raise FileNotFoundError("No se encontraron archivos posts_*uala*.csv en core/data/")
    df = pd.read_csv(files[0])
    df["created_at"] = pd.to_datetime(df["created_at"])
    print(f"✅ Registros Ualá: {len(df)}")

    # ===== Features simples =====
    df["hour"] = df["created_at"].dt.hour
    df["weekday"] = df["created_at"].dt.weekday
    df["is_weekend"] = (df["weekday"] >= 5).astype(int)
    df["hour_sin"] = np.sin(2*np.pi*df["hour"]/24)
    df["hour_cos"] = np.cos(2*np.pi*df["hour"]/24)
    df["text_length"] = df["text"].astype(str).str.len()
    df["has_hashtag"] = df["text"].astype(str).str.contains("#", na=False).astype(int)
    df["has_mention"] = df["text"].astype(str).str.contains("@", na=False).astype(int)

    # Sociales mínimos
    top10 = pd.read_csv(os.path.join(DATA_DIR, "posts_top10.csv"))
    top10["created_at"] = pd.to_datetime(top10["created_at"])
    top10["date"] = top10["created_at"].dt.date
    df["date"]= df["created_at"].dt.date
    df = generar_features_sociales(df_user=df, df_top10=top10)

    # Jump intensity basado principalmente en VIEWS (viralización)
    # Likes/replies tienen menos peso (son engagement, no viralización)
    df["jump_intensity"] = (
        0.7 * df["views"]/(1e-3+df["views"].mean()) +
        0.15 * df["likes"]/(1e-3+df["likes"].mean()) +
        0.15 * df["replies"]/(1e-3+df["replies"].mean())
    )
    df["jump_intensity"] = np.log1p(df["jump_intensity"])
    df["jump_intensity"] = (df["jump_intensity"] - df["jump_intensity"].mean()) / (df["jump_intensity"].std() + 1e-6)

    # Etiqueta is_jump — SOLO VIEWS determinan viralización (likes/replies son engagement)
    # Usamos percentil 70 (top 30% de posts) para ser más sensible
    views_p70 = df["views"].quantile(0.70)
    df["is_jump"] = (df["views"] > views_p70).astype(int)

    pico_pct = df["is_jump"].mean()*100
    print(f"🚀 Posts virales detectados (views > p70={views_p70:.0f}): {df['is_jump'].sum()} ({pico_pct:.1f}%)")

    feat_cols = [
        "hour","weekday","is_weekend","hour_sin","hour_cos",
        "text_length","has_hashtag","has_mention",
        "followers_rel","engagement_rel"  # de generar_features_sociales
    ]
    feat_cols = [c for c in feat_cols if c in df.columns]
    print(f"📊 Features detectadas: {len(feat_cols)}")

    # Split temporal
    cutoff_date = df["created_at"].max() - pd.Timedelta(days=VAL_WINDOW_DAYS + FUTURE_GAP_DAYS)

    results = {}
    for t in TARGETS:
        results[t] = train_two_stage_target(t, df, feat_cols, cutoff_date)

    print("\n" + "="*60)
    print("📊 RESULTADOS FINALES (Two-Stage + Calibración)")
    print("="*60)
    for tgt, res in results.items():
        print(f"{tgt:8s}: MAE={res['mae']:.3f} | F1_jump={res['f1']:.3f} | thr_opt={res['thr']:.2f}")
    print("="*60)

if __name__ == "__main__":
    run_pipeline_two_stage()
