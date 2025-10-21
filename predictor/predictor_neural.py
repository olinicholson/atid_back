"""
Predictor de engagement (PyTorch)
Arquitectura híbrida: BERT-es + MLP tabular
Soporta fine-tuning parcial y mean pooling
"""

import os
import glob
import math
import json
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# =========================
# Configuración global
# =========================
MODEL_NAME = "dccuchile/bert-base-spanish-wwm-cased"
USE_MEAN_POOLING = True         # True: mean pooling; False: [CLS]
UNFREEZE_EPOCH = 3              # Época en la que se descongela parte del BERT
UNFREEZE_LAST_N_LAYERS = 2      # Cantidad de capas a descongelar (empieza desde el final)
HEAD_LR = 2e-4                  # LR para capas tabulares/combiner/heads
TRANSFORMER_LR = 2e-5           # LR para el transformer cuando se descongela
WEIGHT_DECAY = 0.01
PATIENCE = 5
MAX_EPOCHS = 15                 # menos épocas pero con fine-tuning parcial
TEST_DAYS = 30
MAX_LENGTH = 128

# Dispositivo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 Usando dispositivo: {device}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   CUDA Version: {torch.version.cuda}")

# Batch size adaptativo
BATCH_SIZE = 64 if torch.cuda.is_available() else 16
NUM_WORKERS = 0  # en Windows mantener 0

# =========================
# 1. CARGA Y PREPARACIÓN DE DATOS
# =========================
def load_data_from_trends():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(current_dir)
    data_dir = os.path.join(repo_root, 'core', 'data')
    patterns = [
        os.path.join(data_dir, 'posts_*with_trends*.csv'),
        os.path.join(data_dir, 'posts_*_with_trends*.csv')
    ]
    files = []
    for p in patterns:
        files.extend(glob.glob(p))
    files = sorted(set(files))
    if not files:
        raise FileNotFoundError("No se encontraron archivos *with_trends*.csv en core/data")

    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            if 'company' not in df.columns and 'username' in df.columns:
                df['company'] = df['username'].str.lower().str.replace(r'[^a-z]', '', regex=True)
            dfs.append(df)
        except Exception as e:
            print(f"⚠️  Error leyendo {f}: {e}")
    df = pd.concat(dfs, ignore_index=True)
    print(f"✅ Datos cargados: {len(df)} posts desde {len(dfs)} archivos")

    col_map = {'likes':'Likes','retweets':'Retweets','replies':'Reply_count','views':'View_count'}
    df = df.rename(columns=col_map)
    return df

def feature_engineering(df):
    df = df.copy()
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    df = df.dropna(subset=["created_at"])

    # temporales
    df["year"] = df["created_at"].dt.year
    df["month"] = df["created_at"].dt.month
    df["day"] = df["created_at"].dt.day
    df["hour"] = df["created_at"].dt.hour
    df["weekday"] = df["created_at"].dt.weekday
    df["is_weekend"] = (df["weekday"] >= 5).astype(int)
    df["is_golden_hour"] = df["hour"].isin([8,9,10,18,19,20]).astype(int)

    # texto
    df["text_length"] = df["text"].astype(str).str.len()
    offer_keywords = ['descuento','oferta','promoción','gratis','regalo','%','promo']
    df["has_offer"] = df["text"].astype(str).str.lower().str.contains("|".join(offer_keywords), na=False).astype(int)
    df["has_image"] = df["text"].astype(str).str.contains("pic.twitter.com|instagram.com|imgur.com|https://t.co", na=False).astype(int)

    # trends
    if 'trend_similarity' in df.columns:
        df['trend_similarity'] = pd.to_numeric(df['trend_similarity'], errors='coerce').fillna(0.0)
        df['has_trend_match'] = (df['trend_similarity'] > 0).astype(int)
    else:
        df['has_trend_match'] = 0

    followers_mapping = {
        'uala': 190500, 'naranjax': 190300, 'balanz': 38000, 'brubank': 47900,
        'cocos': 86500, 'top10': 100000, 'supervielle': 50000, 'galicia': 80000
    }
    df['followers_count'] = df['company'].map(followers_mapping).fillna(50000)
    df['followers_log'] = np.log1p(df['followers_count'])

    for col in ['Retweets','Likes','Reply_count','View_count']:
        if col not in df.columns:
            df[col] = 0
        avg_per_company = df.groupby("company")[col].mean().to_dict()
        df[f"{col}_norm"] = df.apply(
            lambda row: row[col] / avg_per_company.get(row["company"], 1) if avg_per_company.get(row["company"], 0) > 0 else 0,
            axis=1
        )
        df[f"{col}_norm"] = df[f"{col}_norm"].clip(upper=df[f"{col}_norm"].quantile(0.95))

    return df.fillna(0)

def temporal_split(df, test_days=TEST_DAYS):
    df = df.sort_values("created_at").reset_index(drop=True)
    max_date = df["created_at"].max()
    cutoff = max_date - pd.Timedelta(days=test_days)
    train_df = df[df["created_at"] < cutoff].reset_index(drop=True)
    test_df  = df[df["created_at"] >= cutoff].reset_index(drop=True)
    print("📊 Split temporal:")
    print(f"   Train: {len(train_df)} filas (< {cutoff.date()})")
    print(f"   Test:  {len(test_df)} filas (>= {cutoff.date()})")
    return train_df, test_df

# =========================
# 2. DATASET
# =========================
class PostDataset(Dataset):
    def __init__(self, df, tokenizer, tabular_features, target_cols, max_length=MAX_LENGTH):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.tabular_features = tabular_features
        self.target_cols = target_cols
        self.max_length = max_length
        self.X_tabular = df[tabular_features].values.astype(np.float32)
        self.y = df[target_cols].values.astype(np.float32)

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        text = str(self.df.at[idx, 'text'])
        encoding = self.tokenizer(
            text, max_length=self.max_length, padding='max_length',
            truncation=True, return_tensors='pt'
        )
        return {
            'input_ids':      encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'tabular':        torch.from_numpy(self.X_tabular[idx]),
            'targets':        torch.from_numpy(self.y[idx])
        }

# =========================
# 3. MODELO
# =========================
class HybridEngagementModel(nn.Module):
    def __init__(self, num_tabular_features, num_targets, dropout=0.3):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(MODEL_NAME)
        self.hidden = self.transformer.config.hidden_size  # 768 o 768/1024 según modelo

        # congelar todo al inicio
        for p in self.transformer.parameters():
            p.requires_grad = False

        self.tabular_mlp = nn.Sequential(
            nn.Linear(num_tabular_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        combined_dim = self.hidden + 64
        self.combiner = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(128, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            ) for _ in range(num_targets)
        ])

    def forward(self, input_ids, attention_mask, tabular):
        out = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        if USE_MEAN_POOLING:
            # mean pooling sobre tokens válidos
            last_hidden = out.last_hidden_state  # [B, T, H]
            mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
            sum_embeddings = (last_hidden * mask).sum(1)
            lengths = mask.sum(1).clamp(min=1e-9)
            text_embedding = sum_embeddings / lengths
        else:
            # usar [CLS]
            text_embedding = out.last_hidden_state[:, 0, :]

        tabular_embedding = self.tabular_mlp(tabular)
        combined = torch.cat([text_embedding, tabular_embedding], dim=1)
        combined = self.combiner(combined)
        outputs = torch.cat([head(combined) for head in self.heads], dim=1)
        return outputs

    def unfreeze_last_n_layers(self, n=2):
        # descongela las últimas n capas encoder.layer.(L-n ... L-1)
        if not hasattr(self.transformer, "encoder") or not hasattr(self.transformer.encoder, "layer"):
            # Modelos como RoBERTa/BERT: .encoder.layer existe
            for p in self.transformer.parameters():
                p.requires_grad = True
            return

        total = len(self.transformer.encoder.layer)
        keep_frozen_until = max(0, total - n)
        for i, layer in enumerate(self.transformer.encoder.layer):
            requires = (i >= keep_frozen_until)
            for p in layer.parameters():
                p.requires_grad = requires

# =========================
# 4. TRAIN / EVAL
# =========================
def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for batch in tqdm(dataloader, desc="Training", leave=False):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        tabular = batch['tabular'].to(device)
        targets = batch['targets'].to(device)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(input_ids, attention_mask, tabular)
        loss = criterion(outputs, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    preds, targs = [], []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            tabular = batch['tabular'].to(device)
            targets = batch['targets'].to(device)

            outputs = model(input_ids, attention_mask, tabular)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            preds.append(outputs.cpu().numpy())
            targs.append(targets.cpu().numpy())
    preds = np.vstack(preds)
    targs = np.vstack(targs)
    return total_loss / len(dataloader), preds, targs

# =========================
# 5. MAIN
# =========================
def main():
    # datos
    print("\n📁 Cargando datos...")
    df = load_data_from_trends()
    df = feature_engineering(df)

    train_df, test_df = temporal_split(df, test_days=TEST_DAYS)

    tabular_features = [
        'hour','day','month','weekday','is_weekend','is_golden_hour',
        'text_length','has_offer','has_image','has_trend_match',
        'followers_count','followers_log'
    ]
    target_cols = ['Retweets_norm','Likes_norm','Reply_count_norm','View_count_norm']

    scaler = StandardScaler()
    train_df[tabular_features] = scaler.fit_transform(train_df[tabular_features])
    test_df[tabular_features]  = scaler.transform(test_df[tabular_features])

    print("\n🔤 Cargando tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("\n📦 Creando datasets...")
    train_dataset = PostDataset(train_df, tokenizer, tabular_features, target_cols)
    test_dataset  = PostDataset(test_df,  tokenizer, tabular_features, target_cols)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    print("\n🔍 INFORMACIÓN DEL DATASET:")
    print(f"   📊 Total filas DataFrame original: {len(df)}")
    print(f"   📊 Filas train_df (después split): {len(train_df)}")
    print(f"   📊 Filas test_df (después split):  {len(test_df)}")
    print(f"   📊 len(train_dataset):             {len(train_dataset)}")
    print(f"   📊 len(test_dataset):              {len(test_dataset)}")
    print(f"   📦 BATCH_SIZE:                     {BATCH_SIZE}")
    print(f"   🔢 Batches por época (calculado):  {math.ceil(len(train_dataset)/BATCH_SIZE)}")
    print(f"   🔢 len(train_loader):              {len(train_loader)}")

    print("\n🧠 Inicializando modelo...")
    model = HybridEngagementModel(
        num_tabular_features=len(tabular_features),
        num_targets=len(target_cols),
        dropout=0.3
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Parámetros totales: {total_params:,}")
    print(f"   Parámetros entrenables: {trainable_params:,}")

    # Optimizer con param groups (por si luego descongelamos capas del transformer)
    # Inicialmente, solo HEADS/COMBINER/TABULAR tienen requires_grad=True
    head_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(
        [{"params": head_params, "lr": HEAD_LR}],
        weight_decay=WEIGHT_DECAY
    )
    criterion = nn.MSELoss()
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    # Entrenamiento
    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'val_r2_mean': []}

    print("\n🚀 Iniciando entrenamiento...")
    for epoch in range(1, MAX_EPOCHS+1):
        print(f"\n📈 Epoch {epoch}/{MAX_EPOCHS}")

        # Descongelar parcialmente a partir de cierta época
        if epoch == UNFREEZE_EPOCH:
            print(f"🔓 Descongelando últimas {UNFREEZE_LAST_N_LAYERS} capas del transformer...")
            model.unfreeze_last_n_layers(UNFREEZE_LAST_N_LAYERS)
            # Rehacer param groups: head (HEAD_LR) + transformer (TRANSFORMER_LR)
            head_params = []
            transformer_params = []
            for name, p in model.named_parameters():
                if p.requires_grad:
                    if name.startswith("transformer."):
                        transformer_params.append(p)
                    else:
                        head_params.append(p)
            optimizer = AdamW(
                [
                    {"params": head_params, "lr": HEAD_LR},
                    {"params": transformer_params, "lr": TRANSFORMER_LR}
                ],
                weight_decay=WEIGHT_DECAY
            )
            # resetear scheduler que depende del optimizer
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"   🔁 Nuevos parámetros entrenables: {trainable_params:,}")

        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_preds, val_targets = evaluate(model, test_loader, criterion, device)

        # R² promedio sobre los 4 targets (por monitoreo)
        r2s = []
        for i in range(val_preds.shape[1]):
            r2s.append(r2_score(val_targets[:, i], val_preds[:, i]))
        r2_mean = float(np.mean(r2s))

        scheduler.step(val_loss)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_r2_mean'].append(r2_mean)

        print(f"   Train Loss: {train_loss:.4f}")
        print(f"   Val   Loss: {val_loss:.4f}")
        print(f"   Val   R²μ : {r2_mean:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'scaler': scaler,  # ojo: guardamos el StandardScaler de features tabulares
                'tabular_features': tabular_features,
                'target_cols': target_cols,
                'model_name': MODEL_NAME,
                'use_mean_pooling': USE_MEAN_POOLING
            }, 'best_neural_model.pt')
            print("   ✅ Modelo guardado (nuevo mejor)")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\n⏹️  Early stopping (patience={PATIENCE})")
                break

    # Evaluación final con el mejor checkpoint
    print("\n📊 Evaluación final...")
    checkpoint = torch.load('best_neural_model.pt', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    _, test_preds, test_targets = evaluate(model, test_loader, criterion, device)

    print("\n" + "="*60)
    print("MÉTRICAS FINALES (Test Set)")
    print("="*60)

    metrics = {}
    names = ['Retweets','Likes','Reply_count','View_count']
    for i, name in enumerate(names):
        y_true = test_targets[:, i]
        y_pred = test_preds[:, i]
        mse = mean_squared_error(y_true, y_pred)
        rmse = float(np.sqrt(mse))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        metrics[name] = {'MSE': float(mse), 'RMSE': rmse, 'MAE': float(mae), 'R²': float(r2)}
        print(f"\n{name}:")
        print(f"  MSE:  {mse:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE:  {mae:.4f}")
        print(f"  R²:   {r2:.4f}")

    with open('neural_metrics.json', 'w') as f:
        json.dump({'metrics': metrics, 'history': history}, f, indent=2)

    print("\n✅ Entrenamiento completado")
    print("   Mejor modelo: best_neural_model.pt")
    print("   Métricas:     neural_metrics.json")

    return model, metrics, history

if __name__ == '__main__':
    main()
