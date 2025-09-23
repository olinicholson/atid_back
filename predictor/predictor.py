import pandas as pd
import numpy as np
import os, joblib, json
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score, explained_variance_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
from xgboost import XGBRegressor
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from textblob import TextBlob

# ------------------------
# 1. Cargar datos clusterizados
# ------------------------
def load_clustered_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    clustered_file = os.path.join(os.path.dirname(current_dir), "posts_with_clusters.csv")
    if not os.path.exists(clustered_file):
        raise FileNotFoundError(f"No se encontró {clustered_file}. Ejecuta primero cluster_posts.py")
    df = pd.read_csv(clustered_file)
    print(f"Datos cargados: {len(df)} posts con clusters")
    df = df.rename(columns={
        "retweets": "Retweets",
        "likes": "Likes",
        "replies": "Reply_count",
        "views": "View_count"
    })
    return df

# ------------------------
# 2. Feature engineering
# ------------------------

def feature_engineering(df):
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    df["year"] = df["created_at"].dt.year
    df["month"] = df["created_at"].dt.month
    df["day"] = df["created_at"].dt.day
    df["hour"] = df["created_at"].dt.hour
    df["weekday"] = df["created_at"].dt.weekday
    df["is_weekend"] = (df["weekday"] >= 5).astype(int)
    df["is_golden_hour"] = df["hour"].isin([8,9,10,18,19,20]).astype(int)
    df["text_length"] = df["text"].astype(str).str.len()

    # Keywords
    offer_keywords = ['descuento','oferta','promoción','gratis','regalo','%','promo']
    trend_keywords = ['trending','viral','#','nuevo','lanzamiento']
    df["has_offer"] = df["text"].astype(str).str.lower().str.contains("|".join(offer_keywords), na=False).astype(int)
    df["has_trend"] = df["text"].astype(str).str.lower().str.contains("|".join(trend_keywords), na=False).astype(int)
    df["has_image"] = df["text"].astype(str).str.contains("pic.twitter.com|instagram.com|imgur.com", na=False).astype(int)

    # Followers mapping
    followers_mapping = {
        'uala': 190500,
        'naranjax': 190300,
        'balanz': 38000,
        'brubank': 47900,
        'cocos': 86500,
        'top10': 100000
    }
    df['followers_count'] = df['company'].map(followers_mapping).fillna(50000)
    df['followers_log'] = np.log1p(df['followers_count'])
    df['high_followers'] = (df['followers_count'] > 100000).astype(int)

    # Engagement proxy
    df["likes_over_rt"] = df["Likes"] / (df["Retweets"]+1)
    df["engagement_rate"] = ((df["Retweets"] + df["Likes"] + df["Reply_count"]) / df["followers_count"]).fillna(0)

    # 🔹 Nueva feature: sentimiento del texto
    def get_sentiment(text):
        try:
            blob = TextBlob(str(text))
            return blob.sentiment.polarity
        except:
            return 0
    df["sentiment_polarity"] = df["text"].apply(get_sentiment)
    df["is_payday"] = df["day"].isin([28,29,30,31,1,2,3,4,5]).astype(int)


    # Agregar fecha como columna separada para los merges
    df["date_only"] = df["created_at"].dt.date
    
    posts_per_day_company = df.groupby(["company", "date_only"])["text"].count().reset_index(name="company_posts_day")
    posts_per_day_total = df.groupby("date_only")["text"].count().reset_index(name="total_posts_day")

    df = df.merge(posts_per_day_company, how="left", on=["company", "date_only"])
    df = df.merge(posts_per_day_total, how="left", on="date_only")

    df["competencia_same_day"] = (df["total_posts_day"] - df["company_posts_day"]).fillna(0)

    # Limpiar columnas temporales
    df = df.drop(columns=["date_only"], errors="ignore")

    # Temporal features
    df = df.sort_values("created_at")
    df["days_since_last_post"] = df.groupby("company")["created_at"].diff().dt.days.fillna(0)

    def count_posts_last_7d(group):
        group_sorted = group.sort_values("created_at")
        posts_7d = []
        for i, current_date in enumerate(group_sorted["created_at"]):
            cutoff_date = current_date - pd.Timedelta(days=7)
            count = sum(group_sorted["created_at"][:i+1] >= cutoff_date)
            posts_7d.append(count)
        return pd.Series(posts_7d, index=group_sorted.index)

    df["posts_last_7d"] = df.groupby("company").apply(count_posts_last_7d).values

    if "content_cluster" in df.columns:
        df = pd.concat([df, pd.get_dummies(df["content_cluster"], prefix="Cluster")], axis=1)

    return df.fillna(0)

# ------------------------
# 3. Contexto dinámico de Top10
# ------------------------
def add_top10_dynamic_context(df, window_days=30):
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce", utc=True)
    df = df.dropna(subset=["created_at"])
    df["created_at"] = df["created_at"].dt.tz_convert(None)
    df = df.sort_values("created_at").reset_index(drop=True)

    top10 = df[df["company"] == "top10"].copy().sort_values("created_at")
    top10["top10_rt_roll"] = top10["Retweets"].rolling(window=window_days, min_periods=1).mean()

    for cluster_id in sorted(df["content_cluster"].dropna().unique()):
        mask = top10["content_cluster"] == cluster_id
        col = f"top10_cluster_{cluster_id}_roll"
        top10.loc[mask, col] = top10.loc[mask, "Retweets"].rolling(window=window_days, min_periods=1).mean()

    context_cols = ["top10_rt_roll"] + [c for c in top10.columns if "top10_cluster_" in c]
    df = pd.merge_asof(
        df.sort_values("created_at"),
        top10[["created_at"] + context_cols].sort_values("created_at"),
        on="created_at", direction="backward"
    )
    return df.fillna(0), context_cols

# ------------------------
# 4. Embeddings
# ------------------------
def add_embeddings(df, model_name="all-mpnet-base-v2", n_components=5):
    print(f"🔎 Generando embeddings con {model_name}...")
    model = SentenceTransformer(model_name)
    embeddings = model.encode(df["text"].astype(str).tolist(), show_progress_bar=True, batch_size=64)

    pca = PCA(n_components=n_components, random_state=42)
    embeddings_pca = pca.fit_transform(embeddings)
    joblib.dump(pca, "pca_embeddings.pkl")

    for i in range(n_components):
        df[f"text_emb_{i}"] = embeddings_pca[:, i]
    return df, [f"text_emb_{i}" for i in range(n_components)]

# ------------------------
# 5. Validación temporal
# ------------------------
def temporal_split(df):
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    max_date = df["created_at"].max()
    cutoff = max_date - pd.Timedelta(days=30)
    train_df = df[df["created_at"] < cutoff]
    test_df = df[df["created_at"] >= cutoff]
    print(f"Entrenamiento: {len(train_df)} filas (< {cutoff.date()})")
    print(f"Test: {len(test_df)} filas (>= {cutoff.date()})")
    return train_df, test_df

# ------------------------
# MAIN
# ------------------------
tweets_df = feature_engineering(load_clustered_data())

# ------------------------
# Config de targets
# ------------------------
targets_config = {
    "Retweets": {"clip": 10, "log": True, "smooth": False},
    "Likes": {"clip": 8, "log": True, "smooth": True},
    "View_count": {"clip": 0.5, "log": True, "smooth": True, "scale_followers": True},
    "Reply_count": {"clip": 5, "log": False, "smooth": True},
}

# ------------------------
# Normalizar y transformar
# ------------------------
for col, cfg in targets_config.items():
    if col == "View_count" and cfg.get("scale_followers", False):
        # Normalizar por seguidores → "reach rate"
        tweets_df[f"{col}_norm"] = tweets_df[col] / tweets_df["followers_count"]
    else:
        avg_per_company = tweets_df.groupby("company")[col].mean().to_dict()
        tweets_df[f"{col}_norm"] = tweets_df.apply(
            lambda row: row[col] / avg_per_company[row["company"]] if avg_per_company[row["company"]] > 0 else 0,
            axis=1
        )
    
    tweets_df[f"{col}_norm_clipped"] = tweets_df[f"{col}_norm"].clip(upper=cfg["clip"])
    
    if cfg["smooth"]:
        tweets_df[f"{col}_norm_clipped"] = tweets_df[f"{col}_norm_clipped"].rolling(3, min_periods=1).mean()

# ------------------------
# Targets (transformados según config)
# ------------------------
Y_cols = []
for col, cfg in targets_config.items():
    target_col = f"{col}_norm_clipped"
    if cfg["log"]:
        tweets_df[f"{target_col}_transformed"] = np.log1p(tweets_df[target_col])
    else:
        tweets_df[f"{target_col}_transformed"] = tweets_df[target_col]
    Y_cols.append(f"{target_col}_transformed")

Y = tweets_df[Y_cols]

# Contexto + embeddings
tweets_df, context_cols = add_top10_dynamic_context(tweets_df, window_days=7)
tweets_df, embedding_cols = add_embeddings(tweets_df, n_components=5)

# Features
cluster_cols = [c for c in tweets_df.columns if c.startswith("Cluster_")]
features = [
    "has_offer","text_length","day","month","year","hour",
    "is_golden_hour","has_trend","has_image","is_weekend","weekday",
    "followers_count","followers_log","high_followers"
] + cluster_cols + embedding_cols + context_cols

# ------------------------
# MODELOS HÍBRIDOS
# ------------------------
train_df, test_df = temporal_split(tweets_df)
X_train, X_test = train_df[features], test_df[features]

# A. Retweets + Likes (multi-output)
Y_train_rt_likes = np.log1p(train_df[["Retweets_norm_clipped","Likes_norm_clipped"]])
Y_test_rt_likes = np.log1p(test_df[["Retweets_norm_clipped","Likes_norm_clipped"]])
model_rt_likes = MultiOutputRegressor(
    XGBRegressor(n_estimators=1000, learning_rate=0.05, max_depth=10,
                 subsample=0.8, colsample_bytree=0.8, random_state=42)
).fit(X_train, Y_train_rt_likes)

# B. View_count
Y_train_view = np.log1p(train_df["View_count_norm_clipped"])
Y_test_view = np.log1p(test_df["View_count_norm_clipped"])
model_view = HistGradientBoostingRegressor(max_iter=500, random_state=42).fit(X_train, Y_train_view)

# C. Reply_count
y_train_reply_bin = (train_df["Reply_count_norm_clipped"] > 0).astype(int)
y_test_reply_bin = (test_df["Reply_count_norm_clipped"] > 0).astype(int)
clf_reply = LogisticRegression(max_iter=500).fit(X_train, y_train_reply_bin)

mask_train_reply = y_train_reply_bin == 1
model_reply_reg = XGBRegressor(n_estimators=600, learning_rate=0.05, max_depth=6,
                               subsample=0.8, colsample_bytree=0.8, random_state=42)
if mask_train_reply.sum() > 0:
    model_reply_reg.fit(X_train[mask_train_reply],
                        np.log1p(train_df.loc[mask_train_reply,"Reply_count_norm_clipped"]))

# ------------------------
# Evaluación
# ------------------------
print("\n📊 Evaluación híbrida:")

# Retweets + Likes
Y_pred_rt_likes = model_rt_likes.predict(X_test)
for i, col in enumerate(["Retweets","Likes"]):
    mse = mean_squared_error(Y_test_rt_likes.iloc[:, i], Y_pred_rt_likes[:, i])
    rmse = np.sqrt(mse)
    r2 = r2_score(Y_test_rt_likes.iloc[:, i], Y_pred_rt_likes[:, i])
    print(f"{col}: MSE={mse:.2f} RMSE={rmse:.2f} R²={r2:.2f}")

# View_count
Y_pred_view = model_view.predict(X_test)
print(f"View_count: R²={r2_score(Y_test_view, Y_pred_view):.2f}")

# Reply_count
reply_preds_bin = clf_reply.predict(X_test)
print(f"Reply_count clasificación: ACC={accuracy_score(y_test_reply_bin, reply_preds_bin):.2f}, "
      f"F1={f1_score(y_test_reply_bin, reply_preds_bin):.2f}")
mask_test_reply = y_test_reply_bin == 1
if mask_test_reply.sum() > 0:
    reply_preds_reg = model_reply_reg.predict(X_test[mask_test_reply])
    print(f"Reply_count regresión (positivos): R²={r2_score(np.log1p(test_df.loc[mask_test_reply,'Reply_count_norm_clipped']), reply_preds_reg):.2f}")

# ------------------------
# ANÁLISIS DE IMPORTANCIA DE VARIABLES
# ------------------------
print("\n🔍 ANÁLISIS DE IMPORTANCIA DE VARIABLES")
print("="*60)

from sklearn.inspection import permutation_importance
from sklearn.metrics import explained_variance_score

def analyze_feature_importance(model, X_test, y_test, model_name, features_list):
    print(f"\n📈 {model_name}")
    print("-" * 40)
    
    # 1. Importancia intrínseca del modelo (si está disponible)
    if hasattr(model, 'feature_importances_'):
        intrinsic_importance = pd.DataFrame({
            'feature': features_list,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("🏆 Top 10 características más importantes (intrínseca):")
        for i, row in intrinsic_importance.head(10).iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f} ({row['importance']/intrinsic_importance['importance'].sum()*100:.1f}%)")
    
    # 2. Importancia por permutación
    try:
        perm_importance = permutation_importance(model, X_test, y_test, n_repeats=5, random_state=42, n_jobs=-1)
        perm_df = pd.DataFrame({
            'feature': features_list,
            'importance_mean': perm_importance.importances_mean,
            'importance_std': perm_importance.importances_std
        }).sort_values('importance_mean', ascending=False)
        
        print("\n🎯 Top 10 características por permutación:")
        total_perm_importance = perm_df['importance_mean'].sum()
        for i, row in perm_df.head(10).iterrows():
            percentage = row['importance_mean']/total_perm_importance*100 if total_perm_importance > 0 else 0
            print(f"  {row['feature']}: {row['importance_mean']:.4f} ± {row['importance_std']:.4f} ({percentage:.1f}%)")
            
        return perm_df
    except Exception as e:
        print(f"  Error en análisis de permutación: {e}")
        return None

# Analizar cada modelo
print("\n🧠 MODELOS INDIVIDUALES:")

# 1. Modelo de Retweets + Likes (tomamos el primer estimador para Retweets)
if hasattr(model_rt_likes, 'estimators_'):
    rt_model = model_rt_likes.estimators_[0]  # Retweets
    rt_importance = analyze_feature_importance(
        rt_model, X_test, Y_test_rt_likes.iloc[:, 0], 
        "RETWEETS", features
    )

# 2. Modelo de Views
view_importance = analyze_feature_importance(
    model_view, X_test, Y_test_view,
    "VIEW COUNT", features
)

# 3. Modelo de Reply Classification
reply_clf_importance = analyze_feature_importance(
    clf_reply, X_test, y_test_reply_bin,
    "REPLY COUNT (Clasificación)", features
)

# ------------------------
# ANÁLISIS COMPARATIVO DE VARIABLES
# ------------------------
print("\n🔄 ANÁLISIS COMPARATIVO DE VARIABLES")
print("="*60)

# Agrupar características por categorías
feature_categories = {
    'Temporales': [f for f in features if any(x in f.lower() for x in ['hour', 'day', 'month', 'year', 'weekend', 'golden'])],
    'Texto': [f for f in features if any(x in f.lower() for x in ['length', 'offer', 'trend', 'image', 'sentiment'])],
    'Empresas': [f for f in features if f.startswith('Company_')],
    'Clusters': [f for f in features if f.startswith('Cluster_')],
    'Embeddings': [f for f in features if f.startswith('embed_')],
    'Contexto Social': [f for f in features if any(x in f.lower() for x in ['followers', 'engagement', 'posts_last', 'days_since'])],
    'Normalización': [f for f in features if any(x in f.lower() for x in ['norm', 'scaled'])]
}

# Mostrar importancia por categorías
if rt_importance is not None:
    print("\n📊 IMPORTANCIA POR CATEGORÍAS (Retweets):")
    for category, cat_features in feature_categories.items():
        cat_features_in_model = [f for f in cat_features if f in rt_importance['feature'].values]
        if cat_features_in_model:
            total_importance = rt_importance[rt_importance['feature'].isin(cat_features_in_model)]['importance_mean'].sum()
            percentage = total_importance / rt_importance['importance_mean'].sum() * 100
            print(f"  {category}: {percentage:.1f}% de la importancia total")
            
            # Mostrar top 3 en cada categoría
            top_in_category = rt_importance[rt_importance['feature'].isin(cat_features_in_model)].head(3)
            for _, row in top_in_category.iterrows():
                print(f"    - {row['feature']}: {row['importance_mean']:.4f}")


# ------------------------
# Guardar modelos
# ------------------------
joblib.dump({
    "rt_likes": model_rt_likes,
    "view": model_view,
    "reply_clf": clf_reply,
    "reply_reg": model_reply_reg,
    "features": features
}, "hybrid_models.pkl")
