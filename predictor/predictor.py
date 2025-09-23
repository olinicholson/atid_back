import pandas as pd
import numpy as np
import os
import glob
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
import joblib, json

# ------------------------
# 1. Cargar y combinar datos
# ------------------------
def load_and_combine_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(os.path.dirname(current_dir), "core", "data")
    
    csv_files = glob.glob(os.path.join(data_path, "*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No se encontraron CSV en {data_path}")
    
    all_data = []
    for file in csv_files:
        df = pd.read_csv(file)
        if df.empty:
            continue
        
        company = os.path.basename(file).replace("posts_", "").replace(".csv", "")
        df["company"] = company
        
        # Renombrar columnas
        df = df.rename(columns={
            "retweets": "Retweets",
            "likes": "Likes",
            "replies": "Reply_count",
            "views": "View_count"
        })
        all_data.append(df)
    
    return pd.concat(all_data, ignore_index=True)

# ------------------------
# 2. Feature Engineering
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
    
    offer_keywords = ['descuento','oferta','promoción','gratis','regalo','%','promo']
    trend_keywords = ['trending','viral','#','nuevo','lanzamiento']
    
    df["has_offer"] = df["text"].astype(str).str.lower().str.contains("|".join(offer_keywords), na=False).astype(int)
    df["has_trend"] = df["text"].astype(str).str.lower().str.contains("|".join(trend_keywords), na=False).astype(int)
    df["has_image"] = df["text"].astype(str).str.contains("pic.twitter.com|instagram.com|imgur.com", na=False).astype(int)
    
    df = pd.concat([df, pd.get_dummies(df["company"], prefix="Company")], axis=1)
    return df.fillna(0)

# ------------------------
# 3. Main
# ------------------------
tweets_df = feature_engineering(load_and_combine_data())
print(f"Tweets cargados: {len(tweets_df)} de {tweets_df['company'].nunique()} empresas")

target = "Retweets"
company_cols = [c for c in tweets_df.columns if c.startswith("Company_")]
features = ["has_offer","text_length","day","month","year","hour",
            "is_golden_hour","has_trend","has_image","is_weekend","weekday"] + company_cols

X = tweets_df[features]
y = tweets_df[target]

# Transformar target (log1p)
y_log = np.log1p(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_log, test_size=0.2, random_state=42)

# Modelo XGBoost
xgb_model = XGBRegressor(
    n_estimators=800,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
xgb_model.fit(X_train, y_train)

# Predicciones (revirtiendo log)
y_pred_log = xgb_model.predict(X_test)
y_pred = np.expm1(np.maximum(y_pred_log, 0))

mse = mean_squared_error(y_test, y_pred_log)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred_log)

print(f"MSE (log-space): {mse:.2f}")
print(f"RMSE (log-space): {rmse:.2f}")
print(f"R²: {r2:.2f}")

# Guardar modelo
joblib.dump(xgb_model, "xgboost_retweets_predictor.pkl")
with open("model_info.json", "w") as f:
    json.dump({"features": features, "companies": tweets_df["company"].unique().tolist()}, f, indent=2)

# Importancia
fi = pd.DataFrame({"Feature": features, "Importance": xgb_model.feature_importances_})
print(fi.sort_values("Importance", ascending=False))

# Ejemplo de predicción
i = 0
true_val = y.iloc[i]
pred_val = np.expm1(xgb_model.predict(X.iloc[[i]])[0])
print("\nEjemplo:")
print(dict(zip(features[:10], X.iloc[i][:10])))
print(f"Retweets reales: {true_val}, Predicción: {pred_val:.1f}")
