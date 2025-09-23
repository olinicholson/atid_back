import pandas as pd
import numpy as np
import os
import glob
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
from sklearn.model_selection import KFold, RandomizedSearchCV
from datetime import datetime
import re

# Cargar y combinar todos los datos de core/data
def load_and_combine_data():
    # Usar ruta absoluta desde la ubicación actual
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(os.path.dirname(current_dir), "core", "data")
    
    print(f"Buscando archivos CSV en: {data_path}")
    
    if not os.path.exists(data_path):
        print(f"Error: La ruta {data_path} no existe")
        return pd.DataFrame()
    
    csv_files = glob.glob(os.path.join(data_path, "*.csv"))
    print(f"Archivos CSV encontrados: {csv_files}")
    
    if not csv_files:
        print("No se encontraron archivos CSV")
        return pd.DataFrame()
    
    all_data = []
    
    for file in csv_files:
        print(f"Procesando archivo: {file}")
        try:
            # Leer cada CSV
            df = pd.read_csv(file)
            print(f"  - Filas cargadas: {len(df)}")
            print(f"  - Columnas: {list(df.columns)}")
            
            # Verificar que el DataFrame no esté vacío
            if len(df) == 0:
                print(f"  - Archivo vacío, saltando...")
                continue
            
            # Extraer el nombre de la empresa del archivo
            filename = os.path.basename(file)
            company_name = filename.replace('posts_', '').replace('.csv', '')
            df['company'] = company_name
            
            # Renombrar columnas para consistencia
            if 'retweets' in df.columns:
                df = df.rename(columns={'retweets': 'Retweets'})
            if 'likes' in df.columns:
                df = df.rename(columns={'likes': 'Likes'})
            if 'replies' in df.columns:
                df = df.rename(columns={'replies': 'Reply_count'})
            if 'views' in df.columns:
                df = df.rename(columns={'views': 'View_count'})
            
            all_data.append(df)
            print(f"  - Archivo procesado exitosamente")
            
        except Exception as e:
            print(f"Error procesando {file}: {e}")
    
    if not all_data:
        print("No se pudieron cargar datos válidos")
        return pd.DataFrame()
    
    # Combinar todos los DataFrames
    print(f"Combinando {len(all_data)} archivos...")
    tweets_df = pd.concat(all_data, ignore_index=True)
    return tweets_df

# Función para ingeniería de características
def feature_engineering(df):
    # Convertir created_at a datetime
    df['created_at'] = pd.to_datetime(df['created_at'])
    
    # Extraer componentes de fecha
    df['year'] = df['created_at'].dt.year
    df['month'] = df['created_at'].dt.month
    df['day'] = df['created_at'].dt.day
    df['hour'] = df['created_at'].dt.hour
    df['weekday'] = df['created_at'].dt.weekday
    
    # Crear indicadores de tiempo
    df['is_weekend'] = df['weekday'].apply(lambda x: 1 if x >= 5 else 0)
    df['is_golden_hour'] = df['hour'].apply(lambda x: 1 if x in [8, 9, 10, 18, 19, 20] else 0)
    
    # Características del texto
    df['text_length'] = df['text'].astype(str).str.len()
    
    # Detectar ofertas y tendencias (palabras clave básicas)
    offer_keywords = ['descuento', 'oferta', 'promoción', 'gratis', 'regalo', '%', 'promo']
    trend_keywords = ['trending', 'viral', '#', 'nuevo', 'lanzamiento']
    
    df['has_offer'] = df['text'].astype(str).str.lower().str.contains('|'.join(offer_keywords), na=False).astype(int)
    df['has_trend'] = df['text'].astype(str).str.lower().str.contains('|'.join(trend_keywords), na=False).astype(int)
    df['has_image'] = df['text'].astype(str).str.contains('pic.twitter.com|instagram.com|imgur.com', na=False).astype(int)
    
    # Variables dummy para empresas
    company_dummies = pd.get_dummies(df['company'], prefix='Company')
    df = pd.concat([df, company_dummies], axis=1)
    
    return df

# Cargar y procesar datos
tweets_df = load_and_combine_data()

# Verificar si se cargaron datos
if tweets_df.empty:
    print("No se pudieron cargar datos. Verificar archivos CSV en core/data/")
    exit(1)

tweets_df = feature_engineering(tweets_df)

# Preparar datos para el modelo
print(f"Total de tweets cargados: {len(tweets_df)}")
print(f"Empresas incluidas: {tweets_df['company'].unique()}")

# Mostrar una muestra de los datos
print("\nMuestra de datos:")
print(tweets_df.head())

# Preparar datos para el modelo
print(f"Total de tweets cargados: {len(tweets_df)}")
print(f"Empresas incluidas: {tweets_df['company'].unique()}")

# Separar características (X) y variable objetivo (y)
target = "Retweets"

# Asegurar que tenemos los datos necesarios
if target not in tweets_df.columns:
    print(f"Warning: {target} no encontrado en los datos")
    tweets_df[target] = tweets_df.get('retweets', 0)

# Llenar valores nulos
tweets_df = tweets_df.fillna(0)

# Obtener columnas de empresas dinámicamente
company_columns = [col for col in tweets_df.columns if col.startswith('Company_')]
print(f"Columnas de empresas: {company_columns}")

# Definir características
features = ["has_offer", "text_length", "day", "month", "year", "hour", 
           "is_golden_hour", "has_trend", "has_image", "is_weekend", "weekday"] + company_columns

# Verificar que todas las características existen
available_features = [f for f in features if f in tweets_df.columns]
missing_features = [f for f in features if f not in tweets_df.columns]

if missing_features:
    print(f"Características faltantes: {missing_features}")
    features = available_features

print(f"Características utilizadas: {features}")

X = tweets_df[features]
y = tweets_df[target]

# Eliminar filas con valores nulos en target
mask = ~y.isna()
X = X[mask]
y = y[mask]

print(f"Datos finales para entrenamiento: {len(X)} filas")

# Dividir en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Configurar el modelo XGBoost
xgb_model = XGBRegressor(
    n_estimators=1000,        # Número de árboles
    learning_rate=0.1,       # Tasa de aprendizaje
    max_depth=10,             # Profundidad máxima del árbol
    subsample=0.8,           # Submuestreo de datos
    colsample_bytree=0.8,    # Submuestreo de características
    random_state=42
)

# Entrenar el modelo
xgb_model.fit(X_train, y_train)

# Realizar predicciones
y_pred = xgb_model.predict(X_test)

# Ajustar las predicciones para que los valores menores a 0 sean 0
y_pred = np.maximum(y_pred, 0)

# Evaluar el modelo
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Imprimir métricas
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R2 Score: {r2:.2f}")

rmse = np.sqrt(mse)
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
# Guardar el modelo con nombre específico
import joblib
model_filename = f"xgboost_{target}_predictor_companies.pkl"
joblib.dump(xgb_model, model_filename)
print(f"Modelo guardado como: {model_filename}")

# Guardar también información sobre las características utilizadas
features_info = {
    'features': features,
    'companies': tweets_df['company'].unique().tolist(),
    'target': target,
    'model_performance': {
        'mse': mse,
        'r2': r2,
        'rmse': rmse
    }
}

import json
with open(f"model_info_{target}_companies.json", 'w') as f:
    json.dump(features_info, f, indent=2)

print(f"Información del modelo guardada en: model_info_{target}_companies.json")

# Verificar la importancia de las características
feature_importances = pd.DataFrame({
    "Feature": features,
    "Importance": xgb_model.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("\nImportancia de las características:")
print(feature_importances)

# Mostrar estadísticas por empresa
print("\nEstadísticas por empresa:")
company_stats = tweets_df.groupby('company').agg({
    'Retweets': ['count', 'mean', 'std'],
    'Likes': 'mean',
    'text_length': 'mean'
}).round(2)
print(company_stats)

# Ejemplo de predicción
if len(X_test) > 0:
    example_index = min(18, len(X_test) - 1)  # Asegurar que el índice existe
    example_X = X_test.iloc[example_index]
    example_y_true = y_test.iloc[example_index]
    example_y_pred = np.maximum(xgb_model.predict(example_X.values.reshape(1, -1))[0], 0)

    print(f"\nEjemplo de Predicción (índice {example_index}):")
    print("Características principales:")
    for feature in features[:10]:  # Mostrar solo las primeras 10 características
        if feature in example_X.index:
            print(f"  {feature}: {example_X[feature]}")
    print(f"Valor Verdadero (Y_true): {example_y_true}")
    print(f"Predicción (Y_pred): {example_y_pred:.2f}")
else:
    print("No hay datos de prueba disponibles para mostrar ejemplo.")
