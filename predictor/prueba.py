import pandas as pd
import numpy as np
import joblib

# Cargar los modelos desde archivos PKL
view_count_model = joblib.load("xgboost_View_count_predictor.pkl")
likes_model = joblib.load("xgboost_reply_count_predictor.pkl")
quote_count_model = joblib.load("xgboost_likes_predictor.pkl")
retweets_model = joblib.load("xgboost_Retweets_predictor.pkl")
reply_count_model = joblib.load("xgboost_reply_count_predictor.pkl")
def preprocess_observation(observation):
    # Crear columna 'weekday' a partir de 'day', 'month', 'year'
    date = pd.to_datetime({
        'year': [observation['year']],
        'month': [observation['month']],
        'day': [observation['day']]
    })
    observation['weekday'] = date.dt.weekday[0]

    # Crear indicador 'is_weekend'
    observation['is_weekend'] = 1 if observation['weekday'] >= 5 else 0

    # Convertir columnas booleanas a enteros
    bool_columns = ["has_trend", "Username_Buenbit", "Username_Lemon Argentina", 
                    "Username_Naranja X", "Username_Ualá Argentina", "Username_belo 🌎"]
    for col in bool_columns:
        observation[col] = int(observation[col])

    # Seleccionar las características relevantes
    features = ["has_offer", "text_length", "day", "month", "year", "is_golden_hour", 
                "has_trend", "has_image", "Username_Buenbit", "Username_Lemon Argentina", 
                "Username_Naranja X", "Username_Ualá Argentina", "Username_belo 🌎", 
                "is_weekend", "weekday"]

    # Convertir a DataFrame
    return pd.DataFrame([observation])[features]

# Datos de entrada
new_observation = {
    "has_offer": 1,
    "text_length": 264,
    "day": 25,
    "month": 12,
    "year": 2024,
    "is_golden_hour": 1,
    "has_trend": True,
    "has_image": 1,
    "Username_Buenbit": True,
    "Username_Lemon Argentina": False,
    "Username_Naranja X": False,
    "Username_Ualá Argentina": False,
    "Username_belo 🌎": False
}
# Convertir los datos a un DataFrame (formato requerido por los modelos)

# Preprocesar la observación
processed_observation = preprocess_observation(new_observation)
observation_df = processed_observation

# Realizar predicciones ajustando valores a 0 si son negativos
predictions = {
    "ViewCount": np.maximum(view_count_model.predict(observation_df)[0], 0),
    "Likes": np.maximum(likes_model.predict(observation_df)[0], 0),
    "QuoteCount": np.maximum(quote_count_model.predict(observation_df)[0], 0),
    "Retweets": np.maximum(retweets_model.predict(observation_df)[0], 0),
    "ReplyCount": np.maximum(reply_count_model.predict(observation_df)[0], 0)
}

# Mostrar los resultados
for metric, value in predictions.items():
    print(f"{metric}: {value:.2f}")
