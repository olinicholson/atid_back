import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime
import os

class CompaniesPredictor:
    def __init__(self, model_file="xgboost_Retweets_predictor_companies.pkl", 
                 info_file="model_info_Retweets_companies.json"):
        """
        Inicializar el predictor con el modelo entrenado
        """
        self.model = joblib.load(model_file)
        
        with open(info_file, 'r') as f:
            self.model_info = json.load(f)
        
        self.features = self.model_info['features']
        self.companies = self.model_info['companies']
        
        print(f"Modelo cargado exitosamente")
        print(f"Características: {len(self.features)}")
        print(f"Empresas disponibles: {self.companies}")
    
    def prepare_features(self, text, company, created_at=None):
        """
        Preparar características para un tweet individual
        """
        if created_at is None:
            created_at = datetime.now()
        elif isinstance(created_at, str):
            created_at = pd.to_datetime(created_at)
        
        # Características básicas
        features_dict = {
            'text_length': len(str(text)),
            'year': created_at.year,
            'month': created_at.month,
            'day': created_at.day,
            'hour': created_at.hour,
            'weekday': created_at.weekday(),
            'is_weekend': 1 if created_at.weekday() >= 5 else 0,
            'is_golden_hour': 1 if created_at.hour in [8, 9, 10, 18, 19, 20] else 0
        }
        
        # Detectar ofertas y tendencias
        text_lower = str(text).lower()
        offer_keywords = ['descuento', 'oferta', 'promoción', 'gratis', 'regalo', '%', 'promo']
        trend_keywords = ['trending', 'viral', '#', 'nuevo', 'lanzamiento']
        
        features_dict['has_offer'] = 1 if any(keyword in text_lower for keyword in offer_keywords) else 0
        features_dict['has_trend'] = 1 if any(keyword in text_lower for keyword in trend_keywords) else 0
        features_dict['has_image'] = 1 if any(img in str(text) for img in ['pic.twitter.com', 'instagram.com', 'imgur.com']) else 0
        
        # Variables dummy para empresas (todas en 0 excepto la especificada)
        for comp in self.companies:
            features_dict[f'Company_{comp}'] = 1 if comp == company else 0
        
        # Asegurar que tenemos todas las características necesarias
        for feature in self.features:
            if feature not in features_dict:
                features_dict[feature] = 0
        
        # Crear DataFrame con las características en el orden correcto
        feature_values = [features_dict.get(feature, 0) for feature in self.features]
        return np.array(feature_values).reshape(1, -1)
    
    def predict(self, text, company, created_at=None):
        """
        Predecir retweets para un tweet
        """
        if company not in self.companies:
            print(f"Warning: {company} no está en las empresas conocidas: {self.companies}")
        
        features = self.prepare_features(text, company, created_at)
        prediction = self.model.predict(features)[0]
        
        # Asegurar que la predicción no sea negativa
        prediction = max(0, prediction)
        
        return round(prediction, 2)
    
    def predict_batch(self, tweets_df):
        """
        Predecir para un DataFrame de tweets
        Espera columnas: text, company, created_at (opcional)
        """
        predictions = []
        
        for idx, row in tweets_df.iterrows():
            created_at = row.get('created_at', None)
            pred = self.predict(row['text'], row['company'], created_at)
            predictions.append(pred)
        
        return predictions
    
    def analyze_competition(self, text, created_at=None):
        """
        Analizar el rendimiento potencial del mismo tweet para todas las empresas
        """
        results = {}
        
        for company in self.companies:
            pred = self.predict(text, company, created_at)
            results[company] = pred
        
        # Ordenar por predicción descendente
        sorted_results = dict(sorted(results.items(), key=lambda x: x[1], reverse=True))
        
        return sorted_results

# Ejemplo de uso
if __name__ == "__main__":
    try:
        # Inicializar predictor
        predictor = CompaniesPredictor()
        
        # Ejemplo 1: Predicción individual
        tweet_text = "🚀 Lanzamos una nueva funcionalidad que va a revolucionar tus finanzas! #fintech #innovación"
        company = "uala"
        
        prediction = predictor.predict(tweet_text, company)
        print(f"\nPredicción para {company}: {prediction} retweets")
        
        # Ejemplo 2: Análisis de competencia
        print(f"\nAnálisis de competencia para el mismo tweet:")
        competition_analysis = predictor.analyze_competition(tweet_text)
        
        for comp, pred in competition_analysis.items():
            print(f"{comp}: {pred} retweets")
        
        # Ejemplo 3: Predicción con fecha específica
        from datetime import datetime
        weekend_date = datetime(2025, 9, 28, 19, 0)  # Sábado a las 7 PM
        weekend_prediction = predictor.predict(tweet_text, company, weekend_date)
        print(f"\nPredicción para fin de semana: {weekend_prediction} retweets")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Primero ejecuta predictor.py para entrenar el modelo")
    except Exception as e:
        print(f"Error inesperado: {e}")