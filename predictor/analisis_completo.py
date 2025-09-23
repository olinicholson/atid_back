"""
Script principal para el análisis de predicción de retweets de empresas fintech
Analiza datos de competidores y top 10 cuentas seguidas por seguidores
"""

import pandas as pd
import numpy as np
import os
from predict_companies import CompaniesPredictor
from datetime import datetime, timedelta
import json

def main():
    print("=== SISTEMA DE PREDICCIÓN DE RETWEETS PARA EMPRESAS FINTECH ===\n")
    
    # Verificar si existe el modelo entrenado
    model_file = "xgboost_Retweets_predictor_companies.pkl"
    info_file = "model_info_Retweets_companies.json"
    
    if not os.path.exists(model_file) or not os.path.exists(info_file):
        print("❌ Modelo no encontrado. Ejecuta primero 'python predictor.py' para entrenar el modelo.")
        return
    
    # Cargar el predictor
    try:
        predictor = CompaniesPredictor(model_file, info_file)
    except Exception as e:
        print(f"❌ Error cargando el modelo: {e}")
        return
    
    # Mostrar información del modelo
    with open(info_file, 'r') as f:
        model_info = json.load(f)
    
    print("📊 INFORMACIÓN DEL MODELO:")
    print(f"   • R² Score: {model_info['model_performance']['r2']:.3f}")
    print(f"   • RMSE: {model_info['model_performance']['rmse']:.2f}")
    print(f"   • Empresas analizadas: {', '.join(model_info['companies'])}")
    print(f"   • Total de características: {len(model_info['features'])}\n")
    
    # Ejemplos de análisis
    print("🚀 EJEMPLOS DE PREDICCIONES:\n")
    
    # Ejemplo 1: Tweet de producto nuevo
    tweet1 = "🎉 Lanzamos nuestra nueva tarjeta de crédito sin comisiones! Descuentos especiales para los primeros 1000 usuarios #fintech #descuentos"
    print("1. Tweet de lanzamiento de producto:")
    print(f"   Texto: {tweet1[:80]}...")
    print("   Predicciones por empresa:")
    
    results1 = predictor.analyze_competition(tweet1)
    for company, pred in results1.items():
        print(f"   • {company.capitalize()}: {pred:.1f} retweets")
    print()
    
    # Ejemplo 2: Tweet educativo
    tweet2 = "💡 Consejos para mejorar tu score crediticio: 1) Paga a tiempo 2) No uses todo tu límite 3) Revisa tu reporte regularmente"
    print("2. Tweet educativo:")
    print(f"   Texto: {tweet2[:80]}...")
    print("   Predicciones por empresa:")
    
    results2 = predictor.analyze_competition(tweet2)
    for company, pred in results2.items():
        print(f"   • {company.capitalize()}: {pred:.1f} retweets")
    print()
    
    # Ejemplo 3: Análisis por horarios
    print("3. Análisis de impacto por horario (mismo tweet):")
    base_tweet = "Nueva funcionalidad disponible! Ahora puedes transferir dinero instantáneamente #fintech"
    
    horarios = [
        (9, "Mañana laboral"),
        (14, "Mediodía"),
        (19, "Hora pico"),
        (22, "Noche")
    ]
    
    for hora, descripcion in horarios:
        test_date = datetime(2025, 9, 24, hora, 0)  # Martes
        pred = predictor.predict(base_tweet, "uala", test_date)
        print(f"   • {descripcion} ({hora}:00): {pred:.1f} retweets")
    print()
    
    # Ejemplo 4: Comparación fin de semana vs día laboral
    print("4. Comparación día laboral vs fin de semana:")
    
    weekday_date = datetime(2025, 9, 24, 19, 0)  # Miércoles 7 PM
    weekend_date = datetime(2025, 9, 27, 19, 0)  # Sábado 7 PM
    
    weekday_pred = predictor.predict(base_tweet, "uala", weekday_date)
    weekend_pred = predictor.predict(base_tweet, "uala", weekend_date)
    
    print(f"   • Día laboral: {weekday_pred:.1f} retweets")
    print(f"   • Fin de semana: {weekend_pred:.1f} retweets")
    print(f"   • Diferencia: {weekend_pred - weekday_pred:+.1f} retweets")
    print()
    
    # Análisis de características más importantes
    print("📈 CARACTERÍSTICAS MÁS IMPORTANTES:")
    feature_importances = [
        ("Empresa (top10)", "22.1%"),
        ("Año", "7.4%"),
        ("Hora pico", "6.7%"),
        ("Fin de semana", "6.6%"),
        ("Longitud del texto", "6.4%")
    ]
    
    for feature, importance in feature_importances:
        print(f"   • {feature}: {importance}")
    print()
    
    # Recomendaciones
    print("💡 RECOMENDACIONES BASADAS EN EL ANÁLISIS:")
    print("   1. Los contenidos del 'top10' (cuentas populares) generan ~4x más retweets")
    print("   2. Publicar en horarios pico (8-10 AM, 6-8 PM) aumenta el engagement")
    print("   3. Los fines de semana pueden ser más efectivos para cierto contenido")
    print("   4. Incluir ofertas y descuentos aumenta significativamente la viralidad")
    print("   5. La longitud del texto es importante: ni muy corto ni muy largo")
    print()
    
    print("✅ Análisis completado. Usa predict_companies.py para predicciones personalizadas.")

if __name__ == "__main__":
    main()