"""
Analizador de Sentimiento con Google Gemini
Analiza el sentimiento y emociones de tweets en español con sistema de caché
"""
import json
import os
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
from dotenv import load_dotenv
from datetime import datetime

# Cargar variables de entorno
load_dotenv()



# Directorio para guardar análisis cacheados
CACHE_DIR = Path(__file__).parent.parent.parent / "core" / "data" / "sentiment_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def analyze_sentiment_gemini(text: str) -> Dict:
    """
    Analiza el sentimiento de un texto usando Google Gemini.
    
    Args:
        text: Texto a analizar
        
    Returns:
        Dict con sentiment, score, emotions, tone, y keywords
    """
    # Análisis de sentimiento estático por palabras clave (sin Gemini)
    text_lower = text.lower()
    positive_words = ['excelente', 'genial', 'bueno', 'gracias', '😊', '🎉', '❤️', '👍', 'feliz', 'increíble']
    negative_words = ['malo', 'pésimo', 'error', 'problema', 'horrible', '😠', '😡', 'enojado', 'triste']

    pos_count = sum(1 for word in positive_words if word in text_lower)
    neg_count = sum(1 for word in negative_words if word in text_lower)

    if pos_count > neg_count:
        sentiment = "positivo"
        score = 70
    elif neg_count > pos_count:
        sentiment = "negativo"
        score = 30
    else:
        sentiment = "neutral"
        score = 50

    return {
        "sentiment": sentiment,
        "score": score,
        "confidence": 40,
        "emotions": [],
        "tone": "neutral",
        "keywords": [],
        "explanation": "Análisis básico por palabras clave (sin Gemini)"
    }


def get_cache_path(csv_name: str) -> Path:
    """Obtiene la ruta del archivo de caché para un CSV."""
    base_name = Path(csv_name).stem
    return CACHE_DIR / f"{base_name}_sentiment_analysis.csv"


def analyze_csv_sentiment(csv_path: str, limit: Optional[int] = None, force_reanalyze: bool = False) -> pd.DataFrame:
    """
    Analiza el sentimiento de todos los tweets en un CSV con sistema de caché.
    
    Si existe un análisis previo:
    - Verifica si hay nuevos tweets (compara la última fecha)
    - Solo analiza los tweets nuevos
    - Mantiene los análisis anteriores en caché
    
    Args:
        csv_path: Ruta al archivo CSV
        limit: Límite de tweets a analizar (None = todos)
        force_reanalyze: Si True, ignora el caché y re-analiza todo
        
    Returns:
        DataFrame con análisis de sentimiento agregado
    """
    csv_name = Path(csv_path).name
    cache_path = get_cache_path(csv_name)
    
    # Leer CSV original
    df_original = pd.read_csv(csv_path)
    
    # Convertir created_at a datetime
    if 'created_at' in df_original.columns:
        df_original['created_at'] = pd.to_datetime(df_original['created_at'], errors='coerce')
    
    if limit:
        df_original = df_original.head(limit)
    
    # Verificar si existe caché
    if cache_path.exists() and not force_reanalyze:
        print(f"� Caché encontrado: {cache_path.name}")
        df_cached = pd.read_csv(cache_path)
        
        if 'created_at' in df_cached.columns:
            df_cached['created_at'] = pd.to_datetime(df_cached['created_at'], errors='coerce')
        
        # Obtener la última fecha del caché
        if 'created_at' in df_cached.columns and not df_cached['created_at'].isna().all():
            last_cached_date = df_cached['created_at'].max()
            print(f"   Última fecha en caché: {last_cached_date}")
            
            # Verificar última fecha en el CSV original
            if 'created_at' in df_original.columns and not df_original['created_at'].isna().all():
                last_original_date = df_original['created_at'].max()
                print(f"   Última fecha en CSV: {last_original_date}")
                
                # Si las fechas coinciden, usar el caché directamente
                if last_cached_date >= last_original_date and len(df_cached) >= len(df_original):
                    print(f"✅ Caché actualizado! Usando {len(df_cached)} tweets analizados")
                    return df_cached
                
                # Si hay tweets nuevos, analizar solo los nuevos
                df_new = df_original[df_original['created_at'] > last_cached_date]
                if len(df_new) > 0:
                    print(f"🆕 {len(df_new)} tweets nuevos encontrados. Analizando solo los nuevos...")
                    df_new_analyzed = _analyze_tweets_batch(df_new)
                    
                    # Combinar caché con nuevos análisis
                    df_combined = pd.concat([df_cached, df_new_analyzed], ignore_index=True)
                    
                    # Guardar caché actualizado
                    df_combined.to_csv(cache_path, index=False)
                    print(f"💾 Caché actualizado con {len(df_combined)} tweets")
                    
                    return df_combined
        
        # Si no hay fechas o hay problemas, usar el caché tal cual
        print(f"✅ Usando caché existente con {len(df_cached)} tweets")
        return df_cached
    
    # No hay caché o se forzó el re-análisis
    print(f"�🔍 Analizando sentimiento de {len(df_original)} tweets (sin caché)...")
    df_results = _analyze_tweets_batch(df_original)
    
    # Guardar en caché
    df_results.to_csv(cache_path, index=False)
    print(f"💾 Análisis guardado en caché: {cache_path.name}")
    
    return df_results


def _analyze_tweets_batch(df: pd.DataFrame) -> pd.DataFrame:
    """
    Función interna para analizar un batch de tweets.
    """
    results = []
    for idx, row in df.iterrows():
        text = row.get('text', '')
        if pd.isna(text) or not text.strip():
            continue
            
        analysis = analyze_sentiment_gemini(text)
        results.append({
            'tweet_id': idx,
            'text': text,
            'sentiment': analysis['sentiment'],
            'score': analysis['score'],
            'confidence': analysis['confidence'],
            'emotions': ','.join(analysis['emotions']) if analysis['emotions'] else '',
            'tone': analysis['tone'],
            'keywords': ','.join(analysis['keywords']) if analysis['keywords'] else '',
            'explanation': analysis['explanation'],
            'likes': row.get('likes', 0),
            'retweets': row.get('retweets', 0),
            'replies': row.get('replies', 0),
            'views': row.get('views', 0),
            'created_at': row.get('created_at', '')
        })
        
        if (idx + 1) % 10 == 0:
            print(f"   Procesados {idx + 1}/{len(df)} tweets...")
    
    df_results = pd.DataFrame(results)
    print(f"✅ Análisis completo de {len(df_results)} tweets!")
    
    return df_results


def get_sentiment_summary(df: pd.DataFrame) -> Dict:
    """
    Genera un resumen estadístico del análisis de sentimiento.
    
    Args:
        df: DataFrame con análisis de sentimiento
        
    Returns:
        Dict con estadísticas resumidas
    """
    if df.empty:
        return {}
    
    total_tweets = len(df)
    sentiment_counts = df['sentiment'].value_counts().to_dict()
    
    # Calcular promedios
    avg_score = df['score'].mean()
    avg_confidence = df['confidence'].mean()
    
    # Tweets más positivos y negativos
    most_positive = df.nlargest(3, 'score')[['text', 'score', 'sentiment']].to_dict('records')
    most_negative = df.nsmallest(3, 'score')[['text', 'score', 'sentiment']].to_dict('records')
    
    # Emociones más comunes
    all_emotions = []
    for emotions_str in df['emotions'].dropna():
        if emotions_str:
            all_emotions.extend([e.strip() for e in emotions_str.split(',')])
    emotion_counts = pd.Series(all_emotions).value_counts().head(5).to_dict() if all_emotions else {}
    
    # Tonos más comunes
    tone_counts = df['tone'].value_counts().head(5).to_dict()
    
    # Correlación con engagement (coeficientes de correlación de Pearson)
    correlation_data = {}
    if 'likes' in df.columns and len(df) > 1:
        # Correlación entre score de sentimiento y métricas de engagement
        numeric_cols = ['likes', 'retweets', 'replies', 'views']
        for col in numeric_cols:
            if col in df.columns:
                # Convertir a numérico y eliminar NaN
                df_clean = df[[col, 'score']].dropna()
                if len(df_clean) > 1:
                    correlation = df_clean['score'].corr(df_clean[col])
                    correlation_data[col] = round(correlation, 3) if not pd.isna(correlation) else 0
                else:
                    correlation_data[col] = 0
        
        # También incluir promedios por tipo de sentimiento para contexto
        # Calcular promedios y transponer para que la clave sea el sentimiento
        sentiment_by_engagement_df = df.groupby('sentiment').agg({
            'likes': 'mean',
            'retweets': 'mean',
            'replies': 'mean',
            'views': 'mean'
        }).round(2)
        sentiment_by_engagement = sentiment_by_engagement_df.to_dict(orient='index')
    else:
        correlation_data = {}
        sentiment_by_engagement = {}
    
    return {
        "total_tweets": total_tweets,
        "sentiment_distribution": {
            "positivo": sentiment_counts.get("positivo", 0),
            "neutral": sentiment_counts.get("neutral", 0),
            "negativo": sentiment_counts.get("negativo", 0)
        },
        "sentiment_percentages": {
            "positivo": round((sentiment_counts.get("positivo", 0) / total_tweets) * 100, 1),
            "neutral": round((sentiment_counts.get("neutral", 0) / total_tweets) * 100, 1),
            "negativo": round((sentiment_counts.get("negativo", 0) / total_tweets) * 100, 1)
        },
        "average_score": round(avg_score, 2),
        "average_confidence": round(avg_confidence, 2),
        "most_positive_tweets": most_positive,
        "most_negative_tweets": most_negative,
        "top_emotions": emotion_counts,
        "top_tones": tone_counts,
        "sentiment_correlation": correlation_data,  # Correlaciones de Pearson
        "sentiment_by_engagement": sentiment_by_engagement  # Promedios por tipo
    }


def analyze_batch_texts(texts: List[str]) -> List[Dict]:
    """
    Analiza el sentimiento de múltiples textos.
    
    Args:
        texts: Lista de textos a analizar
        
    Returns:
        Lista de diccionarios con análisis de sentimiento
    """
    results = []
    for text in texts:
        if not text or not text.strip():
            continue
        analysis = analyze_sentiment_gemini(text)
        analysis['text'] = text
        results.append(analysis)
    
    return results


if __name__ == "__main__":
    # Test del analizador
    test_text = "¡Me encanta Ualá! El mejor servicio de fintech que he usado 🎉"
    print("🧪 Test de análisis de sentimiento:")
    print(f"Texto: {test_text}")
    result = analyze_sentiment_gemini(test_text)
    print(f"Resultado: {json.dumps(result, indent=2, ensure_ascii=False)}")
