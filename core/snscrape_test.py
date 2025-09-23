#!/usr/bin/env python3
"""
Script alternativo usando snscrape para obtener tweets reales sin autenticación
Requiere: pip install snscrape
"""

import subprocess
import sys
import os
import csv
import json
from datetime import datetime, timedelta

def install_snscrape():
    """Instala snscrape si no está disponible"""
    try:
        import snscrape
        print("✅ snscrape ya está instalado")
        return True
    except ImportError:
        print("📦 Instalando snscrape...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "snscrape"])
            print("✅ snscrape instalado exitosamente")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Error instalando snscrape: {e}")
            return False

def scrape_with_snscrape(account, since_date, until_date, max_tweets=50):
    """Scrapea tweets usando snscrape"""
    print(f"🔍 Scrapeando @{account} con snscrape...")
    
    # Comando snscrape
    cmd = [
        "snscrape",
        "--jsonl",
        "--max-results", str(max_tweets),
        "twitter-search",
        f"from:{account} since:{since_date} until:{until_date}"
    ]
    
    try:
        # Ejecutar comando
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # Procesar salida JSON
        tweets = []
        for line in result.stdout.strip().split('\n'):
            if line:
                try:
                    tweet_data = json.loads(line)
                    
                    # Convertir a formato compatible
                    formatted_tweet = {
                        "tweet_id": tweet_data.get("id", ""),
                        "user_name": tweet_data.get("user", {}).get("displayname", ""),
                        "text": tweet_data.get("content", ""),
                        "created_at": tweet_data.get("date", ""),
                        "retweet_count": tweet_data.get("retweetCount", 0),
                        "favorite_count": tweet_data.get("likeCount", 0),
                        "quote_count": tweet_data.get("quoteCount", 0),
                        "reply_count": tweet_data.get("replyCount", 0),
                        "view_count": tweet_data.get("viewCount", 0),
                        "place": tweet_data.get("place", None)
                    }
                    tweets.append(formatted_tweet)
                except json.JSONDecodeError:
                    continue
        
        print(f"✅ Obtenidos {len(tweets)} tweets de @{account}")
        return tweets
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error ejecutando snscrape: {e}")
        print(f"Salida de error: {e.stderr}")
        return []

def save_tweets_to_csv(tweets_data, filename):
    """Guarda los tweets en un archivo CSV"""
    if not tweets_data:
        print(f"No hay tweets para guardar en {filename}")
        return
        
    # Crear directorio si no existe
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['tweet_id', 'user_name', 'text', 'created_at', 'retweet_count', 
                     'favorite_count', 'quote_count', 'reply_count', 'view_count', 'place']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for tweet in tweets_data:
            writer.writerow(tweet)
    print(f"✅ Tweets guardados en {filename}")

def save_tweets_to_json(tweets_data, filename):
    """Guarda los tweets en un archivo JSON"""
    if not tweets_data:
        print(f"No hay tweets para guardar en {filename}")
        return
    
    # Crear directorio si no existe
    os.makedirs(os.path.dirname(filename), exist_ok=True)
        
    with open(filename, 'w', encoding='utf-8') as jsonfile:
        json.dump(tweets_data, jsonfile, indent=2, ensure_ascii=False, default=str)
    print(f"✅ Tweets guardados en {filename}")

def main():
    print("🚀 Scrapper de Twitter usando snscrape (sin autenticación)")
    print("🎯 Objetivo: @mercadopago")
    print("=" * 60)
    
    # Verificar/instalar snscrape
    if not install_snscrape():
        print("❌ No se pudo instalar snscrape. Saliendo...")
        return
    
    # Configurar fechas (últimos 30 días para más datos)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    since_date = start_date.strftime('%Y-%m-%d')
    until_date = end_date.strftime('%Y-%m-%d')
    
    print(f"📅 Período: {since_date} hasta {until_date}")
    
    # Cuentas a scrapear
    accounts = ['mercadopago']
    
    for account in accounts:
        print(f"\n{'='*50}")
        print(f"📱 Procesando @{account}")
        print(f"{'='*50}")
        
        # Scrapear tweets (más tweets para mejor dataset)
        tweets_data = scrape_with_snscrape(account, since_date, until_date, max_tweets=200)
        
        if tweets_data:
            # Guardar archivos
            csv_filename = f"../data/tweets_{account.lower()}_real.csv"
            json_filename = f"../data/tweets_{account.lower()}_real.json"
            
            save_tweets_to_csv(tweets_data, csv_filename)
            save_tweets_to_json(tweets_data, json_filename)
            
            # Estadísticas
            print(f"\n📊 Estadísticas para @{account}:")
            print(f"   Total tweets: {len(tweets_data)}")
            
            if tweets_data:
                total_likes = sum(tweet.get('favorite_count', 0) for tweet in tweets_data)
                total_retweets = sum(tweet.get('retweet_count', 0) for tweet in tweets_data)
                total_replies = sum(tweet.get('reply_count', 0) for tweet in tweets_data)
                total_views = sum(tweet.get('view_count', 0) for tweet in tweets_data if tweet.get('view_count'))
                
                print(f"   Total likes: {total_likes:,}")
                print(f"   Total retweets: {total_retweets:,}")
                print(f"   Total replies: {total_replies:,}")
                if total_views > 0:
                    print(f"   Total views: {total_views:,}")
                
                if len(tweets_data) > 0:
                    print(f"   Promedio likes: {total_likes/len(tweets_data):.1f}")
                    print(f"   Promedio retweets: {total_retweets/len(tweets_data):.1f}")
                    print(f"   Promedio replies: {total_replies/len(tweets_data):.1f}")
                
                # Tweet con más engagement
                max_engagement_tweet = max(tweets_data, 
                    key=lambda x: (x.get('favorite_count', 0) + 
                                 x.get('retweet_count', 0) + 
                                 x.get('reply_count', 0)))
                
                print(f"\n🔥 Tweet con más engagement:")
                print(f"   Texto: {max_engagement_tweet['text'][:100]}...")
                print(f"   Likes: {max_engagement_tweet.get('favorite_count', 0):,}")
                print(f"   RTs: {max_engagement_tweet.get('retweet_count', 0):,}")
                print(f"   Replies: {max_engagement_tweet.get('reply_count', 0):,}")
        else:
            print(f"❌ No se obtuvieron tweets para @{account}")
    
    print(f"\n🎉 Scrapping completado!")
    print("📁 Revisa el directorio 'data/' para los archivos generados")

if __name__ == "__main__":
    main()