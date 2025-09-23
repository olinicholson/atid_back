import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime
import os

class ClustersPredictor:
    def __init__(self, model_file="xgboost_retweets_predictor_clusters.pkl", 
                 info_file="model_info_clusters.json"):
        """
        Inicializar el predictor con el modelo entrenado que incluye clusters
        """
        if not os.path.exists(model_file) or not os.path.exists(info_file):
            raise FileNotFoundError(f"Archivos del modelo no encontrados. Ejecuta primero predictor.py")
        
        self.model = joblib.load(model_file)
        
        with open(info_file, 'r') as f:
            self.model_info = json.load(f)
        
        self.features = self.model_info['features']
        self.companies = self.model_info['companies']
        self.clusters = self.model_info['clusters']
        
        print(f"Modelo con clusters cargado exitosamente")
        print(f"Características: {len(self.features)}")
        print(f"Empresas: {self.companies}")
        print(f"Clusters: {self.clusters}")
    
    def prepare_features(self, text, company, content_cluster, created_at=None):
        """
        Preparar características para un tweet individual incluyendo cluster y seguidores
        """
        if created_at is None:
            created_at = datetime.now()
        elif isinstance(created_at, str):
            created_at = pd.to_datetime(created_at)
        
        # Mapeo de seguidores por empresa
        followers_mapping = {
            'uala': 190500,
            'naranjax': 190300,
            'balanz': 38000,
            'brubank': 47900,
            'cocos': 86500,
            'top10': 100000  # Valor promedio para cuentas top10
        }
        
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
        
        # Características de seguidores
        followers_count = followers_mapping.get(company, 50000)  # Default para empresas no listadas
        features_dict['followers_count'] = followers_count
        features_dict['followers_log'] = np.log1p(followers_count)
        features_dict['high_followers'] = 1 if followers_count > 100000 else 0
        
        # Detectar ofertas y tendencias
        text_lower = str(text).lower()
        offer_keywords = ['descuento', 'oferta', 'promoción', 'gratis', 'regalo', '%', 'promo']
        trend_keywords = ['trending', 'viral', '#', 'nuevo', 'lanzamiento']
        
        features_dict['has_offer'] = 1 if any(keyword in text_lower for keyword in offer_keywords) else 0
        features_dict['has_trend'] = 1 if any(keyword in text_lower for keyword in trend_keywords) else 0
        features_dict['has_image'] = 1 if any(img in str(text) for img in ['pic.twitter.com', 'instagram.com', 'imgur.com']) else 0
        
        # Variables dummy para empresas
        for comp in self.companies:
            features_dict[f'Company_{comp}'] = 1 if comp == company else 0
        
        # Variables dummy para clusters
        for cluster in self.clusters:
            features_dict[f'Cluster_{cluster}'] = 1 if cluster == content_cluster else 0
        
        # Asegurar que tenemos todas las características necesarias
        for feature in self.features:
            if feature not in features_dict:
                features_dict[feature] = 0
        
        # Crear array con las características en el orden correcto
        feature_values = [features_dict.get(feature, 0) for feature in self.features]
        return np.array(feature_values).reshape(1, -1)
    
    def predict(self, text, company, content_cluster, created_at=None):
        """
        Predecir retweets para un tweet con cluster específico
        """
        if company not in self.companies:
            print(f"Warning: {company} no está en las empresas conocidas: {self.companies}")
        
        if content_cluster not in self.clusters:
            print(f"Warning: {content_cluster} no está en los clusters conocidos: {self.clusters}")
        
        features = self.prepare_features(text, company, content_cluster, created_at)
        
        # El modelo predice en espacio log, así que aplicamos expm1
        prediction_log = self.model.predict(features)[0]
        prediction = np.expm1(max(0, prediction_log))
        
        return round(prediction, 2)
    
    def predict_all_clusters(self, text, company, created_at=None):
        """
        Predecir para todos los clusters y ver cuál funciona mejor
        """
        results = {}
        
        for cluster in self.clusters:
            pred = self.predict(text, company, cluster, created_at)
            results[f"Cluster_{cluster}"] = pred
        
        # Ordenar por predicción descendente
        sorted_results = dict(sorted(results.items(), key=lambda x: x[1], reverse=True))
        
        return sorted_results
    
    def analyze_content_strategy(self, text, created_at=None):
        """
        Analizar qué combinación empresa-cluster funciona mejor para un texto
        """
        results = []
        
        for company in self.companies:
            for cluster in self.clusters:
                pred = self.predict(text, company, cluster, created_at)
                results.append({
                    'company': company,
                    'cluster': cluster,
                    'predicted_retweets': pred
                })
        
        # Convertir a DataFrame y ordenar
        df_results = pd.DataFrame(results)
        df_results = df_results.sort_values('predicted_retweets', ascending=False)
        
        return df_results
    
    def get_cluster_insights(self):
        """
        Obtener insights sobre qué clusters funcionan mejor
        """
        # Cargar datos originales para análisis
        try:
            clustered_file = "../posts_with_clusters.csv"
            df = pd.read_csv(clustered_file)
            
            cluster_stats = df.groupby('content_cluster').agg({
                'retweets': ['count', 'mean', 'std'],
                'likes': 'mean',
                'text': lambda x: x.str.len().mean()
            }).round(2)
            
            print("Estadísticas por cluster:")
            print(cluster_stats)
            
            return cluster_stats
            
        except Exception as e:
            print(f"No se pudieron cargar las estadísticas: {e}")
            return None

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.decomposition import PCA

def visualize_clusters(file="posts_with_clusters.csv", n_components=2, sample_size=2000):
    # Cargar datos
    df = pd.read_csv(file)
    
    if "content_cluster" not in df.columns:
        raise ValueError("El archivo no contiene la columna 'content_cluster'. Ejecuta primero cluster_posts.py")
    
    # Samplear para que no explote el gráfico
    if len(df) > sample_size:
        df = df.sample(sample_size, random_state=42)
    
    # Usar texto para generar embeddings PCA (ya tenés PCA guardado)
    # Si ya guardaste las columnas text_emb_0... en el csv podés usarlas directamente
    emb_cols = [c for c in df.columns if c.startswith("text_emb_")]
    
    if not emb_cols:
        # Si no hay columnas emb, recomputar PCA desde features básicos (texto)
        print("⚠️ No se encontraron columnas de embeddings, recomputando PCA sobre longitudes de texto y features simples.")
        feature_cols = ["text_length", "likes", "retweets", "replies", "views"]
        X = df[feature_cols].fillna(0)
        pca = PCA(n_components=n_components, random_state=42)
        X_pca = pca.fit_transform(X)
    else:
        X = df[emb_cols].values
        pca = PCA(n_components=n_components, random_state=42)
        X_pca = pca.fit_transform(X)
    
    # Crear DataFrame con PCA
    pca_df = pd.DataFrame(X_pca, columns=[f"PC{i+1}" for i in range(n_components)])
    pca_df["cluster"] = df["content_cluster"].values
    
    # Visualización 2D
    if n_components == 2:
        plt.figure(figsize=(10, 7))
        sns.scatterplot(
            data=pca_df, x="PC1", y="PC2",
            hue="cluster", palette="tab10", alpha=0.7, s=50
        )
        plt.title("Visualización PCA de Clusters de Tweets")
        plt.legend(title="Cluster")
        plt.show()
    
    # Visualización 3D
    elif n_components == 3:
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection="3d")
        scatter = ax.scatter(
            pca_df["PC1"], pca_df["PC2"], pca_df["PC3"],
            c=pca_df["cluster"], cmap="tab10", alpha=0.7
        )
        ax.set_title("Visualización PCA 3D de Clusters de Tweets")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")
        fig.colorbar(scatter, ax=ax, label="Cluster")
        plt.show()
    
    def analyze_followers_impact(self, text, content_cluster, created_at=None):
        """
        Analizar el impacto de los seguidores comparando predicciones entre empresas
        """
        print("Análisis del impacto de seguidores en las predicciones:")
        print("=" * 60)
        
        followers_mapping = {
            'uala': 190500,
            'naranjax': 190300,
            'balanz': 38000,
            'brubank': 47900,
            'cocos': 86500,
            'top10': 100000
        }
        
        results = []
        
        for company in self.companies:
            if company in followers_mapping:
                prediction = self.predict(text, company, content_cluster, created_at)
                followers = followers_mapping[company]
                
                results.append({
                    'company': company,
                    'followers': followers,
                    'prediction': prediction,
                    'prediction_per_1k_followers': (prediction / followers) * 1000
                })
        
        # Ordenar por número de seguidores
        results_sorted = sorted(results, key=lambda x: x['followers'], reverse=True)
        
        print(f"{'Empresa':<12} {'Seguidores':<12} {'Predicción':<12} {'Pred/1k seg':<12}")
        print("-" * 60)
        
        for result in results_sorted:
            print(f"{result['company']:<12} {result['followers']:<12,} {result['prediction']:<12.1f} {result['prediction_per_1k_followers']:<12.3f}")
        
        # Calcular correlación entre seguidores y predicción
        followers_list = [r['followers'] for r in results]
        predictions_list = [r['prediction'] for r in results]
        
        correlation = np.corrcoef(followers_list, predictions_list)[0, 1]
        print(f"\nCorrelación entre seguidores y predicción: {correlation:.3f}")
        
        return results_sorted


# Ejemplo de uso
if __name__ == "__main__":
    try:
        # Inicializar predictor
        predictor = ClustersPredictor()
        
        # Ejemplo 1: Predicción con cluster específico
        tweet_text = "🚀 Lanzamos una nueva funcionalidad que va a revolucionar tus finanzas! #fintech #innovación"
        company = "uala"
        cluster = 3  # Asumiendo que es un cluster de innovación/producto
        
        prediction = predictor.predict(tweet_text, company, cluster)
        print(f"\nPredicción para {company} con cluster {cluster}: {prediction} retweets")
        
        # Ejemplo 2: Probar todos los clusters
        print(f"\nPredicción por cluster para {company}:")
        cluster_predictions = predictor.predict_all_clusters(tweet_text, company)
        
        for cluster_name, pred in cluster_predictions.items():
            print(f"{cluster_name}: {pred} retweets")
        
        # Ejemplo 3: Análisis del impacto de seguidores
        print("\n" + "="*60)
        follower_analysis = predictor.analyze_followers_impact(tweet_text, cluster)
        
        # Ejemplo 4: Comparar empresas con y sin muchos seguidores
        print(f"\nComparación de empresas por tamaño de audiencia:")
        high_follower_companies = [r for r in follower_analysis if r['followers'] > 100000]
        low_follower_companies = [r for r in follower_analysis if r['followers'] <= 100000]
        
        if high_follower_companies and low_follower_companies:
            avg_pred_high = np.mean([r['prediction'] for r in high_follower_companies])
            avg_pred_low = np.mean([r['prediction'] for r in low_follower_companies])
            
            print(f"Predicción promedio (>100k seguidores): {avg_pred_high:.1f} retweets")
            print(f"Predicción promedio (≤100k seguidores): {avg_pred_low:.1f} retweets")
            print(f"Diferencia: {avg_pred_high - avg_pred_low:.1f} retweets (+{((avg_pred_high/avg_pred_low - 1)*100):.1f}%)")
        
        # Ejemplo 5: Predicción optimizada (mejor empresa + cluster)
        print(f"\nOptimización de contenido:")
        best_combinations = []
        
        for company in predictor.companies[:6]:  # Limitar a las principales empresas
            cluster_preds = predictor.predict_all_clusters(tweet_text, company)
            best_cluster = max(cluster_preds.items(), key=lambda x: x[1])
            best_combinations.append({
                'company': company,
                'cluster': best_cluster[0],
                'prediction': best_cluster[1]
            })
        
        best_combinations.sort(key=lambda x: x['prediction'], reverse=True)
        
        print("Top 3 combinaciones empresa + cluster:")
        for i, combo in enumerate(best_combinations[:3], 1):
            print(f"{i}. {combo['company']} con {combo['cluster']}: {combo['prediction']:.1f} retweets")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Primero ejecuta predictor.py para entrenar el modelo con clusters")
    except Exception as e:
        print(f"Error inesperado: {e}")