# cluster_posts.py
import pandas as pd
import os, glob
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

def load_posts():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(os.path.dirname(current_dir), "core", "data")
    csv_files = glob.glob(os.path.join(data_path, "*.csv"))
    all_data = []
    for file in csv_files:
        df = pd.read_csv(file)
        df["company"] = os.path.basename(file).replace("posts_", "").replace(".csv", "")
        all_data.append(df)
    return pd.concat(all_data, ignore_index=True)

if __name__ == "__main__":
    print("🔎 Cargando posts...")
    df = load_posts()
    
    print("⚙️ Generando embeddings...")
    model = SentenceTransformer("paraphrase-MiniLM-L6-v2")  # rápido y liviano
    embeddings = model.encode(df["text"].astype(str).tolist(), show_progress_bar=True)
    
    n_clusters = 5
    print(f"📊 Aplicando KMeans con {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df["content_cluster"] = kmeans.fit_predict(embeddings)
    
    out_file = "posts_with_clusters.csv"
    df.to_csv(out_file, index=False)
    print(f"✅ Clustering guardado en {out_file}")
