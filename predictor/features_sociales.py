import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

def generar_features_sociales(df_user, df_top10):
    """
    Enriquecer df_user con features externas de actividad y afinidad social.
    """

    # ================================
    # 1️⃣ Agregados de actividad diaria (top10)
    # ================================
    df_top10["date"] = pd.to_datetime(df_top10["created_at"]).dt.date
    daily_activity = df_top10.groupby("date").agg(
        tweets_top10=("text", "count"),
        likes_top10=("likes", "mean"),
        retweets_top10=("retweets", "mean")
    ).reset_index()

    # Normalizar para evitar escalas locas
    daily_activity["likes_top10_norm"] = (daily_activity["likes_top10"] - daily_activity["likes_top10"].mean()) / daily_activity["likes_top10"].std()
    daily_activity["tweets_top10_norm"] = (daily_activity["tweets_top10"] - daily_activity["tweets_top10"].mean()) / daily_activity["tweets_top10"].std()

    # ================================
    # 2️⃣ Hashtags compartidos
    # ================================
    def extraer_hashtags(text):
        return [w.lower() for w in text.split() if w.startswith("#")]

    df_user["hashtags"] = df_user["text"].apply(extraer_hashtags)
    df_top10["hashtags"] = df_top10["text"].apply(extraer_hashtags)

    # Expandir los hashtags de top10 por día
    all_tags = (
        df_top10.explode("hashtags")
        .dropna(subset=["hashtags"])
        .groupby("date")["hashtags"]
        .apply(set)
        .to_dict()
    )

    df_user["date"] = pd.to_datetime(df_user["date"]).dt.date

    def overlap_tags(row):
        tags_top10 = all_tags.get(row["date"], set())
        if not tags_top10 or not row["hashtags"]:
            return 0
        inter = len(set(row["hashtags"]).intersection(tags_top10))
        return inter / len(set(row["hashtags"]))

    df_user["hashtag_overlap_top10"] = df_user.apply(overlap_tags, axis=1)

    # ================================
    # 3️⃣ Afinidad semántica (BERT)
    # ================================
    model = SentenceTransformer("sentence-transformers/distiluse-base-multilingual-cased-v2")

    # Embeddings de los posts propios y del promedio diario top10
    df_top10_daily = (
        df_top10.groupby("date")["text"]
        .apply(lambda x: " ".join(x))
        .reset_index()
    )

    df_top10_daily["embedding"] = df_top10_daily["text"].apply(lambda t: model.encode(t))
    emb_dict = df_top10_daily.set_index("date")["embedding"].to_dict()

    def sim_bert(row):
        if row["date"] not in emb_dict:
            return 0
        emb_user = model.encode(row["text"])
        emb_top = emb_dict[row["date"]]
        return float(cosine_similarity([emb_user], [emb_top])[0][0])

    df_user["semantic_sim_top10"] = df_user.apply(sim_bert, axis=1)

    # ================================
    # 4️⃣ Merge final con actividad global
    # ================================
    df_enriched = pd.merge(df_user, daily_activity, on="date", how="left")

    return df_enriched
