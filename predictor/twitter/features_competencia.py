import pandas as pd
import numpy as np

def generar_features_competencia(df_uala, df_comp):
    df_comp = df_comp.copy()
    df_comp["created_at"] = pd.to_datetime(df_comp["created_at"]).dt.tz_localize(None)
    df_uala["created_at"] = pd.to_datetime(df_uala["created_at"]).dt.tz_localize(None)

    # === Clasificar sector si existe dataset_name ===
    def clasificar_sector(name):
        name = str(name).lower()
        if any(k in name for k in ["galicia", "supervielle", "bbva", "santander"]):
            return "banco"
        if any(k in name for k in ["uala", "brubank", "cocos", "mercadopago", "balanz"]):
            return "fintech"
        return "otro"

    df_comp["sector"] = df_comp["dataset_name"].apply(clasificar_sector)

    # === Resumen diario general y por sector ===
    daily = (
        df_comp.groupby("created_at")[["likes","replies","views"]]
        .median()
        .rename(columns=lambda c: f"comp_{c}_median")
    )

    sector = (
        df_comp.groupby(["created_at","sector"])[["likes","replies","views"]]
        .median()
        .unstack("sector")
        .fillna(method="ffill")
    )
    sector.columns = [f"{a}_{b}" for a,b in sector.columns]

    # === Rolling 7 días y delta % ===
    full = pd.concat([daily, sector], axis=1).sort_index()
    roll = full.rolling("7D", min_periods=3).mean()
    delta = roll.pct_change().fillna(0)

    roll = roll.add_suffix("_7dmean")
    delta = delta.add_suffix("_trend")

    comp_feats = pd.concat([roll, delta], axis=1).reset_index().rename(columns={"created_at":"date"})
    df_uala = pd.merge_asof(
        df_uala.sort_values("created_at"),
        comp_feats.sort_values("date"),
        left_on="created_at", right_on="date", direction="backward"
    )
    return df_uala.fillna(0)
