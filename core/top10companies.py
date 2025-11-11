# -*- coding: utf-8 -*-
"""
Lee el CSV (Usuario, Empresa) y construye el TOP 10 de empresas más seguidas
(contando usuarios únicos). Genera: top10_<slug>.csv
"""

import re
import pandas as pd
from pathlib import Path

def slug_from_filename(csv_path: Path) -> str:
    m = re.search(r"reactors_user_company_(.+)\.csv$", csv_path.name)
    return (m.group(1) if m else "post").strip("_")

def main():
    csv_file = input("Ruta al CSV (empresas_*.csv): ").strip()
    path = Path(csv_file)
    if not path.exists():
        print("No existe el archivo.")
        return

    df = pd.read_csv(path)

    # Sanitizar columnas esperadas
    if not {"Usuario","Empresa"}.issubset(df.columns):
        print("El CSV debe tener columnas: Usuario, Empresa")
        return

    # Limpieza mínima
    df["Usuario"] = df["Usuario"].astype(str).str.strip()
    df["Empresa"] = df["Empresa"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    df = df[(df["Usuario"]!="") & (df["Empresa"]!="")]

    # Evitar contar mismo usuario-empresa repetido múltiples veces
    df_unique = df.drop_duplicates(subset=["Usuario", "Empresa"])

    # Conteo por usuarios únicos
    top = (df_unique
           .groupby("Empresa")["Usuario"]
           .nunique()
           .reset_index(name="usuarios_unicos")
           .sort_values(["usuarios_unicos","Empresa"], ascending=[False, True])
           .head(10))

    out = Path(f"top10_{slug_from_filename(path)}.csv")
    top.to_csv(out, index=False, encoding="utf-8")

    print(f"✅ TOP 10 generado: {out}")
    print(top.to_string(index=False))


if __name__ == "__main__":
    main()
