# Este codido me filtra las apariciones de cada cuenta a la que siguen mis seguidores y 
# luego cuenta las apariciones y las ordena de menor a mayor. 

# analizar_following.py
import pandas as pd

INPUT_CSV = "following_of_followers.csv"
OUTPUT_CSV = "following_counts.csv"

if __name__ == "__main__":
    # Cargar datos
    df = pd.read_csv(INPUT_CSV)

    # Contar apariciones de cada following_username
    counts = (
        df.groupby(["following_username", "following_name"])
        .size()
        .reset_index(name="count")
        .sort_values(by="count", ascending=False)
    )

    # Guardar resultado
    counts.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")

    print(f"Guardado ranking en {OUTPUT_CSV}")
    print(counts.head(10))  # mostrar top 10 por consola
