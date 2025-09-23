import pandas as pd
import os

# Ruta a la carpeta con los archivos CSV
folder_path = 'files'

# Lista para almacenar los DataFrames
dataframes = []

# Iterar sobre los archivos en la carpeta
for filename in os.listdir(folder_path):
    if filename.endswith('.csv'):  # Verificar que sea un archivo CSV
        file_path = os.path.join(folder_path, filename)
        df = pd.read_csv(file_path)
        dataframes.append(df)

# Combinar todos los DataFrames
combined_df = pd.concat(dataframes, ignore_index=True)

# Guardar el DataFrame combinado en un archivo CSV
output_file = 'files.csv'
combined_df.to_csv(output_file, index=False)

print(f"Archivos combinados y guardados en: {output_file}")
