import pandas as pd
from sklearn.model_selection import train_test_split
import re
from datetime import timedelta
from transformers import AutoTokenizer, AutoModel
import torch

# Cargar los datos
tweets_df = pd.read_csv("files.csv")  # Cambia por la ruta de tu archivo CSV

# Cargar el CSV de tendencias
trends_csv = 'trends_csv.csv'  # Reemplaza con la ruta al archivo
trends_df = pd.read_csv(trends_csv)
# Convertir las tendencias en listas de palabras clave
trends_df['trends'] = trends_df['trends'].apply(lambda x: [trend.strip().lower() for trend in x.split(',')])
print(trends_df)
tweets_df = tweets_df.drop(columns='Id')

# Lista de palabras clave relacionadas con ofertas ----------------------------------------------------------
keywords = [
    'oferta', 'promoción', 'gratuito', 'descuento', 'rebaja', 'promo', 
    'oportunidad', 'deal', 'sale', 'tiempo limitado', 'beneficio', 'ganancia', 
    'ahorro', 'inversión', 'bonificación', 'regalo', 'sorteo', 'gratis', 'off', 
    'liquidación', 'especial', 'exclusivo', 'oferta única', 'oferta especial', 
    'descuento extra', 'precios bajos', 'rebajas exclusivas', 'ahorra más', 
    'precio reducido', 'imperdible', 'última oportunidad', 'hoy', 'ahora', 'ya', 
    'por tiempo limitado', 'no te lo pierdas', 'solo por hoy', 'fecha límite', 
    'corre', 'cupón', 'código', 'descuento inmediato', 'promoción válida', 'premio', 
    'reembolso', 'gana', 'beneficio extra', 'te obsequiamos', 'participa', 
    'compra ahora', 'adquiere', 'llévate', 'más barato', 'mejor precio', 'pack', 
    'combo', 'paquete', 'promoción 2x1', '3x2', 'mitad de precio', 'free', 
    'discount', 'limited time', 'hot deal', 'exclusive offer', 'special price', 
    'save now', 'value pack', 'bono', 'interés bajo', 'sin intereses', 
    'meses sin intereses', 'financiación', 'costo cero', 'cashback'
]



# Función para verificar si un texto contiene alguna palabra clave

def contains_exact_keywords(text, keywords):
    # Convertir texto a minúsculas para evitar problemas de sensibilidad a mayúsculas
    text = text.lower()
    # Crear un patrón regex que busque palabras completas
    pattern = r'\b(' + '|'.join(re.escape(keyword) for keyword in keywords) + r')\b'
    # Verificar si alguna palabra clave está presente en el texto
    return bool(re.search(pattern, text))

# Crear la nueva variable 'has_offer' (1 si contiene palabras clave, 0 si no)
tweets_df['has_offer'] = tweets_df['Text'].apply(lambda x: 1 if contains_exact_keywords(x, keywords) else 0)

# Crear la columna 'text_length' con la longitud del texto de cada tweet
tweets_df['text_length'] = tweets_df['Text'].apply(len)
# Convertir la columna 'Created At' a datetime con formato correcto
tweets_df['Created At'] = pd.to_datetime(tweets_df['Created At'], format='%a %b %d %H:%M:%S %z %Y')

# Extraer los componentes y crear nuevas columnas
# Restar 3 horas
tweets_df['Created At Argentina'] = tweets_df['Created At'] - timedelta(hours=3)

# Extraer componentes con la hora ajustada
tweets_df['day'] = tweets_df['Created At Argentina'].dt.day
tweets_df['month'] = tweets_df['Created At Argentina'].dt.month
tweets_df['year'] = tweets_df['Created At Argentina'].dt.year
tweets_df['hour'] = tweets_df['Created At Argentina'].dt.strftime('%H:%M:%S')
tweets_df['weekday'] = tweets_df['Created At Argentina'].dt.day_name()

def es_golden_hour(fecha):
    hora = fecha.hour
    return 1 if (9 <= hora < 11) or (17 <= hora < 19) else 0

tweets_df['is_golden_hour'] = tweets_df['Created At Argentina'].apply(es_golden_hour)


# Función para verificar si un texto contiene alguna palabra clave
def contains_trends(text, trends):
    text = text.lower()
    pattern = r'\b(' + '|'.join(re.escape(trend) for trend in trends) + r')\b'
    return bool(re.search(pattern, text))
# Verificar si los tweets contienen tendencias del mes y año correspondiente
def check_trends(row):
    # Filtrar tendencias por mes y año
    trends = trends_df[
        (trends_df['mes'] == row['month']) & (trends_df['año'] == row['year'])
    ]['trends'].values
    if len(trends) > 0:
        return contains_trends(row['Text'], trends[0])
    return False

# Aplicar la función para crear la columna 'has_trend'
tweets_df['has_trend'] = tweets_df.apply(check_trends, axis=1)

print(tweets_df["has_trend"].value_counts())



# Verificar el resultado
print(tweets_df[['Text', 'has_offer']].head())

# Crear un nuevo atributo "has_image" (1 si contiene 'https://t.co/', 0 si no)
tweets_df['has_image'] = tweets_df['Text'].apply(lambda x: 1 if "https://t.co/" in x else 0)

# Aplicar One Hot Encoding a la columna 'Usuario'
one_hot_encoded = pd.get_dummies(tweets_df['Username'], prefix='Username')

# Concatenar las columnas codificadas al DataFrame original
tweets_df = pd.concat([tweets_df, one_hot_encoded], axis=1)
# Separar las columnas relacionadas con el engagement
engagement_columns = ['Likes', 'Retweets', 'Quote_count', 'Reply_count', 'View_count']


# Cargar modelo preentrenado de Sentence-BERT
model_name = "sentence-transformers/distiluse-base-multilingual-cased-v1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Función para obtener embeddings
def text_to_embedding_transformers(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).detach().numpy()
    return embeddings[0]

# Aplicar a cada texto
tweets_df['Embedding'] = tweets_df['Text'].apply(text_to_embedding_transformers)


# Separar las características relevantes y las etiquetas (engagement)
tweets_df = tweets_df.drop(columns=["Username", "Created At"])
tweets_df.to_csv("training_data.csv", index=False)

print("El archivo training_data.csv ha sido exportado correctamente.")
# X = tweets_df.drop(columns=engagement_columns)  # Conjunto de características
# y = tweets_df[engagement_columns]  # Etiquetas (opcional para entrenamiento futuro)

# # Dividir en conjunto de entrenamiento y prueba
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# # Imprimir una muestra para verificar
# print(X_train.head())
# print(y_train.head())

