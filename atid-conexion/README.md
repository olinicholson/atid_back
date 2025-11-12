# ATID Conexión

Microservicio FastAPI para conectar frontend y backend. Expone un endpoint POST /predict_tweet
que recibe texto y fecha y devuelve predicciones (likes, replies, comments, views, retweets).

Cómo usar (local, sin Docker):

1. Crear y activar un entorno virtual:

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Ejecutar el servicio:

```powershell
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

3. Probar endpoint:

POST http://localhost:8000/predict_tweet

Body (JSON):

{
  "text": "Hola, este es un tweet de prueba #test @user",
  "created_at": "2025-11-10T12:00:00",
  "platform": "twitter"
}

Notas:
- Si el proyecto contiene un archivo `predictor/twitter/jump_models.pkl` el servicio intentará cargarlo
  (uso `best-effort`). Si no hay modelos, se usa un predictor heurístico para desarrollo rápido.
