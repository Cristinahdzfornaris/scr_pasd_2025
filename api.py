# api.py

import os
import pickle
import json
import time
from functools import lru_cache
import requests  # Importante: para hacer peticiones HTTP
import pandas as pd
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel
from typing import Dict, Any, List

from prometheus_client import Histogram, generate_latest

# --- Modelos Pydantic (se mantienen igual) ---
class PredictionRequest(BaseModel):
    features: Dict[str, float]

class PredictionResponse(BaseModel):
    prediction: Any
    model_details: Dict[str, str]

# --- Métricas de Prometheus (se mantienen igual) ---
INFERENCE_LATENCY = Histogram(
    'inference_latency_seconds',
    'Latencia de las peticiones de inferencia (en segundos)',
    ['dataset', 'model_type']
)

# --- Aplicación FastAPI (SIN LIFESPAN) ---
app = FastAPI(
    title="API de Inferencia (v3 - Desacoplada)",
    version="3.0.0",
    description="Provee acceso a modelos de ML consumiendo la API de Registro."
)

# URL del servicio de gestión, leído del entorno
MANAGEMENT_API_URL = os.environ.get("MANAGEMENT_API_URL", "http://localhost:9000")

# --- NUEVAS Funciones de Ayuda ---
@lru_cache(maxsize=32)
def get_artifact_from_management_api(dataset: str, model_type: str, artifact_name: str):
    """
    Llama a la management-api para obtener un artefacto, lo deserializa y lo cachea.
    """
    print(f"CACHE MISS: Pidiendo a management-api: {dataset}/{model_type}/{artifact_name}")
    try:
        url = f"{MANAGEMENT_API_URL}/registry/artifacts/{dataset}/{model_type}/{artifact_name}"
        response = requests.get(url, timeout=10)
        
        response.raise_for_status()  # Lanza una excepción para códigos de error (4xx o 5xx)
        
        serialized_obj_bytes = response.content
        
        # Deserializar según el tipo
        if artifact_name == "metrics":
            return json.loads(serialized_obj_bytes.decode('utf-8'))
        else:
            return pickle.loads(serialized_obj_bytes)

    except requests.exceptions.HTTPError as e:
        status_code = e.response.status_code if e.response else 503
        try:
            # Intentar parsear el detalle del error que viene de management-api
            detail = e.response.json().get("detail", e.response.text)
        except json.JSONDecodeError:
            detail = e.response.text
        raise HTTPException(status_code=status_code, detail=f"Error desde management-api: {detail}")
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=503, detail=f"No se pudo contactar a management-api: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error inesperado al procesar artefacto: {e}")

# --- Endpoints de la API (REESCRITOS PARA USAR LA NUEVA LÓGICA) ---

@app.get("/models", summary="Listar todos los modelos disponibles")
def list_models():
    """Consulta a la management-api para obtener la lista de modelos."""
    try:
        url = f"{MANAGEMENT_API_URL}/registry/models"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"No se pudo conectar a management-api para listar modelos: {e}")

@app.get("/models/{dataset}/{model_type}/features", summary="Obtener características de un modelo")
def get_model_features(dataset: str, model_type: str):
    """Obtiene la lista de características desde la management-api."""
    try:
        features = get_artifact_from_management_api(dataset, model_type, "feature_names")
        return {"features": features}
    except HTTPException as e:
        raise e  # Re-lanzar la excepción que ya viene formateada

@app.get("/models/{dataset}/{model_type}/metrics", summary="Obtener métricas de entrenamiento")
def get_model_metrics(dataset: str, model_type: str):
    """Obtiene las métricas del modelo desde la management-api."""
    try:
        metrics = get_artifact_from_management_api(dataset, model_type, "metrics")
        return metrics
    except HTTPException as e:
        raise e

@app.post("/predict/{dataset}/{model_type}", response_model=PredictionResponse, summary="Realizar una predicción")
def predict(dataset: str, model_type: str, request: PredictionRequest):
    """Realiza una predicción usando un modelo y registra la latencia."""
    start_time = time.time()
    required_features = [] # Para el mensaje de error
    try:
        pipeline = get_artifact_from_management_api(dataset, model_type, "pipeline")
        required_features = get_artifact_from_management_api(dataset, model_type, "feature_names")

        input_df = pd.DataFrame([request.features])[required_features]
        prediction = pipeline.predict(input_df)
        
        prediction_value = prediction[0].item() if hasattr(prediction[0], 'item') else prediction[0]
        
        return PredictionResponse(prediction=prediction_value, model_details={"dataset": dataset, "model_type": model_type})
        
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Falta la característica requerida: {e}. Se requieren: {required_features}")
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error durante la predicción: {e}")
    finally:
        latency = time.time() - start_time
        INFERENCE_LATENCY.labels(dataset=dataset, model_type=model_type).observe(latency)
        print(f"PREDICT LATENCY: {dataset}/{model_type} -> {latency:.4f}s")



@app.get("/health", summary="Comprobación de Salud", status_code=200)
def health_check():
    """
    Endpoint simple para que Docker Healthcheck pueda verificar que el servicio está vivo.
    """
    return {"status": "ok"}

@app.get("/metrics", summary="Exponer métricas para Prometheus")
def get_prometheus_metrics():
    return Response(content=generate_latest(), media_type="text/plain")