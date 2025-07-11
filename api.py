# api.py
import os
import pickle
import json
import time
from functools import lru_cache
from contextlib import asynccontextmanager

import ray
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from typing import Dict, Any, List

from prometheus_client import Histogram, generate_latest

# Importar train.py es necesario para que Ray Cliente pueda entender las clases 
# personalizadas como ModelRegistryActor si alguna vez se pasan por referencia.
# Es una buena práctica en arquitecturas de Ray.
import train 

# --- Modelos Pydantic (para la validación de peticiones y respuestas) ---
class PredictionRequest(BaseModel):
    features: Dict[str, float]

class PredictionResponse(BaseModel):
    prediction: Any
    model_details: Dict[str, str]

# --- Métricas de Prometheus ---
INFERENCE_LATENCY = Histogram(
    'inference_latency_seconds',
    'Latencia de las peticiones de inferencia (en segundos)',
    ['dataset', 'model_type']
)

# --- Gestor de Ciclo de Vida (Lifespan) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestiona la conexión al clúster Ray al iniciar y cerrar la aplicación."""
    print(">>> LIFESPAN (API Service): Evento de inicio iniciado <<<")
    try:
        ray_address = os.environ.get("RAY_ADDRESS", "ray://localhost:10001")
        if not ray.is_initialized():
            ray.init(address=ray_address, namespace="mi_plataforma", ignore_reinit_error=True)
        print(">>> LIFESPAN (API Service): Conectado a Ray y listo. <<<")
    except Exception as e:
        print(f"!!! LIFESPAN (API Service): ERROR CRÍTICO DURANTE EL STARTUP: {e} !!!")
    
    yield
    
    if ray.is_initialized():
        print(">>> LIFESPAN (API Service): Desconectando de Ray. <<<")
        ray.shutdown()

# --- Aplicación FastAPI ---
app = FastAPI(
    title="API de Inferencia Distribuida (v2 - Actor-Driven)", 
    version="3.1.0",
    description="Provee acceso a modelos de ML entrenados en un clúster de Ray, obteniendo los artefactos a través de un actor de registro.",
    lifespan=lifespan
)

# --- Funciones de Ayuda ---
@lru_cache(maxsize=32)
def get_artifact_from_actor(dataset: str, model_type: str, artifact_name: str):
    """
    Llama al método inteligente del actor para que nos dé el artefacto ya deserializado.
    La caché LRU almacena el objeto Python final.
    """
    print(f"CACHE MISS: Pidiendo al actor: {dataset}/{model_type}/{artifact_name}")
    try:
        registry_actor = ray.get_actor("model_registry", namespace="mi_plataforma")
        
        # La llamada al método del actor devuelve una ObjectRef al resultado del método.
        # Usamos ray.get() para obtener ese resultado (el artefacto deserializado).
        artifact_ref = registry_actor.get_deserialized_artifact.remote(dataset, model_type, artifact_name)
        artifact = ray.get(artifact_ref)
        
        if artifact is None:
            # Si el actor devolvió None, significa que no pudo encontrar o procesar el artefacto.
            raise HTTPException(status_code=404, detail=f"Artefacto '{artifact_name}' no encontrado para {dataset}/{model_type} por el actor de registro.")
            
        return artifact
            
    except ValueError: # Ocurre si get_actor falla
        raise HTTPException(status_code=503, detail="ModelRegistryActor no disponible.")
    except Exception as e:
        # Captura otros errores de Ray o de la lógica
        raise HTTPException(status_code=500, detail=f"Error inesperado al contactar al actor o procesar la petición: {e}")

# --- Endpoints de la API ---

@app.get("/models", summary="Listar todos los modelos disponibles")
async def list_models():
    """Consulta al ModelRegistryActor para obtener una lista de todos los datasets y sus modelos."""
    try:
        registry_actor = ray.get_actor("model_registry", namespace="mi_plataforma")
        # El método del actor ya devuelve un diccionario simple, por lo que no se necesita un response_model complejo
        return await registry_actor.list_models_details.remote()
    except ValueError:
        raise HTTPException(status_code=503, detail="ModelRegistryActor no disponible en este momento.")

@app.get("/models/{dataset}/{model_type}/features", summary="Obtener características de un modelo")
def get_model_features(dataset: str, model_type: str):
    """Devuelve la lista de nombres de características que un modelo espera como entrada."""
    try:
        required_features = get_artifact_from_actor(dataset, model_type, "feature_names")
        return {"features": required_features}
    except HTTPException as e:
        raise e

@app.get("/models/{dataset}/{model_type}/metrics", summary="Obtener métricas de entrenamiento")
def get_model_metrics(dataset: str, model_type: str):
    """Devuelve las métricas de entrenamiento (accuracy, reporte, etc.) de un modelo."""
    try:
        return get_artifact_from_actor(dataset, model_type, "metrics")
    except HTTPException as e:
        raise e

@app.post("/predict/{dataset}/{model_type}", response_model=PredictionResponse, summary="Realizar una predicción")
def predict(dataset: str, model_type: str, request: PredictionRequest):
    """Realiza una predicción usando un modelo y registra la latencia de la operación."""
    start_time = time.time()
    try:
        # Cargar los artefactos necesarios desde el actor (usará la caché si es posible)
        pipeline = get_artifact_from_actor(dataset, model_type, "pipeline")
        required_features = get_artifact_from_actor(dataset, model_type, "feature_names")

        # Crear un DataFrame con el orden correcto de las columnas
        input_df = pd.DataFrame([request.features])[required_features]
        prediction = pipeline.predict(input_df)
        
        # Convertir la predicción de numpy a un tipo nativo de Python
        prediction_value = prediction[0].item() if hasattr(prediction[0], 'item') else prediction[0]
        
        return PredictionResponse(prediction=prediction_value, model_details={"dataset": dataset, "model_type": model_type})
        
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Falta la característica requerida en la petición: {e}. Se requieren: {required_features}")
    except HTTPException as e:
        # Re-lanzar excepciones HTTP que vienen de get_artifact_from_actor
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error durante la predicción: {e}")
    finally:
        # Este bloque se ejecuta siempre, garantizando que se registre la latencia
        latency = time.time() - start_time
        INFERENCE_LATENCY.labels(dataset=dataset, model_type=model_type).observe(latency)
        print(f"PREDICT LATENCY: {dataset}/{model_type} -> {latency:.4f}s")

@app.get("/metrics", summary="Exponer métricas para Prometheus")
def get_prometheus_metrics():
    """Endpoint para que Prometheus pueda recolectar las métricas de la aplicación."""
    return Response(content=generate_latest(), media_type="text/plain")

# --- Bloque para ejecución local (opcional) ---
if __name__ == "__main__":
    print("Iniciando API de Inferencia en modo local para depuración...")
    print("Asegúrate de tener un clúster Ray corriendo y accesible.")
    uvicorn.run(app, host="0.0.0.0", port=8000)