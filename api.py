# api.py
import ray
import os
import pickle
import json
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
from functools import lru_cache
from contextlib import asynccontextmanager
import train # Necesario para ModelRegistryActor

# --- Modelos Pydantic ---
class PredictionRequest(BaseModel):
    features: Dict[str, float]

class PredictionResponse(BaseModel):
    prediction: Any
    model_details: Dict[str, str]

# --- Lifespan Manager ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    print(">>> LIFESPAN (API Service): Evento de inicio iniciado <<<")
    try:
        ray_address = os.environ.get("RAY_ADDRESS", "auto")
        if not ray.is_initialized():
            ray.init(address=ray_address, namespace="mi_plataforma", ignore_reinit_error=True)
        # Asegurar que el actor exista (o obtener una referencia a él)
        train.ModelRegistryActor.options(name="model_registry", get_if_exists=True, namespace="mi_plataforma").remote()
        print(">>> LIFESPAN (API Service): listo. ModelRegistryActor asegurado.")
    except Exception as e:
        print(f"!!! LIFESPAN (API Service): ERROR CRÍTICO DURANTE EL STARTUP: {e} !!!")
    yield
    if ray.is_initialized():
        ray.shutdown()

# --- App FastAPI ---
app = FastAPI(
    title="API de Inferencia Distribuida", 
    version="2.0.0",
    lifespan=lifespan
)

# --- Funciones de Ayuda ---
@lru_cache(maxsize=16)
def get_deserialized_artifact(dataset: str, model_type: str, artifact_name: str):
    """
    Obtiene la ObjectRef del diccionario de artefactos, lo resuelve,
    extrae el artefacto deseado, lo deserializa y lo cachea.
    """
    print(f"CACHE MISS: Obteniendo de Ray: {dataset}/{model_type}/{artifact_name}")
    try:
        registry_actor = ray.get_actor("model_registry", namespace="mi_plataforma")
        
        # Paso 1: Obtener la referencia al diccionario de artefactos
        artifacts_dict_ref = ray.get(registry_actor.get_model_artifacts_ref.remote(dataset, model_type))
        if not artifacts_dict_ref: 
            raise ValueError("Referencia al diccionario de artefactos no encontrada en el registro.")
        
        # Paso 2: Obtener el diccionario de artefactos (que contiene los bytes serializados)
        artifacts_dict = ray.get(artifacts_dict_ref)
        if not artifacts_dict: 
            raise ValueError("El diccionario de artefactos está vacío (None).")

        # Paso 3: Extraer los bytes del artefacto específico
        serialized_obj = artifacts_dict[artifact_name]
        
        # Paso 4: Deserializar
        return pickle.loads(serialized_obj) if artifact_name != "metrics" else json.loads(serialized_obj.decode('utf-8'))
    except Exception as e:
        # Importante: Si falla, no debe cachear el resultado None. lru_cache no cachea excepciones.
        print(f"ERROR obteniendo artefacto '{artifact_name}' para {dataset}/{model_type}: {e}")
        raise HTTPException(status_code=500, detail=f"Error interno al obtener el artefacto: {e}")

# --- Endpoints ---
@app.get("/models")
async def list_models():
    """Consulta al ModelRegistryActor para listar los modelos disponibles."""
    try:
        registry_actor = ray.get_actor("model_registry", namespace="mi_plataforma")
        return await registry_actor.list_models_details.remote()
    except ValueError:
        raise HTTPException(status_code=503, detail="ModelRegistryActor no disponible.")

@app.post("/predict/{dataset}/{model_type}", response_model=PredictionResponse)
def predict(dataset: str, model_type: str, request: PredictionRequest):
    """Realiza una predicción utilizando un modelo obtenido de forma distribuida."""
    try:
        pipeline = get_deserialized_artifact(dataset, model_type, "pipeline")
        required_features = get_deserialized_artifact(dataset, model_type, "feature_names")
    except HTTPException as e:
        raise e # Re-lanzar la excepción HTTP generada por la función de ayuda

    if not pipeline or not required_features:
        # Este caso es redundante si la función de ayuda siempre lanza excepciones, pero es una buena guarda.
        raise HTTPException(status_code=404, detail=f"Pipeline o feature_names no se pudieron cargar para {dataset}/{model_type}")

    try:
        # Crear un DataFrame con el orden correcto de las columnas
        input_df = pd.DataFrame([request.features])[required_features]
        prediction = pipeline.predict(input_df)
        prediction_value = prediction[0].item() if hasattr(prediction[0], 'item') else prediction[0]
        
        return PredictionResponse(prediction=prediction_value, model_details={"dataset": dataset, "model_type": model_type})
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Falta la característica requerida en la petición: {e}. Se requieren: {required_features}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error durante la predicción: {e}")