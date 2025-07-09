# api.py
import ray
import os
import pickle
import json
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
import uvicorn
from functools import lru_cache

# --- Modelos Pydantic ---
class PredictionRequest(BaseModel):
    features: Dict[str, float]

class PredictionResponse(BaseModel):
    prediction: Any
    model_details: Dict[str, str]

app = FastAPI(title="API de Inferencia Distribuida", version="2.0.0")

# --- Lógica de Conexión y Caché ---
@lru_cache(maxsize=32)
def get_deserialized_artifact_from_ray(dataset: str, model_type: str, artifact_name: str):
    print(f"Buscando en caché o Ray: {dataset}/{model_type}/{artifact_name}")
    try:
        registry_actor = ray.get_actor("model_registry")
        artifact_ref_dict_ref = registry_actor.get_model_artifacts_ref.remote(dataset, model_type)
        artifact_ref_dict = ray.get(artifact_ref_dict_ref)
        
        if not artifact_ref_dict:
            raise ValueError(f"Modelo {dataset}/{model_type} no encontrado en registro.")
        
        # Ahora obtenemos el ObjectRef para el artefacto específico
        task_result_ref = artifact_ref_dict # El valor guardado es la ref al resultado de la tarea
        task_result = ray.get(task_result_ref)
        
        if not task_result or artifact_name not in task_result:
            raise ValueError(f"Artefacto '{artifact_name}' no encontrado en el resultado de la tarea.")

        serialized_obj = task_result[artifact_name]
        
        if artifact_name == "metrics":
            return json.loads(serialized_obj.decode('utf-8'))
        else:
            return pickle.loads(serialized_obj)
    except Exception as e:
        print(f"ERROR obteniendo artefacto: {e}")
        return None

@app.on_event("startup")
def startup_event():
    ray_address = os.environ.get("RAY_ADDRESS", "ray://localhost:10001")
    if not ray.is_initialized():
        ray.init(address=ray_address, ignore_reinit_error=True)
    print("API de Inferencia conectada a Ray.")

# --- Endpoints ---
@app.get("/models")
async def list_models():
    try:
        registry_actor = ray.get_actor("model_registry")
        return await registry_actor.list_models_details.remote()
    except ValueError:
        raise HTTPException(status_code=503, detail="ModelRegistryActor no disponible.")

@app.post("/predict/{dataset}/{model_type}", response_model=PredictionResponse)
def predict(dataset: str, model_type: str, request: PredictionRequest):
    pipeline = get_deserialized_artifact_from_ray(dataset, model_type, "pipeline")
    required_features = get_deserialized_artifact_from_ray(dataset, model_type, "feature_names")

    if not pipeline or not required_features:
        raise HTTPException(status_code=404, detail=f"Modelo o feature_names no encontrados para {dataset}/{model_type}")

    try:
        input_df = pd.DataFrame([request.features])
        input_df = input_df[required_features] # Asegurar el orden de las columnas

        prediction = pipeline.predict(input_df)
        prediction_value = prediction[0].item() if hasattr(prediction[0], 'item') else prediction[0]
        
        return PredictionResponse(prediction=prediction_value, model_details={"dataset": dataset, "model_type": model_type})
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Falta la característica requerida: {e}. Se requieren: {required_features}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error durante la predicción: {e}")