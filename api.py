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
import train # Necesario para la clase ModelRegistryActor

# --- Modelos Pydantic ---
class PredictionRequest(BaseModel):
    features: Dict[str, float]

class PredictionResponse(BaseModel):
    prediction: Any
    model_details: Dict[str, str]

# --- Lifespan Manager para la API de Inferencia ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Código que se ejecuta al iniciar la aplicación (startup)
    print(">>> LIFESPAN (API Service): Evento de inicio iniciado <<<")
    try:
        ray_address = os.environ.get("RAY_ADDRESS", "auto")
        if not ray.is_initialized():
            print(f">>> LIFESPAN (API Service): Intentando conectar a Ray en {ray_address} con namespace 'mi_plataforma'...")
            ray.init(address=ray_address, namespace="mi_plataforma", ignore_reinit_error=True)
        
        print(">>> LIFESPAN (API Service): Conectado a Ray. Asegurando que ModelRegistryActor exista...")
        # INTENTA CREAR/OBTENER EL ACTOR. SI YA EXISTE (creado por management-api), SOLO OBTIENE LA REFERENCIA.
        # Esto elimina la condición de carrera.
        train.ModelRegistryActor.options(
            name="model_registry", 
            get_if_exists=True, 
            namespace="mi_plataforma", 
            max_restarts=-1, 
            lifetime="detached", 
            resources={"is_head_node": 1}
        ).remote()
        
        print(">>> LIFESPAN (API Service): listo. ModelRegistryActor asegurado desde API Service. <<<")
    except Exception as e:
        print(f"!!! LIFESPAN (API Service): ERROR CRÍTICO DURANTE EL STARTUP: {e} !!!")
        import traceback
        traceback.print_exc()

    yield # La aplicación se ejecuta aquí
    
    # Código de apagado
    print(">>> LIFESPAN (API Service): Apagando...")
    if ray.is_initialized():
        ray.shutdown()

# --- Inicialización de la Aplicación FastAPI ---
app = FastAPI(
    title="API de Inferencia Distribuida", 
    version="2.0.0",
    lifespan=lifespan # <--- ¡IMPORTANTE! Asignar el lifespan manager aquí.
)

# --- Funciones de Ayuda ---
@lru_cache(maxsize=16)
def get_deserialized_artifact(dataset: str, model_type: str, artifact_name: str):
    """
    Obtiene una ObjectRef del actor, la resuelve con ray.get(),
    la deserializa y la cachea.
    """
    print(f"CACHE MISS: Obteniendo de Ray: {dataset}/{model_type}/{artifact_name}")
    try:
        # Obtener el actor del namespace correcto
        registry_actor = ray.get_actor("model_registry", namespace="mi_plataforma")
        
        task_ref = ray.get(registry_actor.get_model_artifacts_ref.remote(dataset, model_type))
        if not task_ref: raise ValueError("Referencia de tarea no encontrada en el registro.")
        
        task_result = ray.get(task_ref)
        if not task_result: raise ValueError("El resultado de la tarea de entrenamiento estaba vacío (None).")

        serialized_obj = task_result[artifact_name]
        
        # Deserializar según el tipo de artefacto
        return pickle.loads(serialized_obj) if artifact_name != "metrics" else json.loads(serialized_obj.decode('utf-8'))
    except ValueError as ve: # Errores de 'get_actor'
        print(f"ERROR: No se pudo obtener el actor o la referencia del modelo. {ve}")
        raise HTTPException(status_code=503, detail=f"Servicio no listo o modelo no encontrado: {ve}")
    except Exception as e:
        print(f"ERROR obteniendo artefacto: {e}")
        # Devolver None aquí podría ser manejado por el endpoint, pero lanzar un error es más claro.
        raise HTTPException(status_code=500, detail=f"Error interno al obtener el artefacto '{artifact_name}': {e}")

# --- Endpoints ---
@app.get("/models")
async def list_models():
    """
    Consulta al ModelRegistryActor para devolver una lista de todos los modelos 
    registrados y disponibles para predicción.
    """
    try:
        registry_actor = ray.get_actor("model_registry", namespace="mi_plataforma")
        return await registry_actor.list_models_details.remote()
    except ValueError:
        # Esta excepción se lanza si get_actor no encuentra el actor
        raise HTTPException(status_code=503, detail="ModelRegistryActor no disponible. Asegúrate de que la management-api se haya iniciado correctamente.")

@app.post("/predict/{dataset}/{model_type}", response_model=PredictionResponse)
def predict(dataset: str, model_type: str, request: PredictionRequest):
    """
    Realiza una predicción utilizando un modelo específico obtenido de forma distribuida
    desde el clúster Ray.
    """
    try:
        pipeline = get_deserialized_artifact(dataset, model_type, "pipeline")
        required_features = get_deserialized_artifact(dataset, model_type, "feature_names")
    except HTTPException as e:
        # Re-lanzar las excepciones que nuestra función de ayuda ya genera
        raise e

    if not pipeline or not required_features:
        # Este caso es redundante si get_deserialized_artifact lanza excepciones, pero es seguro tenerlo.
        raise HTTPException(status_code=404, detail=f"El pipeline o los feature_names no se pudieron cargar para {dataset}/{model_type}")

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