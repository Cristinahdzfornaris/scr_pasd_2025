# api.py
import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, List, Any
import uvicorn

# --- Configuración ---
BASE_MODEL_DIR = "/app/models_output"
MODELS_CACHE = {}  # Caché para almacenar los modelos cargados, métricas y nombres de características

# --- Modelo de Datos para la API (Pydantic) ---
class PredictionRequest(BaseModel):
    # Permite un diccionario flexible donde las claves son nombres de características y los valores son números
    features: Dict[str, float]

class PredictionResponse(BaseModel):
    prediction: Any
    model_details: Dict[str, str]

# --- Inicialización de la Aplicación FastAPI ---
app = FastAPI(
    title="Plataforma de Aprendizaje Supervisado Distribuido",
    description="API para servir modelos entrenados con Ray y Scikit-Learn.",
    version="1.0.0"
)

# --- Lógica de Carga de Modelos ---
def scan_and_load_models():
    """
    Escanea el directorio de modelos, carga los pipelines, características y métricas en caché.
    Esta función implementa el "autodescubrimiento" de modelos.
    """
    print("Iniciando escaneo de modelos...")
    if not os.path.exists(BASE_MODEL_DIR):
        print(f"Advertencia: El directorio de modelos '{BASE_MODEL_DIR}' no existe. No se cargarán modelos.")
        return

    for dataset_id in os.listdir(BASE_MODEL_DIR):
        dataset_path = os.path.join(BASE_MODEL_DIR, dataset_id)
        if os.path.isdir(dataset_path):
            for model_type in os.listdir(dataset_path):
                model_path = os.path.join(dataset_path, model_type)
                if os.path.isdir(model_path):
                    pipeline_file = os.path.join(model_path, "best_pipeline.joblib")
                    features_file = os.path.join(model_path, "feature_names.joblib")
                    metrics_file = os.path.join(model_path, "metrics.json")
                    
                    if os.path.exists(pipeline_file) and os.path.exists(features_file):
                        print(f"Cargando modelo: Dataset='{dataset_id}', Tipo='{model_type}'")
                        try:
                            pipeline = joblib.load(pipeline_file)
                            feature_names = joblib.load(features_file)
                            
                            if dataset_id not in MODELS_CACHE:
                                MODELS_CACHE[dataset_id] = {}
                            
                            MODELS_CACHE[dataset_id][model_type] = {
                                "pipeline": pipeline,
                                "feature_names": feature_names,
                                "pipeline_path": pipeline_file
                            }
                        except Exception as e:
                            print(f"Error al cargar el modelo '{model_type}' para el dataset '{dataset_id}': {e}")
                    else:
                        print(f"Advertencia: Faltan artefactos para el modelo '{model_type}' en '{dataset_id}'. Se omitirá.")
    print("Escaneo de modelos completado.")
    print(f"Modelos cargados en caché: {list_loaded_models()}")

def list_loaded_models():
    """Devuelve una lista estructurada de los modelos en caché."""
    summary = {}
    for dataset, models in MODELS_CACHE.items():
        summary[dataset] = {
            model_type: {
                # AQUÍ ESTÁ EL CAMBIO: Convertimos el array de NumPy a una lista de Python
                "features_required": details["feature_names"].tolist() if hasattr(details["feature_names"], 'tolist') else details["feature_names"]
            } for model_type, details in models.items()
        }
    return summary

# --- Evento de Inicio de la API ---
@app.on_event("startup")
def startup_event():
    """Al iniciar la API, escanea y carga los modelos."""
    scan_and_load_models()

# --- Endpoints de la API ---
@app.get("/", summary="Endpoint de Bienvenida")
def read_root():
    """Endpoint raíz para verificar que la API está funcionando."""
    return {"message": "Bienvenido a la API de servicio de modelos. Usa /docs para ver la documentación."}

@app.get("/models", summary="Listar Modelos Disponibles")
def get_models():
    """
    Devuelve una lista de todos los modelos cargados y disponibles para predicción,
    junto con las características que cada uno requiere.
    """
    if not MODELS_CACHE:
        raise HTTPException(status_code=404, detail="No se encontraron modelos. Asegúrate de que el proceso de entrenamiento se haya completado.")
    return list_loaded_models()

@app.post("/predict/{dataset}/{model_type}", response_model=PredictionResponse, summary="Realizar una Predicción")
def predict(dataset: str, model_type: str, request: PredictionRequest):
    """
    Realiza una predicción utilizando un modelo específico para un dataset.
    El cuerpo de la solicitud debe ser un JSON con una clave "features" que contenga
    un diccionario de `nombre_caracteristica: valor`.
    """
    if dataset not in MODELS_CACHE or model_type not in MODELS_CACHE[dataset]:
        raise HTTPException(status_code=404, detail=f"Modelo no encontrado. Modelos disponibles: {list_loaded_models()}")

    model_info = MODELS_CACHE[dataset][model_type]
    pipeline = model_info["pipeline"]
    required_features = model_info["feature_names"]
    
    # Validar que todas las características requeridas están en la solicitud
    if not all(feature in request.features for feature in required_features):
        raise HTTPException(status_code=400, detail=f"Faltan características. Se requieren: {required_features}")

    # Crear un DataFrame con el orden correcto de las columnas
    try:
        input_df = pd.DataFrame([request.features], columns=required_features)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al crear el DataFrame de entrada: {e}")

    # Realizar la predicción
    try:
        prediction_result = pipeline.predict(input_df)
        # Convertir a un tipo nativo de Python si es un array de numpy
        prediction_value = prediction_result[0].item() if hasattr(prediction_result[0], 'item') else prediction_result[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error durante la predicción: {e}")

    return PredictionResponse(
        prediction=prediction_value,
        model_details={"dataset": dataset, "model_type": model_type}
    )

if __name__ == "__main__":
    # Esto permite ejecutar la API directamente para pruebas locales
    uvicorn.run(app, host="0.0.0.0", port=8000)