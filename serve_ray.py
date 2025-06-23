# serve_ray.py
import os
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from ray import serve
import pandas as pd
import traceback
import logging # Para logging más detallado

# --- Configuración del Logger ---
# (Puedes configurar esto de forma más elaborada si lo necesitas)
logger = logging.getLogger("ray.serve") # Usar el logger de Ray Serve
logger.setLevel(logging.INFO) 
# Si quieres que los logs de tu aplicación también vayan a la consola de Ray Serve:
# console_handler = logging.StreamHandler()
# console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
# logger.addHandler(console_handler)


# --- Configuración ---
BASE_MODEL_DIR = "/app/models_output"

# --- Modelos Pydantic para la API ---
class PredictionInput(BaseModel):
    features: list[float]

class ModelInfo(BaseModel):
    dataset_name: str
    model_type: str
    model_path: str
    feature_names_path: str = None
    metrics_path: str = None # Para el path de las métricas
    status: str
    error_message: str = None
    accuracy: float = None # Para mostrar la precisión en /models

# Crear una instancia de la aplicación FastAPI
app_fastapi = FastAPI(
    title="Plataforma de Aprendizaje Supervisado Distribuido - API",
    version="1.0.1", # Actualizado
    description="API para interactuar con modelos de ML entrenados y desplegados."
)

# --- Ray Serve Deployment ---
@serve.deployment(
    name="ModelServingDeployment",
    num_replicas=1,
)
@serve.ingress(app_fastapi)
class ModelServer:
    def __init__(self):
        self.pipelines: dict[str, any] = {}
        self.feature_names_map: dict[str, list[str]] = {}
        self.model_metrics: dict[str, dict] = {} # Para almacenar métricas cargadas
        self.available_models: list[ModelInfo] = []
        self._load_available_models()

    def _load_available_models(self):
        logger.info(f"Servidor: Buscando modelos disponibles en: {BASE_MODEL_DIR}")
        self.available_models = []
        self.pipelines = {}
        self.feature_names_map = {}
        self.model_metrics = {}

        if not os.path.exists(BASE_MODEL_DIR):
            logger.warning(f"Servidor: El directorio base de modelos {BASE_MODEL_DIR} no existe.")
            return

        for dataset_name in os.listdir(BASE_MODEL_DIR):
            dataset_path = os.path.join(BASE_MODEL_DIR, dataset_name)
            if os.path.isdir(dataset_path):
                for model_type in os.listdir(dataset_path):
                    model_type_path = os.path.join(dataset_path, model_type)
                    if not os.path.isdir(model_type_path): continue

                    pipeline_filename = "best_pipeline.joblib"
                    feature_names_filename = "feature_names.joblib"
                    metrics_filename = "metrics.json" # Nombre del archivo de métricas
                    
                    pipeline_path = os.path.join(model_type_path, pipeline_filename)
                    feature_names_path_full = os.path.join(model_type_path, feature_names_filename)
                    metrics_path_full = os.path.join(model_type_path, metrics_filename)
                    
                    model_key = f"{dataset_name}__{model_type}"
                    accuracy_val = None
                    
                    # Cargar métricas si existen
                    if os.path.exists(metrics_path_full):
                        try:
                            with open(metrics_path_full, 'r') as f:
                                metrics_data = json.load(f)
                            self.model_metrics[model_key] = metrics_data
                            accuracy_val = metrics_data.get('accuracy')
                            logger.info(f"Servidor: Métricas cargadas para {model_key}")
                        except Exception as e:
                            logger.error(f"Servidor: Error al cargar métricas para {model_key} desde {metrics_path_full}: {e}")
                    
                    current_model_info = ModelInfo(
                        dataset_name=dataset_name, model_type=model_type,
                        model_path=pipeline_path, 
                        feature_names_path=feature_names_path_full if os.path.exists(feature_names_path_full) else None,
                        metrics_path=metrics_path_full if os.path.exists(metrics_path_full) else None,
                        status="no_encontrado", accuracy=accuracy_val
                    )

                    if os.path.exists(pipeline_path):
                        try:
                            self.pipelines[model_key] = joblib.load(pipeline_path)
                            current_model_info.status = "pipeline_cargado"
                            logger.info(f"Servidor: Pipeline cargado - {model_key}")

                            if os.path.exists(feature_names_path_full):
                                self.feature_names_map[model_key] = joblib.load(feature_names_path_full)
                                current_model_info.status = "completo_cargado"
                                logger.info(f"Servidor:   Feature names cargados para {model_key}")
                            else:
                                current_model_info.status = "pipeline_cargado_sin_features"
                                logger.warning(f"Servidor:   Archivo de feature_names no encontrado para {model_key}.")
                        except Exception as e:
                            error_msg = f"Error al cargar {model_key}: {e}"; logger.error(f"Servidor: {error_msg}")
                            current_model_info.status = "error_carga"; current_model_info.error_message = str(e)
                            if model_key in self.pipelines: del self.pipelines[model_key]
                            if model_key in self.feature_names_map: del self.feature_names_map[model_key]
                    else:
                        current_model_info.status = "pipeline_no_encontrado"
                        logger.warning(f"Servidor: Archivo de pipeline no encontrado para {model_key} en {pipeline_path}")
                    self.available_models.append(current_model_info)
        
        if not self.pipelines: logger.warning("Servidor: No se cargaron pipelines válidos.")
        else: logger.info(f"Servidor: Carga de modelos completada. Pipelines: {list(self.pipelines.keys())}")

    @app_fastapi.post("/reload_models", tags=["Management"], summary="Re-escanea y carga modelos")
    async def reload_models_endpoint(self):
        self._load_available_models()
        return {"message": "Recarga de modelos completada.", "loaded_pipelines_count": len(self.pipelines),
                "models_status": [model.model_dump() for model in self.available_models]}

    @app_fastapi.get("/models", response_model=list[ModelInfo], tags=["Management"], summary="Lista modelos")
    async def list_models_endpoint(self):
        if not self.available_models and os.path.exists(BASE_MODEL_DIR):
            logger.info("Servidor: /models - lista vacía, intentando recarga..."); self._load_available_models()
        return self.available_models

    @app_fastapi.get("/models/{dataset_name}/{model_type}/metrics", tags=["Management"], summary="Obtiene métricas de un modelo específico")
    async def get_model_metrics_endpoint(self, dataset_name: str, model_type: str):
        model_key = f"{dataset_name}__{model_type}"
        if model_key in self.model_metrics:
            return self.model_metrics[model_key]
        else:
            # Intentar cargar si no está en caché pero el archivo existe
            metrics_path_full = os.path.join(BASE_MODEL_DIR, dataset_name, model_type, "metrics.json")
            if os.path.exists(metrics_path_full):
                try:
                    with open(metrics_path_full, 'r') as f:
                        metrics_data = json.load(f)
                    self.model_metrics[model_key] = metrics_data # Cache it
                    return metrics_data
                except Exception as e:
                    raise HTTPException(status_code=500, detail=f"Error al cargar metrics.json para {model_key}: {e}")
            raise HTTPException(status_code=404, detail=f"Métricas para el modelo '{model_key}' no encontradas.")


    @app_fastapi.post("/predict/{dataset_name}/{model_type}", tags=["Predictions"], summary="Realiza una predicción")
    async def predict_specific_model_endpoint(self, dataset_name: str, model_type: str, data: PredictionInput):
        model_key = f"{dataset_name}__{model_type}"
        start_time = time.time()

        if model_key not in self.pipelines:
            raise HTTPException(status_code=404, detail=f"Pipeline '{model_key}' no encontrado.")

        pipeline = self.pipelines[model_key]
        
        try:
            input_data_for_prediction = None
            if model_key in self.feature_names_map:
                current_feature_names = self.feature_names_map[model_key]
                if len(data.features) != len(current_feature_names):
                    raise HTTPException(status_code=400, detail=f"Incorrect number of features. Expected {len(current_feature_names)} ({', '.join(current_feature_names)}) for '{model_key}', got {len(data.features)}.")
                input_df = pd.DataFrame([data.features], columns=current_feature_names)
                input_data_for_prediction = input_df
            else:
                logger.warning(f"Servidor: No feature_names for {model_key}. Predicting with raw array.")
                input_data_for_prediction = [data.features]
            
            prediction_result = pipeline.predict(input_data_for_prediction)
            latency_ms = (time.time() - start_time) * 1000
            logger.info(f"PREDICT_SUCCESS - Model: {model_key}, Latency: {latency_ms:.2f}ms")
            return {"dataset": dataset_name, "model": model_type, "prediction": prediction_result.tolist(), "latency_ms": latency_ms}
        
        except ValueError as ve:
            latency_ms = (time.time() - start_time) * 1000
            logger.error(f"PREDICT_FAIL - Model: {model_key}, Latency: {latency_ms:.2f}ms, ValueError: {ve}")
            traceback.print_exc()
            raise HTTPException(status_code=400, detail=f"ValueError during prediction: {ve}")
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            logger.error(f"PREDICT_FAIL - Model: {model_key}, Latency: {latency_ms:.2f}ms, Exception: {e}")
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Internal error: {e}")

    @app_fastapi.get("/health", tags=["Healthcheck"], summary="Verifica estado del servidor")
    async def health_check_endpoint(self):
        loaded_ok = len(self.pipelines) > 0
        msg = "Servidor activo." if loaded_ok else "Servidor activo pero SIN pipelines cargados."
        if loaded_ok: return {"status": "ok", "message": msg, "loaded_pipelines_count": len(self.pipelines)}
        else: raise HTTPException(status_code=503, detail={"status": "error", "message": msg, "loaded_pipelines_count": 0})

deployment_graph = ModelServer.bind()