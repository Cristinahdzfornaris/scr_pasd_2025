# management_api.py

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Response
import pandas as pd
import ray
import os
import train  # Importar nuestro módulo de entrenamiento
import traceback
from contextlib import asynccontextmanager
from typing import List
import json
import time

# --- Lifespan Manager (Se mantiene igual, sigue siendo el responsable del actor) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    print(">>> LIFESPAN (Management API): Iniciando...")
    try:
        ray_address = os.environ.get("RAY_ADDRESS", "auto")
        if not ray.is_initialized():
            ray.init(address=ray_address, namespace="mi_plataforma", ignore_reinit_error=True)

        train.ModelRegistryActor.options(
            name="model_registry",
            get_if_exists=True,
            namespace="mi_plataforma",
            lifetime="detached",
            max_restarts=-1,
        ).remote()
        print(">>> LIFESPAN (Management API): Solicitud de creación de ModelRegistryActor enviada.")

        actor_ready = False
        for i in range(30):
            try:
                ray.get_actor("model_registry", namespace="mi_plataforma")
                print(f">>> LIFESPAN (Management API): ¡ModelRegistryActor está listo! (Intento {i+1})")
                actor_ready = True
                break
            except ValueError:
                print(f">>> LIFESPAN (Management API): Esperando que ModelRegistryActor esté disponible... (Intento {i+1})")
                time.sleep(1)

        if not actor_ready:
            error_message = "!!! ERROR CRÍTICO: El ModelRegistryActor no estuvo disponible. El servicio no puede iniciar."
            print(error_message)
            raise RuntimeError(error_message)

    except Exception as e:
        print(f"!!! LIFESPAN (Management API) ERROR DURANTE EL ARRANQUE: {e} !!!")
        traceback.print_exc()
        raise e

    yield

    if ray.is_initialized():
        ray.shutdown()

app = FastAPI(title="API de Gestión y Registro de Modelos", version="2.0.0", lifespan=lifespan)

# --- Endpoints de Gestión (los que ya tenías) ---
@app.post("/datasets/train_batch", summary="Subir y Entrenar un Lote de Datasets")
async def upload_and_train_batch(
    configs: str = Form(...),
    models_to_train: List[str] = Form(...),
    files: List[UploadFile] = File(...)
):
    print(f"API: Recibiendo petición de entrenamiento en lote para modelos: {models_to_train}")
    
    try:
        dataset_configs = json.loads(configs)
        uploaded_files_map = {file.filename: file for file in files}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error procesando la entrada: {e}")

    launched_jobs_info = []
    errors = []

    for config in dataset_configs:
        dataset_name = config.get("dataset_name")
        target_column = config.get("target_column")
        filename = config.get("filename")

        if not all([dataset_name, target_column, filename]) or filename not in uploaded_files_map:
            errors.append({"dataset_config": config, "error": "Configuración incompleta o archivo no encontrado."})
            continue

        file = uploaded_files_map[filename]
        
        try:
            df = pd.read_csv(file.file)
            await file.close()
            
            if target_column not in df.columns:
                errors.append({"dataset_config": config, "error": f"Columna objetivo '{target_column}' no encontrada."})
                continue

            result_message = train.run_complete_training_job(
                dataset_name, df, target_column, models_to_train
            )
            launched_jobs_info.append({"dataset_name": dataset_name, "status": result_message})

        except Exception as e:
            error_msg = f"Error al lanzar el job para '{dataset_name}': {e}"
            errors.append({"dataset_config": config, "error": error_msg})
            traceback.print_exc()

    return {
        "message": "Procesamiento de lote completado.",
        "launched_jobs": launched_jobs_info,
        "errors": errors
    }

@app.delete("/models/{dataset_name}", summary="Eliminar Modelos de un Dataset")
async def delete_models(dataset_name: str):
    try:
        registry_actor = ray.get_actor("model_registry", namespace="mi_plataforma")
        success = await registry_actor.delete_dataset_models.remote(dataset_name)
        if success:
            return {"message": f"Modelos para '{dataset_name}' eliminados del registro."}
        else:
            raise HTTPException(status_code=404, detail=f"No se encontraron modelos para el dataset '{dataset_name}'.")
    except ValueError:
        raise HTTPException(status_code=503, detail="ModelRegistryActor no disponible.")

# --- Endpoint de Salud (para Docker Compose) ---
@app.get("/health", summary="Comprobación de Salud", status_code=200)
async def health_check():
    return {"status": "ok"}

# =======================================================
# === NUEVOS ENDPOINTS PARA EXPONER EL REGISTRO (API INTERNA) ===
# =======================================================

@app.get("/registry/models", summary="[REGISTRY] Listar todos los modelos disponibles")
async def registry_list_models():
    """
    Expone la lista de modelos del actor a través de una API REST.
    Este endpoint será consumido por 'api-service'.
    """
    try:
        registry_actor = ray.get_actor("model_registry", namespace="mi_plataforma")
        models_details = await registry_actor.list_models_details.remote()
        return models_details
    except ValueError:
        raise HTTPException(status_code=503, detail="ModelRegistryActor no disponible.")

@app.get("/registry/artifacts/{dataset_name}/{model_type}/{artifact_name}", summary="[REGISTRY] Obtener un artefacto de modelo")
async def registry_get_artifact(dataset_name: str, model_type: str, artifact_name: str):
    """
    Devuelve un artefacto específico (serializado) de un modelo.
    Por ejemplo: 'pipeline', 'feature_names', 'metrics'.
    Devuelve los bytes crudos del artefacto serializado.
    """
    try:
        registry_actor = ray.get_actor("model_registry", namespace="mi_plataforma")
        
        # Obtenemos el diccionario completo de artefactos para un modelo
        artifacts_dict = await registry_actor.get_model_artifacts_ref.remote(dataset_name, model_type)
        
        if not artifacts_dict or artifact_name not in artifacts_dict:
            raise HTTPException(status_code=404, detail=f"Artefacto '{artifact_name}' no encontrado en el registro.")
            
        serialized_artifact = artifacts_dict[artifact_name]

        # Devolvemos los bytes crudos. El cliente se encargará de deserializar.
        return Response(content=serialized_artifact, media_type="application/octet-stream")

    except ValueError:
        raise HTTPException(status_code=503, detail="ModelRegistryActor no disponible.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno obteniendo el artefacto: {e}")