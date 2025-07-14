# management_api.py

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Response
import pandas as pd
import ray
import os
import train
import traceback
from contextlib import asynccontextmanager
from typing import List
import json
import time
import pickle

# --- Lifespan Manager (Con Carga de Modelos Persistentes) ---
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
            raise RuntimeError("!!! ERROR CRÍTICO: El ModelRegistryActor no estuvo disponible.")

        # --- NUEVO: Cargar modelos desde el disco al arrancar ---
        print(">>> LIFESPAN: Cargando modelos persistentes al registro en memoria...")
        actor = ray.get_actor("model_registry", namespace="mi_plataforma")
        base_model_dir = "/app/persistent_models"
        if os.path.isdir(base_model_dir):
            for filename in os.listdir(base_model_dir):
                if filename.endswith(".pkl"):
                    try:
                        parts = filename.replace(".pkl", "").rsplit("_", 1)
                        if len(parts) == 2:
                            dataset_name, model_type = parts
                            file_path = os.path.join(base_model_dir, filename)
                            with open(file_path, "rb") as f:
                                result_dictionary = pickle.load(f)
                            
                            actor.register_model.remote(dataset_name, model_type, result_dictionary)
                            print(f"--- Modelo persistente cargado: {dataset_name}/{model_type}")
                    except Exception as e:
                        print(f"--- Error cargando modelo persistente {filename}: {e}")
        
    except Exception as e:
        print(f"!!! LIFESPAN (Management API) ERROR DURANTE EL ARRANQUE: {e} !!!")
        traceback.print_exc()
        raise e

    yield

    if ray.is_initialized():
        ray.shutdown()

app = FastAPI(title="API de Gestión y Registro de Modelos", version="2.1.0", lifespan=lifespan)

# --- Endpoints de Gestión (Sin cambios) ---
@app.post("/datasets/train_batch", summary="Subir y Entrenar un Lote de Datasets")
async def upload_and_train_batch(
    configs: str = Form(...),
    models_to_train: List[str] = Form(...),
    files: List[UploadFile] = File(...)
):
    # (El código de este endpoint es correcto, no se necesita cambiar)
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
    # (El código de este endpoint es correcto, no se necesita cambiar)
    try:
        registry_actor = ray.get_actor("model_registry", namespace="mi_plataforma")
        success = await registry_actor.delete_dataset_models.remote(dataset_name)
        if success:
            # Opcional: Borrar también los archivos del disco
            base_model_dir = "/app/persistent_models"
            for filename in os.listdir(base_model_dir):
                if filename.startswith(f"{dataset_name}_") and filename.endswith(".pkl"):
                    os.remove(os.path.join(base_model_dir, filename))
            return {"message": f"Modelos para '{dataset_name}' eliminados del registro y del disco."}
        else:
            raise HTTPException(status_code=404, detail=f"No se encontraron modelos para el dataset '{dataset_name}'.")
    except ValueError:
        raise HTTPException(status_code=503, detail="ModelRegistryActor no disponible.")

# --- Endpoint de Salud (Sin cambios) ---
@app.get("/health", summary="Comprobación de Salud", status_code=200)
async def health_check():
    return {"status": "ok"}

# --- Endpoints de Registro (MODIFICADOS) ---
@app.get("/registry/models", summary="[REGISTRY] Listar todos los modelos disponibles")
async def registry_list_models():
    try:
        registry_actor = ray.get_actor("model_registry", namespace="mi_plataforma")
        models_details = await registry_actor.list_models_details.remote()
        return models_details
    except ValueError:
        raise HTTPException(status_code=503, detail="ModelRegistryActor no disponible.")

@app.get("/registry/artifacts/{dataset_name}/{model_type}/{artifact_name}", summary="[REGISTRY] Obtener un artefacto de modelo")
async def registry_get_artifact(dataset_name: str, model_type: str, artifact_name: str):
    try:
        # 1. Intenta obtener el artefacto del actor en memoria (rápido)
        registry_actor = ray.get_actor("model_registry", namespace="mi_plataforma")
        artifacts_dict = await registry_actor.get_model_artifacts.remote(dataset_name, model_type)
        
        if not artifacts_dict:
            # 2. Si no está en memoria, intenta cargarlo desde el disco (fallback)
            model_path = f"/app/persistent_models/{dataset_name}_{model_type}.pkl"
            print(f"Artefacto no en memoria. Intentando cargar desde disco: {model_path}")
            if not os.path.exists(model_path):
                raise HTTPException(status_code=404, detail="Modelo no encontrado ni en memoria ni en disco.")
            with open(model_path, "rb") as f:
                artifacts_dict = pickle.load(f)

        serialized_artifact = artifacts_dict.get(artifact_name)
        
        if serialized_artifact is None:
            raise HTTPException(status_code=404, detail=f"Artefacto '{artifact_name}' no encontrado en el modelo.")
            
        return Response(content=serialized_artifact, media_type="application/octet-stream")

    except ValueError:
        raise HTTPException(status_code=503, detail="ModelRegistryActor no disponible.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno obteniendo el artefacto: {e}")