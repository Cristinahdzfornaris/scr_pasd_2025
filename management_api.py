# management_api.py
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
import pandas as pd
import ray
import os
import train # Importar nuestro módulo de entrenamiento
import traceback
from contextlib import asynccontextmanager
from typing import List
import json
import time
# --- Lifespan Manager (CORREGIDO Y ROBUSTO) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    print(">>> LIFESPAN (Management API): Iniciando...")
    try:
        # 1. Conectar a Ray
        ray_address = os.environ.get("RAY_ADDRESS", "auto")
        if not ray.is_initialized():
            ray.init(address=ray_address, namespace="mi_plataforma", ignore_reinit_error=True)
        
        # 2. Solicitar la creación del actor con opciones de robustez
        train.ModelRegistryActor.options(
            name="model_registry",
            get_if_exists=True,
            namespace="mi_plataforma",
            lifetime="detached",
            max_restarts=-1,
            
        ).remote()
        print(">>> LIFESPAN (Management API): Solicitud de creación de ModelRegistryActor enviada.")

        # 3. Bucle de verificación para confirmar que el actor está vivo
        actor_ready = False
        # Intentar por hasta 30 segundos
        for i in range(30):
            try:
                # El intento de obtener el actor sirve como verificación
                ray.get_actor("model_registry", namespace="mi_plataforma")
                print(f">>> LIFESPAN (Management API): ¡ModelRegistryActor está listo y disponible! (Intento {i+1})")
                actor_ready = True
                break
            except ValueError:
                print(f">>> LIFESPAN (Management API): Esperando que ModelRegistryActor esté disponible... (Intento {i+1})")
                time.sleep(1)
        
        if not actor_ready:
            # Si el actor no aparece después del tiempo de espera, se lanza un error y el servidor no arrancará
            error_message = "!!! ERROR CRÍTICO: El ModelRegistryActor no estuvo disponible después del tiempo de espera. El servicio no puede iniciar."
            print(error_message)
            raise RuntimeError(error_message)



    except Exception as e:
        print(f"!!! LIFESPAN (Management API) ERROR DURANTE EL ARRANQUE: {e} !!!")
        traceback.print_exc()
        raise e # Relanzar la excepción para detener el inicio del servidor
    
    # La API sólo se ejecuta si todo lo anterior tuvo éxito
    yield
    
    # Código de limpieza al apagar
    if ray.is_initialized():
        ray.shutdown()

# --- App FastAPI (asegúrate que use el lifespan) ---
app = FastAPI(title="API de Gestión de Entrenamiento", version="1.2.0", lifespan=lifespan)

# En management_api.py

# ... (tus otros imports y la app FastAPI) ...

@app.get("/debug/registry_state", summary="[DEBUG] Muestra el estado interno del registro de modelos")
async def debug_get_registry_state():
    """
    Endpoint de depuración para ver el contenido completo del ModelRegistryActor.
    ¡NO USAR EN PRODUCCIÓN!
    """
    try:
        registry_actor = ray.get_actor("model_registry", namespace="mi_plataforma")
        # Necesitamos un nuevo método en el actor que nos devuelva su estado interno
        # para inspección.
        internal_state = await registry_actor.get_internal_state.remote()
        return internal_state
    except ValueError:
        raise HTTPException(status_code=503, detail="ModelRegistryActor no disponible.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al inspeccionar el actor: {e}")
# --- Endpoints ---
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
            await file.close() # Importante cerrar el archivo subido
            
            if target_column not in df.columns:
                errors.append({"dataset_config": config, "error": f"Columna objetivo '{target_column}' no encontrada."})
                continue

            # --- CORRECCIÓN AQUÍ ---
            # Llamar a la función de entrenamiento directamente.
            # Esta llamada es bloqueante pero rápida, ya que solo lanza tareas a Ray.
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

# Endpoint para eliminar, se mantiene igual
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