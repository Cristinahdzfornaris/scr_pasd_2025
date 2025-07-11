# management_api.py
import ray
import os
import traceback
import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from contextlib import asynccontextmanager

# Importar la lógica de entrenamiento y el actor desde el módulo train
import train 

# --- Lifespan Manager (La nueva forma de manejar startup/shutdown) ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gestiona los eventos de inicio y apagado de la aplicación FastAPI.
    """
    # --- CÓDIGO DE INICIO (STARTUP) ---
    print(">>> LIFESPAN: Evento de inicio para Management API... <<<")
    try:
        # La dirección se toma de la variable de entorno configurada en docker-compose.yml
        # o usa 'auto' si no está definida (para pruebas locales fuera de Docker)
        ray_address = os.environ.get("RAY_ADDRESS", "auto")
        
        # Conectarse a Ray usando un namespace para aislar esta aplicación
        if not ray.is_initialized():
            print(f">>> LIFESPAN: Intentando conectar a Ray en {ray_address} con namespace 'mi_plataforma'...")
            ray.init(
                address=ray_address, 
                namespace="mi_plataforma", # Namespace para aislar actores y trabajos
                ignore_reinit_error=True
            )
        
        print(">>> LIFESPAN: Conectado a Ray. Asegurando que ModelRegistryActor exista...")
        
        # Crear/obtener el actor de registro. Es CRUCIAL para el sistema.
        # Se fija al nodo head usando un recurso personalizado.
        train.ModelRegistryActor.options(
            name="model_registry", 
            get_if_exists=True, 
            namespace="mi_plataforma", # Usar el mismo namespace
            max_restarts=-1,         # Reiniciar indefinidamente si falla
            lifetime="detached",     # El actor sobrevive al proceso que lo creó
            resources={"is_head_node": 1}
        ).remote()
        
        print(">>> LIFESPAN: Management API lista. ModelRegistryActor asegurado. <<<")

    except Exception as e:
        print("!!! LIFESPAN: ERROR CRÍTICO DURANTE EL STARTUP !!!")
        print(f"Error: {e}")
        traceback.print_exc()
        # En un sistema real, podrías querer que la aplicación falle al iniciar si no puede conectar a Ray.
    
    yield # Aquí es donde la aplicación FastAPI se ejecuta y atiende peticiones
    
    # --- CÓDIGO DE APAGADO (SHUTDOWN) ---
    print(">>> LIFESPAN: Evento de apagado para Management API...")
    if ray.is_initialized():
        ray.shutdown()
        print(">>> LIFESPAN: Desconexión de Ray completada.")

# --- Inicialización de la Aplicación FastAPI ---
app = FastAPI(
    title="API de Gestión de Entrenamiento", 
    description="API para subir datasets, iniciar trabajos de entrenamiento distribuidos y gestionar modelos.",
    version="1.0.0",
    lifespan=lifespan # Asignar el lifespan manager
)

# --- Endpoints de la API ---

@app.post("/datasets/{dataset_name}/train", summary="Subir Dataset y Entrenar")
async def upload_and_train(
    dataset_name: str,
    target_column: str = Form(..., description="El nombre exacto de la columna en el CSV que se debe predecir."),
    file: UploadFile = File(..., description="El archivo del dataset en formato CSV.")
):
    """
    Recibe un dataset CSV, lo pone en Ray y lanza un trabajo de entrenamiento distribuido.
    El entrenamiento se ejecuta como una tarea Ray remota y asíncrona.
    """
    print(f"API: Recibiendo petición para entrenar el dataset '{dataset_name}' con columna objetivo '{target_column}'...")
    
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Formato de archivo inválido. Por favor, sube un archivo CSV.")

    try:
        # Cargar el CSV en un DataFrame de Pandas
        df = pd.read_csv(file.file)
        print(f"API: CSV '{file.filename}' cargado, shape: {df.shape}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"No se pudo parsear el archivo CSV: {e}")

    if target_column not in df.columns:
        raise HTTPException(status_code=400, detail=f"La columna objetivo '{target_column}' no se encuentra en las columnas del dataset: {df.columns.tolist()}")

    try:
        print(f"API: Lanzando el trabajo de entrenamiento para '{dataset_name}' en el clúster de Ray...")
        
        # Lanzar el trabajo de entrenamiento completo como una tarea Ray remota
        # Esta llamada devuelve inmediatamente una ObjectRef (o ClientObjectRef)
        job_ref = train.run_complete_training_job.remote(dataset_name, df, target_column)
        
        print(f"API: Trabajo de Ray lanzado exitosamente. Ref: {job_ref}")

    except Exception as e:
        print(f"API: ERROR al lanzar el trabajo de entrenamiento en Ray: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"No se pudo lanzar el trabajo de entrenamiento en Ray: {e}")

    return {
        "message": "Trabajo de entrenamiento lanzado exitosamente.",
        "dataset_name": dataset_name,
        "job_info": "El entrenamiento se ejecuta en segundo plano. Monitorea el progreso en el Ray Dashboard o los logs.",
        "job_ref_str": str(job_ref)
    }

@app.delete("/models/{dataset_name}", summary="Eliminar Modelos de un Dataset")
async def delete_models(dataset_name: str):
    """
    Le dice al ModelRegistryActor que elimine del registro todos los modelos 
    asociados a un dataset específico.
    """
    print(f"API: Recibiendo petición para eliminar modelos del dataset '{dataset_name}'...")
    try:
        # Obtener una referencia al actor que ya debería existir
        registry_actor = ray.get_actor("model_registry", namespace="mi_plataforma")
        
        # Llamar al método del actor de forma remota
        success = await registry_actor.delete_dataset_models.remote(dataset_name)
        
        if success:
            return {"message": f"Modelos para el dataset '{dataset_name}' eliminados del registro."}
        else:
            raise HTTPException(status_code=404, detail=f"No se encontraron modelos para el dataset '{dataset_name}' en el registro.")
            
    except ValueError:
        # Este error ocurre si ray.get_actor no encuentra el actor
        raise HTTPException(status_code=503, detail="Servicio no disponible: ModelRegistryActor no encontrado en el clúster Ray.")
    except Exception as e:
        print(f"API: ERROR al eliminar modelos: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error interno del servidor: {e}")



@app.get("/debug/actor_status", tags=["Debug"])
async def get_actor_status():
    """
    Endpoint de depuración para verificar el estado del ModelRegistryActor 
    desde la perspectiva de la API de Gestión.
    """
    try:
        # Intenta obtener el actor usando el mismo nombre y namespace
        actor = ray.get_actor("model_registry", namespace="mi_plataforma")
        
        # Llama a un método del actor para asegurarte de que responde
        model_list = await actor.list_models_details.remote()
        
        return {
            "actor_found": True,
            "actor_name": "model_registry",
            "actor_status": "ALIVE_AND_RESPONDING",
            "registered_models": model_list
        }
    except ValueError:
        # Esto ocurre si get_actor no encuentra el actor
        return {
            "actor_found": False,
            "actor_name": "model_registry",
            "actor_status": "NOT_FOUND",
            "detail": "ray.get_actor('model_registry') lanzó ValueError."
        }
    except Exception as e:
        return {
            "actor_found": "unknown",
            "actor_status": "ERROR",
            "detail": f"Ocurrió un error inesperado: {str(e)}"
        }