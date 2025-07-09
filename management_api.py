# management_api.py
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
import pandas as pd
import ray
import os
import train # Importamos nuestro módulo de entrenamiento

# --- App FastAPI ---
app = FastAPI(title="API de Gestión de Entrenamiento", version="1.0.0")

# --- Evento de Inicio ---
@app.on_event("startup")
def startup_event():
    ray_address = os.environ.get("RAY_ADDRESS", "auto")
    if not ray.is_initialized():
        ray.init(address=ray_address, ignore_reinit_error=True)
    print("Management API conectada a Ray.")

# --- Endpoints ---
@app.post("/datasets/{dataset_name}/train")
async def upload_and_train(
    dataset_name: str,
    target_column: str = Form(...),
    file: UploadFile = File(...)
):
    """
    Recibe un dataset CSV, lo pone en Ray y lanza un trabajo de entrenamiento distribuido.
    """
    print(f"Recibiendo dataset '{dataset_name}' con columna objetivo '{target_column}'...")
    
    try:
        df = pd.read_csv(file.file)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"No se pudo parsear el archivo CSV: {e}")

    if target_column not in df.columns:
        raise HTTPException(status_code=400, detail=f"La columna objetivo '{target_column}' no se encuentra en el dataset.")

    # Lanzar el trabajo de entrenamiento principal como una tarea Ray remota desvinculada.
    # Esto permite que la API responda inmediatamente sin esperar a que el entrenamiento termine.
    try:
        # Usamos `detached=True` para que el job siga corriendo si este script termina.
        # `run_training_job.options(name=f"training_job_{dataset_name}", get_if_exists=True).remote(...)`
        # Por simplicidad, aquí lo haremos síncrono para la demo, pero devolvemos una referencia.
        
        print(f"Lanzando el trabajo de entrenamiento para '{dataset_name}' en el clúster de Ray...")
        job_ref = train.run_complete_training.remote(dataset_name, df, target_column)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"No se pudo lanzar el trabajo de entrenamiento en Ray: {e}")

    return {
        "message": "Trabajo de entrenamiento lanzado exitosamente.",
        "dataset_name": dataset_name,
        "job_info": "Puedes monitorear el progreso en el Ray Dashboard.",
        "job_ref": str(job_ref) # Devuelve la referencia al resultado del job completo
    }

@app.delete("/models/{dataset_name}")
async def delete_models(dataset_name: str):
    """
    Le dice al ModelRegistryActor que elimine los modelos de un dataset.
    """
    try:
        registry_actor = ray.get_actor("model_registry")
        success = await registry_actor.delete_dataset_models.remote(dataset_name)
        if success:
            return {"message": f"Modelos para el dataset '{dataset_name}' eliminados del registro."}
        else:
            raise HTTPException(status_code=404, detail=f"No se encontraron modelos para el dataset '{dataset_name}'.")
    except ValueError:
        raise HTTPException(status_code=503, detail="ModelRegistryActor no disponible.")