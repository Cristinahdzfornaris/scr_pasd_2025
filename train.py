# train.py
import ray
import pickle  # Usar pickle para serializar objetos complejos como pipelines y arrays
import json
import traceback
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
import socket
import time

# --- Actor de Registro de Modelos ---
# (Este actor vivirá en el clúster Ray y será el "cerebro" que sabe dónde están los modelos)
@ray.remote
class ModelRegistryActor:
    def __init__(self):
        # El estado es un diccionario: {dataset_name: {model_type: ObjectRef_a_diccionario_de_artefactos}}
        self.registered_models = {}
        print(f"[{self.__class__.__name__}] Actor de Registro de Modelos inicializado.")

    def register_model(self, dataset_name, model_type, artifacts_dict_ref):
        """Guarda la ObjectRef que apunta al diccionario de artefactos de un modelo."""
        if dataset_name not in self.registered_models:
            self.registered_models[dataset_name] = {}
        self.registered_models[dataset_name][model_type] = artifacts_dict_ref
        print(f"[{self.__class__.__name__}] Modelo registrado: {dataset_name}/{model_type}")
        return True

    def get_model_artifacts_ref(self, dataset_name, model_type):
        """Devuelve la ObjectRef que apunta al diccionario de artefactos de un modelo específico."""
        return self.registered_models.get(dataset_name, {}).get(model_type)

    def list_models_details(self):
        """Devuelve una estructura simple de los modelos disponibles (sin las ObjectRefs)."""
        details = {}
        for dataset, models in self.registered_models.items():
            model_types = list(models.keys())
            if model_types:
                details[dataset] = {"available_models": model_types}
        return details

    def delete_dataset_models(self, dataset_name):
        """Elimina todos los modelos asociados a un dataset del registro."""
        if dataset_name in self.registered_models:
            # En un sistema real, se necesitaría un manejo más complejo para liberar explícitamente los ObjectRefs
            del self.registered_models[dataset_name]
            print(f"[{self.__class__.__name__}] Modelos para el dataset '{dataset_name}' han sido eliminados del registro.")
            return True
        return False

# --- Tarea de Entrenamiento Remota (Modificada para devolver artefactos) ---
@ray.remote(num_cpus=1)
def train_and_serialize_model(
    data_df_input, 
    target_column_name: str, 
    model_config: dict, 
    param_grid_config: dict,
    dataset_id: str,
    feature_names: list,
    class_names_for_metrics: list
):
    hostname_str = socket.gethostname()
    log_prefix = f"Worker [{hostname_str}]"
    
    # Manejar la entrada (puede ser DataFrame u ObjectRef)
    data_df = ray.get(data_df_input) if isinstance(data_df_input, ray.ObjectRef) else data_df_input

    print(f"{log_prefix} - Iniciando entrenamiento para {model_config['type']} en {dataset_id}...")

    try:
        X = data_df.drop(columns=[target_column_name]); y = data_df[target_column_name]
        
        # Lógica de train/test split
        stratify_option = y if len(y.unique()) > 1 and all(y.value_counts() >= 3) else None
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=stratify_option)

        # Creación del pipeline
        preprocessor = Pipeline([('imputer', SimpleImputer(strategy='mean')), ('scaler', StandardScaler())])
        if model_config['type'] == "logistic_regression": model_instance = LogisticRegression(**model_config['params'])
        elif model_config['type'] == "decision_tree": model_instance = DecisionTreeClassifier(**model_config['params'])
        else: model_instance = RandomForestClassifier(**model_config['params'])
        
        full_pipeline = Pipeline([('preprocessor', preprocessor), ('classifier', model_instance)])
        param_grid = {f'classifier__{k}': v for k, v in param_grid_config.items()}

        # GridSearchCV
        grid_search = GridSearchCV(estimator=full_pipeline, param_grid=param_grid, cv=3, n_jobs=1)
        start_fit_time = time.time()
        grid_search.fit(X_train, y_train)
        fit_duration = time.time() - start_fit_time

        # Obtener los mejores artefactos y métricas
        best_pipeline_obj = grid_search.best_estimator_
        y_pred = best_pipeline_obj.predict(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred, target_names=class_names_for_metrics, output_dict=True, zero_division=0),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'best_hyperparameters': grid_search.best_params_,
            'training_duration_sec': fit_duration
        }
        
        print(f"{log_prefix} - Entrenamiento para {model_config['type']} en {dataset_id} completado. Accuracy: {metrics['accuracy']:.4f}")
        
        # Serializar los artefactos para devolverlos
        # Estos se colocarán en el Object Store de Ray automáticamente
        return {
            "pipeline": pickle.dumps(best_pipeline_obj),
            "feature_names": pickle.dumps(feature_names),
            "metrics": json.dumps(metrics).encode('utf-8') # JSON a bytes
        }
    except Exception as e:
        print(f"{log_prefix} - ERROR en la tarea de entrenamiento para {dataset_id}/{model_config['type']}: {e}")
        traceback.print_exc()
        return None

# --- Función Orquestadora Principal (Diseñada para ser llamada por la API de Gestión) ---
@ray.remote
def run_complete_training_job(dataset_name: str, df: pd.DataFrame, target_column: str):
    """
    Orquesta el ciclo completo de entrenamiento para un nuevo dataset.
    """
    print(f"ORQUESTADOR: Iniciando trabajo de entrenamiento completo para dataset '{dataset_name}'...")
    
    # Crear o obtener el actor de registro, fijándolo al nodo head
    registry_actor = ModelRegistryActor.options(
        name="model_registry", get_if_exists=True, max_restarts=-1, resources={"is_head_node": 1}
    ).remote()
    
    feature_names = [col for col in df.columns if col != target_column]
    class_names = [str(c) for c in sorted(df[target_column].unique())]
    
    df_ref = ray.put(df) # Poner el DataFrame principal en el Object Store

    # Configuraciones de modelos a entrenar
    model_configurations = [
        {'type': "logistic_regression", 'params': {'max_iter': 200, 'random_state': 42}, 'param_grid': {'C': [0.1, 1.0, 10.0]}},
        {'type': "decision_tree", 'params': {'random_state': 42}, 'param_grid': {'max_depth': [3, 5, None], 'min_samples_split': [2, 5, 10]}},
        {'type': "random_forest", 'params': {'n_jobs': 1, 'random_state': 42}, 'param_grid': {'n_estimators': [50, 100], 'max_depth': [5, 10]}}
    ]

    task_refs_to_register = {}

    print(f"ORQUESTADOR: Lanzando {len(model_configurations)} tareas de entrenamiento para '{dataset_name}'...")
    for model_config in model_configurations:
        model_type = model_config['type']
        
        # Lanzar la tarea que entrena Y serializa
        task_ref = train_and_serialize_model.remote(
            df_ref, target_column, model_config, model_config['param_grid'], dataset_name, feature_names, class_names
        )
        # Guardar la referencia a la tarea (que contendrá el diccionario de artefactos)
        task_refs_to_register[model_type] = task_ref

    # Registrar los futuros resultados en el actor
    # Pasamos el diccionario de {model_type: ObjectRef_a_resultados}
    for model_type, task_ref in task_refs_to_register.items():
        # El actor guardará la referencia a la tarea que, una vez completada, contendrá los artefactos
        registry_actor.register_model.remote(dataset_name, model_type, task_ref)

    # Esperar a que todas las tareas terminen (opcional, el registro ya se hizo)
    # Esto es útil para saber cuándo ha terminado el "trabajo" completo
    try:
        results = ray.get(list(task_refs_to_register.values()))
        successful_count = sum(1 for r in results if r is not None)
        print(f"ORQUESTADOR: Trabajo de entrenamiento para '{dataset_name}' completado. {successful_count}/{len(results)} modelos entrenados exitosamente.")
        return f"Completado. {successful_count} modelos entrenados para {dataset_name}."
    except Exception as e:
        print(f"ORQUESTADOR: Error al esperar los resultados del entrenamiento para '{dataset_name}': {e}")
        return f"Error durante el entrenamiento para {dataset_name}."

# Esto permite que el archivo sea importado sin ejecutar código automáticamente.
# Ya no es el punto de entrada principal.
if __name__ == "__main__":
    print("Este script contiene la lógica de entrenamiento y está diseñado para ser importado por la API de gestión.")