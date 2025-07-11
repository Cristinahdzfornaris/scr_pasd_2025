# train.py
import ray
import os
import pickle
import json
import traceback
import pandas as pd
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

# --- Actor de Registro de Modelos (con lógica robusta) ---
@ray.remote
class ModelRegistryActor:
    def __init__(self):
        # El estado es: {dataset_name: {model_type: ObjectRef_que_apunta_a_diccionario_de_artefactos}}
        self.registered_models = {}
        print(f"[{self.__class__.__name__}] Actor de Registro de Modelos inicializado.")

    def register_model(self, dataset_name, model_type, training_result):
        """
        Registra un modelo. Si recibe una ObjectRef, la guarda.
        Si recibe el diccionario de resultados directamente, lo pone en el
        Object Store y guarda la nueva referencia. Esto hace al actor robusto.
        """
        if dataset_name not in self.registered_models:
            self.registered_models[dataset_name] = {}

        if isinstance(training_result, ray.ObjectRef):
            # El llamador ya hizo ray.put(), guardamos la referencia directamente.
            model_artifacts_ref = training_result
        else:
            # El llamador pasó el objeto Python. Lo ponemos en el Object Store para obtener una referencia.
            print(f"[{self.__class__.__name__}] Recibido objeto directo para {dataset_name}/{model_type}, poniendo en Object Store...")
            model_artifacts_ref = ray.put(training_result)

        self.registered_models[dataset_name][model_type] = model_artifacts_ref
        print(f"[{self.__class__.__name__}] Modelo registrado: {dataset_name}/{model_type}")
        return True

    def get_model_artifacts_ref(self, dataset_name, model_type):
        """Devuelve la ObjectRef que apunta al diccionario de artefactos de un modelo específico."""
        return self.registered_models.get(dataset_name, {}).get(model_type)

    def list_models_details(self):
        """Devuelve una estructura simple de los modelos disponibles."""
        details = {}
        for dataset, models in self.registered_models.items():
            details[dataset] = {"available_models": list(models.keys())}
        return details

    def delete_dataset_models(self, dataset_name):
        """Elimina todos los modelos asociados a un dataset del registro."""
        if dataset_name in self.registered_models:
            del self.registered_models[dataset_name]
            print(f"[{self.__class__.__name__}] Modelos para el dataset '{dataset_name}' han sido eliminados del registro.")
            return True
        return False

# --- Tarea de Entrenamiento Remota ---
@ray.remote(num_cpus=1)
def train_and_serialize_model(data_df_input, target_column_name, model_config, param_grid_config, dataset_id, feature_names):
    hostname_str = socket.gethostname()
    log_prefix = f"Worker [{hostname_str}]"
    
    data_df = ray.get(data_df_input) if isinstance(data_df_input, ray.ObjectRef) else data_df_input
    print(f"{log_prefix} - Iniciando entrenamiento para {model_config['type']} en {dataset_id}...")

    try:
        X = data_df.drop(columns=[target_column_name]); y = data_df[target_column_name]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        preprocessor = Pipeline([('imputer', SimpleImputer(strategy='mean')), ('scaler', StandardScaler())])
        if model_config['type'] == "logistic_regression": model_instance = LogisticRegression(**model_config['params'])
        elif model_config['type'] == "decision_tree": model_instance = DecisionTreeClassifier(**model_config['params'])
        else: model_instance = RandomForestClassifier(**model_config['params'])
        
        full_pipeline = Pipeline([('preprocessor', preprocessor), ('classifier', model_instance)])
        param_grid = {f'classifier__{k}': v for k, v in param_grid_config.items()}

        grid_search = GridSearchCV(estimator=full_pipeline, param_grid=param_grid, cv=3, n_jobs=1)
        grid_search.fit(X_train, y_train)

        best_pipeline_obj = grid_search.best_estimator_
        y_pred = best_pipeline_obj.predict(X_test)
        
        metrics = {'accuracy': accuracy_score(y_test, y_pred)}
        print(f"{log_prefix} - Entrenamiento para {model_config['type']} en {dataset_id} completado. Accuracy: {metrics['accuracy']:.4f}")
        
        # Devolver un diccionario con los artefactos serializados
        return {
            "pipeline": pickle.dumps(best_pipeline_obj),
            "feature_names": pickle.dumps(feature_names),
            "metrics": json.dumps(metrics).encode('utf-8')
        }
    except Exception as e:
        print(f"{log_prefix} - ERROR en la tarea de entrenamiento para {dataset_id}/{model_config['type']}: {e}")
        traceback.print_exc()
        return None

# --- Función Orquestadora Principal ---
@ray.remote
def run_complete_training_job(dataset_name: str, df: pd.DataFrame, target_column: str):
    """
    Orquesta el ciclo de entrenamiento para un dataset de forma SECUENCIAL
    para garantizar la estabilidad en entornos con memoria limitada.
    """
    print(f"ORQUESTADOR: Iniciando trabajo de entrenamiento SECUENCIAL para dataset '{dataset_name}'...")
    
    registry_actor = ModelRegistryActor.options(name="model_registry", get_if_exists=True, max_restarts=-1, lifetime="detached", namespace="mi_plataforma", resources={"is_head_node": 1}).remote()
    
    feature_names = [col for col in df.columns if col != target_column]
    df_ref = ray.put(df)

    model_configurations = [
        {'type': "logistic_regression", 'params': {'max_iter': 200}, 'param_grid': {'C': [0.1, 1.0]}},
        {'type': "decision_tree", 'params': {}, 'param_grid': {'max_depth': [5], 'min_samples_split': [5]}},
        {'type': "random_forest", 'params': {'n_jobs': 1}, 'param_grid': {'n_estimators': [50], 'max_depth': [10]}}
    ]

    successful_count = 0
    for model_config in model_configurations:
        model_type = model_config['type']
        print(f"--- Iniciando entrenamiento para: {model_type} en dataset {dataset_name} ---")
        
        task_ref = train_and_serialize_model.remote(df_ref, target_column, model_config, model_config['param_grid'], dataset_name, feature_names)
        
        result_dict = ray.get(task_ref) 
        
        if result_dict is not None:
            # Pasa el diccionario de resultados directamente al actor.
            # El actor se encargará de hacer ray.put() si es necesario.
            registry_actor.register_model.remote(dataset_name, model_type, result_dict)
            successful_count += 1
            print(f"--- Entrenamiento para {model_type} completado y enviado para registro. ---")
        else:
            print(f"--- FALLO en entrenamiento para {model_type}. ---")

    final_message = f"Trabajo completado. {successful_count} de {len(model_configurations)} modelos para '{dataset_name}' fueron enviados para registro."
    print(f"ORQUESTADOR: {final_message}")
    return final_message

if __name__ == "__main__":
    print("Este script contiene la lógica de entrenamiento y debe ser importado por un servicio de gestión.")