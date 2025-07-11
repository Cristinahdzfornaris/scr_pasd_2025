# train.py
import ray
import os
import pickle
import json
import traceback
import pandas as pd
import socket
import time
from typing import List

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# --- Actor de Registro de Modelos (Sin cambios) ---
@ray.remote
class ModelRegistryActor:
    def __init__(self):
        self.registered_models = {}
        print(f"[{self.__class__.__name__}] Actor de Registro de Modelos inicializado.")

    def register_model(self, dataset_name, model_type, task_result_ref):
        if dataset_name not in self.registered_models:
            self.registered_models[dataset_name] = {}
        self.registered_models[dataset_name][model_type] = task_result_ref
        print(f"[{self.__class__.__name__}] Modelo registrado: {dataset_name}/{model_type} con ref: {task_result_ref}")
        return True

    def get_model_artifacts_ref(self, dataset_name, model_type):
        return self.registered_models.get(dataset_name, {}).get(model_type)
    
    def get_deserialized_artifact(self, dataset_name, model_type, artifact_name):
        print(f"ACTOR: Solicitud para {dataset_name}/{model_type}/{artifact_name}")
        task_ref = self.get_model_artifacts_ref(dataset_name, model_type)
        if not task_ref:
            print(f"ACTOR: No se encontró la referencia de la tarea.")
            return None
        
        try:
            artifacts_dict = ray.get(task_ref)
            if not artifacts_dict or artifact_name not in artifacts_dict:
                print(f"ACTOR: Artefacto '{artifact_name}' no encontrado en el diccionario. Claves disponibles: {list(artifacts_dict.keys()) if isinstance(artifacts_dict, dict) else 'No es un dict'}")
                return None

            serialized_obj_bytes = artifacts_dict[artifact_name]
            
            if artifact_name == "metrics":
                return json.loads(serialized_obj_bytes.decode('utf-8'))
            else:
                return pickle.loads(serialized_obj_bytes)
        except Exception as e:
            print(f"ACTOR: ERROR al procesar artefacto para {dataset_name}/{model_type}: {e}")
            traceback.print_exc()
            return None
    def list_models_details(self):
        details = {}
        for dataset, models in self.registered_models.items():
            model_types = list(models.keys())
            if model_types:
                details[dataset] = {"available_models": model_types}
        return details

    def delete_dataset_models(self, dataset_name):
        if dataset_name in self.registered_models:
            del self.registered_models[dataset_name]
            print(f"[{self.__class__.__name__}] Modelos para '{dataset_name}' eliminados del registro.")
            return True
        return False

    def get_internal_state(self):
        """[DEBUG] Devuelve el diccionario completo de modelos registrados."""
        # Devolvemos una representación en string para evitar problemas de serialización
        # de ObjectRefs al cliente HTTP directamente.
        state_repr = {}
        for dataset, models in self.registered_models.items():
            state_repr[dataset] = {}
            for model_type, task_ref in models.items():
                state_repr[dataset][model_type] = str(task_ref)
        return state_repr

# --- Tarea de Entrenamiento Remota (CORREGIDA) ---
@ray.remote(num_cpus=1)
def train_and_serialize_model(
    data_df_input, 
    target_column_name: str, 
    model_config: dict, 
    param_grid_config: dict,
    dataset_id: str,
    feature_names: List[str], # Es crucial que este argumento se reciba
    class_names_for_metrics: List[str] # Para las etiquetas de las métricas
):
    """
    Tarea Ray que entrena un modelo, optimiza hiperparámetros, calcula métricas,
    serializa los artefactos resultantes y los devuelve como un diccionario.
    """
    hostname_str = socket.gethostname()
    log_prefix = f"Worker [{hostname_str}]"
    
    # 1. Obtener el DataFrame de forma robusta
    data_df = ray.get(data_df_input) if isinstance(data_df_input, ray.ObjectRef) else data_df_input
    
    model_type = model_config['type']
    print(f"{log_prefix} - Iniciando entrenamiento para {model_type} en {dataset_id}...")
    
    try:
        # 2. Preparar los datos X e y
        X = data_df.drop(columns=[target_column_name])
        y = data_df[target_column_name]
        
        # 3. Dividir los datos en entrenamiento y prueba
        stratify_option = y if len(y.unique()) > 1 and all(y.value_counts() >= 3) else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=stratify_option
        )

        # 4. Definir el pipeline de preprocesamiento y el modelo
        preprocessor = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])
        
        base_model_params = model_config.get('params', {})
        if model_type == "logistic_regression":
            model_instance = LogisticRegression(**base_model_params)
        elif model_type == "decision_tree":
            model_instance = DecisionTreeClassifier(**base_model_params)
        elif model_type == "random_forest":
            model_instance = RandomForestClassifier(**base_model_params)
        else:
            raise ValueError(f"Tipo de modelo '{model_type}' no reconocido.")
        
        full_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', model_instance)
        ])
        
        # 5. Configurar y ejecutar GridSearchCV
        param_grid = {f'classifier__{k}': v for k, v in param_grid_config.items()}
        grid_search = GridSearchCV(estimator=full_pipeline, param_grid=param_grid, cv=3, n_jobs=1)
        
        print(f"{log_prefix} - PUNTO DE CONTROL: Iniciando GridSearchCV.fit() para {model_type} en {dataset_id}...")
        start_fit_time = time.time()
        grid_search.fit(X_train, y_train)
        fit_duration = time.time() - start_fit_time
        print(f"{log_prefix} - PUNTO DE CONTROL: GridSearchCV.fit() completado. Duración: {fit_duration:.2f}s")

        # 6. Obtener el mejor pipeline y calcular métricas
        best_pipeline_obj = grid_search.best_estimator_
        y_pred = best_pipeline_obj.predict(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred, target_names=class_names_for_metrics, output_dict=True, zero_division=0),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'best_hyperparameters': grid_search.best_params_,
            'training_duration_sec': fit_duration
        }
        
        print(f"{log_prefix} - Entrenamiento para {model_type} en {dataset_id} completado. Accuracy: {metrics['accuracy']:.4f}")
        
        # 7. Serializar los artefactos y devolverlos en un diccionario
        result_dict = {
            "pipeline": pickle.dumps(best_pipeline_obj),
            "feature_names": pickle.dumps(feature_names),
            "metrics": json.dumps(metrics).encode('utf-8')
        }
        
        # Print de depuración final
        print(f"{log_prefix} - Devolviendo diccionario con claves: {list(result_dict.keys())}")
        
        return result_dict

    except Exception as e:
        print(f"{log_prefix} - ERROR en la tarea de entrenamiento para {dataset_id}/{model_config['type']}: {e}")
        traceback.print_exc()
        # Devolver None indica que la tarea falló
        return None

# --- Función Orquestadora Principal (CORREGIDA) ---
def run_complete_training_job(dataset_name: str, df: pd.DataFrame, target_column: str, models_to_train: List[str]):
    print(f"ORQUESTADOR: Iniciando trabajo para '{dataset_name}'. Modelos solicitados: {models_to_train}")
    
    # Obtener el actor de registro (asegúrate de que ya está creado por la management-api)
    registry_actor = ray.get_actor("model_registry", namespace="mi_plataforma")
    
    # Calcular los feature_names Y los class_names aquí
    feature_names = [col for col in df.columns if col != target_column]
    # --- AÑADIDO: OBTENER LOS NOMBRES DE LAS CLASES ---
    class_names = [str(c) for c in sorted(df[target_column].unique())]
    
    # Poner el DataFrame principal en el Object Store
    df_ref = ray.put(df)

    # Configuraciones de modelos a entrenar (puedes ajustar esto)
    all_model_configurations = {
        'logistic_regression': {'params': {'max_iter': 200, 'random_state': 42}, 'param_grid': {'C': [0.1, 1.0]}},
        'decision_tree': {'params': {'random_state': 42}, 'param_grid': {'max_depth': [5], 'min_samples_split': [5]}},
        'random_forest': {'params': {'n_jobs': 1, 'random_state': 42}, 'param_grid': {'n_estimators': [50], 'max_depth': [10]}}
    }
    
    selected_model_configs = {
        model_type: config for model_type, config in all_model_configurations.items() if model_type in models_to_train
    }

    if not selected_model_configs:
        return f"Error: Ninguno de los modelos solicitados {models_to_train} es válido."

    print(f"ORQUESTADOR: Lanzando {len(selected_model_configs)} tareas de entrenamiento para '{dataset_name}'...")
    
    for model_type, model_config in selected_model_configs.items():
        task_ref = train_and_serialize_model.remote(
            df_ref, 
            target_column, 
            {'type': model_type, 'params': model_config['params']}, 
            model_config['param_grid'], 
            dataset_name, 
            feature_names,
            class_names  # <--- CORRECCIÓN: PASAR EL ARGUMENTO FALTANTE
        )
        # Registrar la referencia a la tarea en el actor
        registry_actor.register_model.remote(dataset_name, model_type, task_ref)

    final_message = f"Trabajo lanzado. {len(selected_model_configs)} modelos para '{dataset_name}' se están entrenando en segundo plano."
    print(f"ORQUESTADOR: {final_message}")
    return final_message