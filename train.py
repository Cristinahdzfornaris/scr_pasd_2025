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

# --- Actor de Registro de Modelos (Guarda artefactos en memoria) ---
@ray.remote
class ModelRegistryActor:
    def __init__(self):
        # Almacena el diccionario completo de artefactos serializados (bytes)
        self.registered_models = {}
        print(f"[{self.__class__.__name__}] Actor de Registro de Modelos (en memoria) inicializado.")
    
    def register_model(self, dataset_name, model_type, result_dict):
        """
        Registra un modelo bajo un dataset específico.
        """

        if dataset_name not in self.registered_models:
            self.registered_models[dataset_name] = {}
        
        self.registered_models[dataset_name][model_type] = result_dict
        print(f"Registro en memoria para {dataset_name}/{model_type} completado.")
        return True

    def get_model_artifacts(self, dataset_name, model_type):
        """
        Devuelve el diccionario de artefactos serializados desde la memoria.
        """
        return self.registered_models.get(dataset_name, {}).get(model_type)

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
            # Nota: Esto no borra el archivo del disco. Se podría añadir esa lógica si se desea.
            return True
        return False

# --- Tarea de Entrenamiento Remota (Sin Cambios) ---
@ray.remote(num_cpus=1)
def train_and_serialize_model(
    data_df_input,
    target_column_name: str,
    model_config: dict,
    param_grid_config: dict,
    dataset_id: str,
    feature_names: List[str],
    class_names_for_metrics: List[str]
):
    hostname_str = socket.gethostname()
    log_prefix = f"Worker [{hostname_str}]"
    data_df = data_df_input

    model_type = model_config['type']
    print(f"{log_prefix} - Iniciando entrenamiento para {model_type} en {dataset_id}...")
    
    try:
        X = data_df.drop(columns=[target_column_name])
        y = data_df[target_column_name]
        
        stratify_option = y if len(y.unique()) > 1 and all(y.value_counts() >= 3) else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=stratify_option
        )

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
        
        param_grid = {f'classifier__{k}': v for k, v in param_grid_config.items()}
        grid_search = GridSearchCV(estimator=full_pipeline, param_grid=param_grid, cv=3, n_jobs=1)
        
        start_fit_time = time.time()
        grid_search.fit(X_train, y_train)
        fit_duration = time.time() - start_fit_time
        
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
        
        result_dict = {
            "pipeline": pickle.dumps(best_pipeline_obj),
            "feature_names": pickle.dumps(feature_names),
            "metrics": json.dumps(metrics).encode('utf-8')
        }
        return result_dict

    except Exception as e:
        print(f"{log_prefix} - ERROR en la tarea de entrenamiento para {dataset_id}/{model_config['type']}: {e}")
        traceback.print_exc()
        return None

# --- Función Orquestadora Principal (Con Persistencia) ---
def run_complete_training_job(dataset_name: str, df: pd.DataFrame, target_column: str, models_to_train: List[str]):
    print(f"ORQUESTADOR: Iniciando trabajo para '{dataset_name}'. Modelos: {models_to_train}")
    
    registry_actor = ray.get_actor("model_registry", namespace="mi_plataforma")
    
    feature_names = [col for col in df.columns if col != target_column]
    class_names = [str(c) for c in sorted(df[target_column].unique())]
    
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

    print(f"ORQUESTADOR: Lanzando {len(selected_model_configs)} tareas para '{dataset_name}'...")
    
    task_refs = []
    for model_type, model_config in selected_model_configs.items():
        task_ref = train_and_serialize_model.remote(
            df, 
            target_column, 
            {'type': model_type, 'params': model_config['params']}, 
            model_config['param_grid'], 
            dataset_name, 
            feature_names,
            class_names
        )
        task_refs.append((model_type, task_ref))

    print(f"ORQUESTADOR: Esperando la finalización de {len(task_refs)} tareas...")
    
    base_model_dir = "/app/persistent_models"
    os.makedirs(base_model_dir, exist_ok=True)
    
    for model_type, ref in task_refs:
        try:
            result_dictionary = ray.get(ref)
            if result_dictionary:
                # 1. Registrar en el actor en memoria para acceso rápido
                registry_actor.register_model.remote(dataset_name, model_type, result_dictionary)
                
                # 2. Guardar en disco para persistencia a largo plazo
                model_file_path = os.path.join(base_model_dir, f"{dataset_name}_{model_type}.pkl")
                with open(model_file_path, "wb") as f:
                    pickle.dump(result_dictionary, f)
                
                print(f"ORQUESTADOR: Modelo guardado en disco en: {model_file_path}")
            else:
                print(f"ORQUESTADOR: Tarea para {model_type} en {dataset_name} falló (devolvió None).")
        except Exception as e:
            print(f"ORQUESTADOR: Excepción al obtener resultado para {model_type}: {e}")
            traceback.print_exc()

    final_message = f"Trabajo para '{dataset_name}' completado."
    print(f"ORQUESTADOR: {final_message}")
    return final_message