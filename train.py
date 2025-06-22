# train.py
import ray
import joblib
import os
from datetime import datetime
import traceback # Para imprimir tracebacks completos

# 2. Imports de librerías de terceros
import pandas as pd
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# --- Configuración ---
BASE_MODEL_DIR = "/app/models_output"

# --- Funciones Auxiliares --- (Asumo que get_dataset está definida como antes)
def get_dataset(dataset_name: str):
    data_loader = None
    if dataset_name == "iris":
        data_loader = load_iris
        print(f"Cargando dataset Iris...")
    elif dataset_name == "wine":
        data_loader = load_wine
        print(f"Cargando dataset Wine...")
    elif dataset_name == "breast_cancer":
        data_loader = load_breast_cancer
        print(f"Cargando dataset Breast Cancer...")
    else:
        print(f"Error: Dataset '{dataset_name}' no reconocido.")
        return None, None, None

    data_sklearn = data_loader()
    feature_names = data_sklearn.feature_names if hasattr(data_sklearn, 'feature_names') else [f'feature_{i}' for i in range(data_sklearn.data.shape[1])]
    df = pd.DataFrame(data_sklearn.data, columns=feature_names)
    df['target'] = data_sklearn.target
    
    print(f"Dataset {dataset_name}: {df.shape[0]} muestras, {len(feature_names)} características.")
    if df.isnull().values.any():
        print(f"Advertencia: El dataset {dataset_name} contiene valores NaN que serán imputados.")
    
    return df, 'target', dataset_name


@ray.remote(num_cpus=1)
def train_model_with_hyperparam_tuning(
    data_input, 
    target_column_name: str, 
    model_config: dict, 
    param_grid_config: dict,
    dataset_id: str,
    cv_folds: int = 3,
    test_size: float = 0.2, 
    random_state: int = 42
):
    
    # --- Obtención robusta del Worker ID ---
    worker_id_str = "unavailable" # Default si no se puede obtener
    
    try:
        if ray.is_initialized():
            runtime_context = ray.getRuntimeContext() # Uso recomendado para versiones más nuevas de Ray
            if runtime_context and hasattr(runtime_context, 'get_worker_id'):
                worker_id_full = runtime_context.get_worker_id()
                if worker_id_full: # Asegurarse de que no sea None o vacío
                    # El WorkerID es un objeto binario, convertirlo a hexadecimal
                    worker_id_str = worker_id_full.hex()[:8] 
    except Exception as e_worker_id:
        print(f"Advertencia (Worker ID): No se pudo obtener el worker_id: {e_worker_id}")
        # worker_id_str permanecerá como "unavailable"

    worker_id_short = worker_id_str # Usar el worker_id_str obtenido
    # --- Fin de Obtención del Worker ID ---
    
    print(f"Worker [{worker_id_short}] - Tarea iniciada. Tipo de data_input recibido: {type(data_input)}")
    import time
# ...
    
# ...
    # Comprueba si lo que recibiste es una ObjectRef o el DataFrame directamente
    if isinstance(data_input, ray.ObjectRef):
        print(f"Worker [{worker_id_short}] - data_input es una ObjectRef, llamando a ray.get()...")
        try:
            data_df = ray.get(data_input)
        except Exception as e_get:
            print(f"Worker [{worker_id_short}] - ERROR al hacer ray.get(): {e_get}")
            traceback.print_exc()
            return None, 0.0, model_config.get('type', 'unknown'), dataset_id, None, None, None
    elif isinstance(data_input, pd.DataFrame):
        print(f"Worker [{worker_id_short}] - data_input YA es el DataFrame, usándolo directamente.")
        data_df = data_input
    else:
        error_msg = f"Tipo de dato inesperado recibido para data_input: {type(data_input)}"
        print(f"Worker [{worker_id_short}] - {error_msg}")
        return None, 0.0, model_config.get('type', 'unknown'), dataset_id, None, None, None

    model_type = model_config['type']
    base_model_params = model_config.get('params', {})

    print(f"Worker [{worker_id_short}] - Iniciando entrenamiento con HP Tuning: {model_type} para dataset {dataset_id}...")
    print(f"Worker [{worker_id_short}] - Simulating long work for {model_type} on {dataset_id}...")
    time.sleep(30) # Espera 30 segundos
    try:
        X = data_df.drop(columns=[target_column_name])
        y = data_df[target_column_name]

        stratify_option = None
        if len(y.unique()) > 1 and y.dtype.kind in 'iufc':
            if all(y.value_counts() >= cv_folds) and all(y.value_counts() >= 2): # Condición más segura
                 stratify_option = y
            else:
                print(f"Advertencia (Worker [{worker_id_short}]): No se puede estratificar de forma segura para {dataset_id} con {cv_folds} pliegues. Algunas clases tienen muy pocas muestras.")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=stratify_option
        )

        imputer = SimpleImputer(strategy='mean')
        scaler = StandardScaler()

        preprocessor_pipeline = Pipeline([
            ('imputer', imputer),
            ('scaler', scaler)
        ])

        if model_type == "logistic_regression":
            model_instance = LogisticRegression(**base_model_params)
        elif model_type == "decision_tree":
            model_instance = DecisionTreeClassifier(**base_model_params)
        elif model_type == "random_forest":
            model_instance = RandomForestClassifier(**base_model_params)
        else:
            print(f"Worker [{worker_id_short}] - Error: Tipo de modelo '{model_type}' no reconocido.")
            return None, 0.0, model_type, dataset_id, None, None, None

        full_pipeline_for_gridsearch = Pipeline([
            ('preprocessor', preprocessor_pipeline),
            ('classifier', model_instance)
        ])
        
        current_param_grid = {f'classifier__{k}': v for k, v in param_grid_config.items()}

        print(f"Worker [{worker_id_short}] - Iniciando GridSearchCV para {model_type} en {dataset_id} con rejilla: {current_param_grid} y CV={cv_folds}...")
        
        grid_search = GridSearchCV(
            estimator=full_pipeline_for_gridsearch, 
            param_grid=current_param_grid, 
            cv=cv_folds, 
            scoring='accuracy',
            n_jobs=1 
        )

        grid_search.fit(X_train, y_train)

        best_pipeline = grid_search.best_estimator_
        best_params = grid_search.best_params_
        
        print(f"Worker [{worker_id_short}] - Mejores Hiperparámetros para {model_type} en {dataset_id}: {best_params}")

        y_pred = best_pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Worker [{worker_id_short}] - Precisión con Mejores HP ({model_type} en {dataset_id} [test set]): {accuracy:.4f}")

        output_subdir = os.path.join(BASE_MODEL_DIR, dataset_id, model_type)
        os.makedirs(output_subdir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pipeline_filename = os.path.join(output_subdir, f"best_pipeline_{timestamp}.joblib")
        best_imputer_filename = os.path.join(output_subdir, f"imputer_{timestamp}.joblib")
        best_scaler_filename = os.path.join(output_subdir, f"scaler_{timestamp}.joblib")

        joblib.dump(best_pipeline, pipeline_filename)
        joblib.dump(best_pipeline.named_steps['preprocessor'].named_steps['imputer'], best_imputer_filename)
        joblib.dump(best_pipeline.named_steps['preprocessor'].named_steps['scaler'], best_scaler_filename)
        
        print(f"Worker [{worker_id_short}] - Artefactos (con HP tuning) para {model_type} en {dataset_id} guardados en: {output_subdir}")
        return pipeline_filename, accuracy, model_type, dataset_id, best_imputer_filename, best_scaler_filename, best_params
    
    except Exception as e:
        print(f"Worker [{worker_id_short}] - ERROR en HP Tuning para {model_type} en {dataset_id}: {e}")
        traceback.print_exc()
        return None, 0.0, model_type, dataset_id, None, None, None


# --- Flujo Principal de Entrenamiento ---
if __name__ == "__main__":
    print("Iniciando script de entrenamiento distribuido con optimización de hiperparámetros...")
    
    try:
        if not ray.is_initialized():
            ray.init(address='auto', ignore_reinit_error=True) 
        print(f"Conectado a Ray. Nodos del clúster: {ray.nodes()}")
    except Exception as e:
        print(f"Error crítico al inicializar o conectar con Ray: {e}")
        traceback.print_exc()
        exit(1)
    
    try:
        os.makedirs(BASE_MODEL_DIR, exist_ok=True)

        datasets_to_process = ["iris", "wine", "breast_cancer"]
        
        model_configurations = [
            {
                'type': "logistic_regression", 
                'params': {'solver': 'liblinear', 'max_iter': 200, 'random_state': 42},
                'param_grid': {'C': [0.1, 1.0, 10.0]}
            },
            {
                'type': "decision_tree", 
                'params': {'random_state': 42},
                'param_grid': {'max_depth': [3, 5, None], 'min_samples_split': [2, 5]}
            },
            {
                'type': "random_forest", 
                'params': {'random_state': 42},
                'param_grid': {'n_estimators': [50, 100], 'max_depth': [5, None]}
            }
        ]

        all_training_task_refs = []

        for dataset_name in datasets_to_process:
            print(f"\n--- Procesando dataset: {dataset_name} ---")
            data_df, target_col, dataset_id_str = get_dataset(dataset_name)
            
            if data_df is None:
                print(f"Saltando dataset {dataset_name} debido a error en la carga.")
                continue

            data_df_ref = ray.put(data_df)
            print(f"Tipo de data_df_ref ANTES de llamar a .remote() para {dataset_name}: {type(data_df_ref)}")

            print(f"Lanzando entrenamientos con HP Tuning para el dataset: {dataset_name}")
            for model_config_entry in model_configurations:
                task_ref = train_model_with_hyperparam_tuning.remote(
                    data_df_ref, 
                    target_col,
                    {'type': model_config_entry['type'], 'params': model_config_entry['params']},
                    model_config_entry['param_grid'],
                    dataset_id_str,
                    cv_folds=3 
                )
                all_training_task_refs.append(task_ref)

        print(f"\nTodos los {len(all_training_task_refs)} trabajos de entrenamiento (con HP tuning) han sido lanzados. Esperando resultados...")
        
        training_results = ray.get(all_training_task_refs)

        print("\n--- Resumen Final del Entrenamiento (con HP Tuning) ---")
        successful_trainings = 0
        for i, result in enumerate(training_results):
            original_task_ref = all_training_task_refs[i]

            if result and result[0] is not None: 
                pipeline_path, acc, model_t, ds_id, imputer_path, scaler_path, best_hp = result
                print(f"  Dataset: {ds_id:<15} | Modelo: {model_t:<20} | Precisión (Test): {acc:<.4f}")
                print(f"    Mejores HP: {best_hp}")
                print(f"    Pipeline: {pipeline_path}")
                successful_trainings += 1
            else:
                print(f"  FALLO - Tarea (ObjectRef: {original_task_ref}) no completada exitosamente o no devolvió el formato esperado.")
                if result:
                     # Intentar desempaquetar con cuidado si el formato es el esperado pero pipeline_path es None
                     try:
                         _, _, model_t_fail, ds_id_fail, _, _, _ = result 
                         print(f"    Información parcial: Dataset: {ds_id_fail:<15} | Modelo: {model_t_fail:<20}")
                     except ValueError:
                         print(f"    El resultado de la tarea no pudo ser desempaquetado: {result}")
                else:
                     print(f"    El resultado de la tarea fue None.")
                
        print(f"\nEntrenamiento (con HP tuning) completado. {successful_trainings} de {len(all_training_task_refs)} entrenamientos fueron exitosos.")
    
    except ray.exceptions.RayTaskError as e:
        print(f"ERROR: Una o más tareas de Ray fallaron durante ray.get().")
        # El objeto e ya es la RayTaskError, así que e.cause es la excepción original de la tarea.
        print(f"Error original en la tarea que causó la RayTaskError: {e.cause}")
        print("\nTraceback de la tarea fallida (si está disponible en e.cause):")
        if hasattr(e, 'cause') and e.cause and hasattr(e.cause, '__traceback__'):
             traceback.print_exception(type(e.cause), e.cause, e.cause.__traceback__)
        else:
             # Si no hay e.cause o traceback en e.cause, imprimir el traceback de la RayTaskError misma
             traceback.print_exc() 
        exit(1)
    except Exception as e:
        print(f"ERROR inesperado en el script principal del driver: {e}")
        traceback.print_exc()
        exit(1)
        
    print("Script de entrenamiento finalizado.")