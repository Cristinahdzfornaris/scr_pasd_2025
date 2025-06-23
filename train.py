# train.py
import ray
import joblib
import os
from datetime import datetime
import time
import json # Para guardar métricas y mejores HPs

import pandas as pd
import numpy as np # Para matriz de confusión
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, 
    classification_report, # Para precision, recall, f1-score
    confusion_matrix # Para la matriz de confusión
)
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
import socket

# --- Configuración ---
BASE_MODEL_DIR = "/app/models_output"

# --- Funciones Auxiliares ---
def get_dataset(dataset_name: str):
    data_loader = None
    target_names_map = None # Para nombres de clases en matriz de confusión
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
        return None, None, None, None, None

    data_sklearn = data_loader()
    feature_names_list = data_sklearn.feature_names if hasattr(data_sklearn, 'feature_names') else [f'feature_{i}' for i in range(data_sklearn.data.shape[1])]
    df = pd.DataFrame(data_sklearn.data, columns=feature_names_list)
    df['target'] = data_sklearn.target
    class_names = list(data_sklearn.target_names) if hasattr(data_sklearn, 'target_names') else [str(i) for i in sorted(df['target'].unique())]
    
    print(f"Dataset {dataset_name}: {df.shape[0]} muestras, {len(feature_names_list)} características. Clases: {class_names}")
    if df.isnull().values.any():
        print(f"Advertencia: El dataset {dataset_name} contiene valores NaN que serán imputados.")
    
    return df, 'target', dataset_name, feature_names_list, class_names


@ray.remote(num_cpus=1)
def train_model_with_hyperparam_tuning( 
    data_input, 
    target_column_name: str, 
    model_config: dict, 
    param_grid_config: dict,
    dataset_id: str,
    feature_names: list,
    class_names_for_metrics: list, # Nombres de las clases para la matriz de confusión
    cv_folds: int = 3, # Reducido para rapidez en un proyecto de curso
    test_size: float = 0.2, 
    random_state: int = 42
):
    worker_id_str = "unavailable" 
    hostname_str = socket.gethostname()
    if ray.is_initialized():
        try:
            runtime_ctx = ray.runtime_context.get_runtime_context()
            current_worker_id = runtime_ctx.get_worker_id()
            if current_worker_id: worker_id_str = current_worker_id[:8] 
            else:
                job_id = runtime_ctx.get_job_id(); actor_id = runtime_ctx.get_actor_id()
                if job_id and not actor_id: worker_id_str = f"driver_job_{job_id.hex()[:4]}"
                else: worker_id_str = "ctx_no_wid"
        except Exception:  worker_id_str = "err_get_wid"
    log_prefix = f"Worker [{worker_id_str}@{hostname_str}]"
    print(f"{log_prefix} - Tarea iniciada. Tipo de data_input recibido: {type(data_input)}")

    data_df = None
    if isinstance(data_input, ray.ObjectRef): data_df = ray.get(data_input)
    elif isinstance(data_input, pd.DataFrame): data_df = data_input
    else:
        error_msg = f"Tipo de dato inesperado: {type(data_input)}"; print(f"{log_prefix} - {error_msg}")
        return None, {}, model_config.get('type', 'unknown'), dataset_id, None, None, None, hostname_str, worker_id_str

    model_type = model_config['type']
    base_model_params = model_config.get('params', {})
    print(f"{log_prefix} - Iniciando HP Tuning: {model_type} para {dataset_id}...")

    try:
        X = data_df.drop(columns=[target_column_name]); y = data_df[target_column_name]
        stratify_option = y if len(y.unique()) > 1 and all(y.value_counts() >= cv_folds) else None
        if stratify_option is None and len(y.unique()) > 1:
             print(f"{log_prefix} - Advertencia: No se puede estratificar para {dataset_id}, pocas muestras por clase para {cv_folds} pliegues.")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=stratify_option)

        preprocessor = Pipeline([('imputer', SimpleImputer(strategy='mean')), ('scaler', StandardScaler())])
        if model_type == "logistic_regression": model_instance = LogisticRegression(**base_model_params)
        elif model_type == "decision_tree": model_instance = DecisionTreeClassifier(**base_model_params)
        elif model_type == "random_forest": model_instance = RandomForestClassifier(**base_model_params)
        else:
            print(f"{log_prefix} - Error: Modelo '{model_type}' no reconocido."); return None, {}, model_type, dataset_id, None, None, None, hostname_str, worker_id_str
        
        full_pipeline = Pipeline([('preprocessor', preprocessor), ('classifier', model_instance)])
        current_param_grid = {f'classifier__{k}': v for k, v in param_grid_config.items()}
        
        print(f"{log_prefix} - PUNTO DE CONTROL: Iniciando GridSearchCV.fit() para {model_type} en {dataset_id}...")
        start_fit_time = time.time()
        grid_search = GridSearchCV(estimator=full_pipeline, param_grid=current_param_grid, cv=cv_folds, scoring='accuracy', n_jobs=1)
        grid_search.fit(X_train, y_train)
        fit_duration = time.time() - start_fit_time
        print(f"{log_prefix} - PUNTO DE CONTROL: GridSearchCV.fit() completado. Duración: {fit_duration:.2f}s")

        best_pipeline_obj = grid_search.best_estimator_
        best_params_found = grid_search.best_params_
        y_pred_test = best_pipeline_obj.predict(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred_test),
            'classification_report': classification_report(y_test, y_pred_test, target_names=class_names_for_metrics, output_dict=True, zero_division=0),
            'confusion_matrix': confusion_matrix(y_test, y_pred_test).tolist(), # tolist para que sea serializable a JSON
            'best_hyperparameters': best_params_found,
            'training_duration_sec': fit_duration
        }
        print(f"{log_prefix} - Métricas para {model_type} en {dataset_id} [Test Set]: Accuracy={metrics['accuracy']:.4f}")

        output_subdir = os.path.join(BASE_MODEL_DIR, dataset_id, model_type)
        os.makedirs(output_subdir, exist_ok=True)
        
        pipeline_filename = os.path.join(output_subdir, "best_pipeline.joblib")
        feature_names_filename = os.path.join(output_subdir, "feature_names.joblib")
        metrics_filename = os.path.join(output_subdir, "metrics.json") # Para guardar todas las métricas

        joblib.dump(best_pipeline_obj, pipeline_filename)
        joblib.dump(feature_names, feature_names_filename)
        with open(metrics_filename, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        print(f"{log_prefix} - Artefactos para {model_type} en {dataset_id} guardados en: {output_subdir}")
        
        # Devolvemos el path del pipeline, las métricas completas, y otros datos
        return pipeline_filename, metrics, model_type, dataset_id, feature_names_filename, best_params_found, hostname_str, worker_id_str

    except Exception as e:
        print(f"{log_prefix} - ERROR en HP Tuning para {model_type} en {dataset_id}: {e}")
        traceback.print_exc()
        return None, {}, model_type, dataset_id, None, None, hostname_str, worker_id_str

# --- NUEVA FUNCIÓN PARA GENERAR GRÁFICAS DE ENTRENAMIENTO ---
def generar_graficas_entrenamiento(training_results_list, base_dir):
    print("\n--- Generando Gráficas de Entrenamiento ---")
    
    plot_data_accuracy = []
    all_metrics_data = [] # Para almacenar todas las métricas para posible futuro uso

    for result_item in training_results_list:
        if result_item and result_item[0] is not None:
            # pipeline_path, metrics_dict, model_t, ds_id, fn_path, best_hp, exec_host, exec_worker = result_item
            metrics_dict = result_item[1]
            model_t = result_item[2]
            ds_id = result_item[3]
            
            plot_data_accuracy.append({'dataset': ds_id, 'model_type': model_t, 'accuracy': metrics_dict['accuracy']})
            all_metrics_data.append({
                'dataset': ds_id, 
                'model_type': model_t, 
                **metrics_dict # Desempaquetar todas las métricas
            })
    
    if not plot_data_accuracy:
        print("No hay datos de entrenamiento exitosos para graficar precisión.")
        return

    df_accuracy = pd.DataFrame(plot_data_accuracy)
    df_all_metrics = pd.DataFrame(all_metrics_data)

    graphs_dir = os.path.join(base_dir, "training_graphs")
    os.makedirs(graphs_dir, exist_ok=True)

    # Gráfica 1: Precisión por tipo de modelo para cada dataset
    try:
        plt.figure(figsize=(12, 7)); sns.set_theme(style="whitegrid")
        sns.barplot(data=df_accuracy, x='dataset', y='accuracy', hue='model_type', palette='viridis')
        plt.title('Precisión de Modelos por Dataset (Test Set)'); plt.ylabel('Precisión (Accuracy)'); plt.xlabel('Dataset')
        plt.xticks(rotation=30, ha='right'); plt.ylim(0, 1.05)
        plt.legend(title='Tipo de Modelo', bbox_to_anchor=(1.05, 1), loc='upper left'); plt.tight_layout()
        graph_path1 = os.path.join(graphs_dir, "accuracy_by_dataset_model.png"); plt.savefig(graph_path1); plt.close()
        print(f"Gráfica guardada en: {graph_path1}")
    except Exception as e: print(f"Error al generar 'accuracy_by_dataset_model.png': {e}")

    # Gráfica 2: Matrices de Confusión
    for index, row in df_all_metrics.iterrows():
        if 'confusion_matrix' in row and row['confusion_matrix'] is not None:
            try:
                cm = np.array(row['confusion_matrix']) # Ya debería ser una lista de listas, convertir a array
                dataset_name = row['dataset']
                model_type_name = row['model_type']
                # Obtener class_names para este dataset (esto es un poco truco, idealmente lo pasarías o guardarías)
                _, _, _, _, class_names = get_dataset(dataset_name) # Re-obtenemos para los nombres

                plt.figure(figsize=(8, 6)); sns.set_theme(style="white")
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                            xticklabels=class_names, yticklabels=class_names)
                plt.title(f'Matriz de Confusión: {model_type_name} en {dataset_name}'); plt.ylabel('Clase Verdadera'); plt.xlabel('Clase Predicha')
                plt.tight_layout()
                graph_path_cm = os.path.join(graphs_dir, f"confusion_matrix_{dataset_name}_{model_type_name}.png"); plt.savefig(graph_path_cm); plt.close()
                print(f"Gráfica de Matriz de Confusión guardada en: {graph_path_cm}")
            except Exception as e: print(f"Error al generar matriz de confusión para {row.get('dataset', 'unk')}_{row.get('model_type', 'unk')}: {e}")
    
    # Gráfica 3: F1-score (promedio macro) por modelo y dataset
    f1_plot_data = []
    for index, row in df_all_metrics.iterrows():
        if 'classification_report' in row and row['classification_report'] and 'macro avg' in row['classification_report']:
             f1_plot_data.append({
                 'dataset': row['dataset'],
                 'model_type': row['model_type'],
                 'f1_score_macro': row['classification_report']['macro avg']['f1-score']
             })
    if f1_plot_data:
        df_f1 = pd.DataFrame(f1_plot_data)
        try:
            plt.figure(figsize=(12, 7)); sns.set_theme(style="whitegrid")
            sns.barplot(data=df_f1, x='dataset', y='f1_score_macro', hue='model_type', palette='crest')
            plt.title('F1-score (Macro Avg) por Modelo y Dataset'); plt.ylabel('F1-score (Macro Avg)'); plt.xlabel('Dataset')
            plt.xticks(rotation=30, ha='right'); plt.ylim(0, 1.05)
            plt.legend(title='Tipo de Modelo', bbox_to_anchor=(1.05, 1), loc='upper left'); plt.tight_layout()
            graph_path_f1 = os.path.join(graphs_dir, "f1_score_by_dataset_model.png"); plt.savefig(graph_path_f1); plt.close()
            print(f"Gráfica de F1-score guardada en: {graph_path_f1}")
        except Exception as e: print(f"Error al generar 'f1_score_by_dataset_model.png': {e}")


# --- Flujo Principal de Entrenamiento ---
if __name__ == "__main__":
    print("Iniciando script de entrenamiento distribuido con optimización de hiperparámetros...")
    
    try:
        if not ray.is_initialized():
            ray.init(address='auto', ignore_reinit_error=True) 
        print(f"Conectado a Ray. Nodos del clúster: {ray.nodes()}")
    except Exception as e:
        print(f"Error crítico al inicializar o conectar con Ray: {e}"); exit(1)
    
    os.makedirs(BASE_MODEL_DIR, exist_ok=True)
    datasets_to_process = ["iris", "wine", "breast_cancer"]
    model_configurations = [
        {'type': "logistic_regression", 'params': {'solver': 'liblinear', 'max_iter': 300, 'random_state': 42}, 'param_grid': {'C': [0.1, 1.0, 10.0]}},
        {'type': "decision_tree", 'params': {'random_state': 42}, 'param_grid': {'max_depth': [3, 5, None], 'min_samples_split': [2, 5, 10]}},
        {'type': "random_forest", 'params': {'random_state': 42, 'n_jobs': 1}, 'param_grid': {'n_estimators': [50, 100], 'max_depth': [5, 10]}} # Reducido para rapidez
    ]
    all_training_task_refs = []

    for dataset_name in datasets_to_process:
        print(f"\n--- Procesando dataset: {dataset_name} ---")
        data_df, target_col, dataset_id_str, feature_names_list, class_names_list = get_dataset(dataset_name)
        if data_df is None: continue
        data_df_ref = ray.put(data_df)
        feature_names_ref = ray.put(feature_names_list)
        class_names_ref = ray.put(class_names_list) # Poner nombres de clases en object store

        print(f"Lanzando entrenamientos con HP Tuning para el dataset: {dataset_name}")
        for model_config_entry in model_configurations:
            task_ref = train_model_with_hyperparam_tuning.remote(
                data_df_ref, target_col,
                {'type': model_config_entry['type'], 'params': model_config_entry['params']},
                model_config_entry['param_grid'], dataset_id_str,
                feature_names_ref, # Pasar referencia
                class_names_ref,   # Pasar referencia
                cv_folds=3 # Reducido para rapidez general
            )
            all_training_task_refs.append(task_ref)

    print(f"\nTodos los {len(all_training_task_refs)} trabajos lanzados. Esperando resultados...")
    try:
        training_results = ray.get(all_training_task_refs)
    except ray.exceptions.RayTaskError as e:
        print(f"ERROR FATAL: Tarea Ray falló: {e.cause}"); exit(1)
    except Exception as e:
        print(f"ERROR inesperado en ray.get(): {e}"); traceback.print_exc(); exit(1)

    print("\n--- Resumen Final del Entrenamiento (con HP Tuning) ---")
    successful_trainings = 0
    parsed_results_for_graphing = [] # Lista para pasar a la función de graficar

    for result in training_results:
        if result and result[0] is not None:
            pipeline_path, metrics_dict, model_t, ds_id, fn_path, best_hp, exec_host, exec_worker = result
            parsed_results_for_graphing.append(result) # Guardar el resultado completo
            print(f"  Dataset: {ds_id:<15} | Modelo: {model_t:<20} | Precisión (Test): {metrics_dict['accuracy']:.4f} | Host: {exec_host} ({exec_worker})")
            print(f"    Mejores HP: {best_hp}")
            print(f"    Pipeline: {pipeline_path}")
            # Imprimir más métricas del classification_report
            if 'classification_report' in metrics_dict and metrics_dict['classification_report']:
                for class_label, report_metrics in metrics_dict['classification_report'].items():
                    if isinstance(report_metrics, dict): # Para cada clase y promedios
                        print(f"      {class_label:<25} - Precision: {report_metrics.get('precision', 0):.2f}, Recall: {report_metrics.get('recall', 0):.2f}, F1: {report_metrics.get('f1-score', 0):.2f}")
            successful_trainings += 1
        else: # Manejo de fallos
            # ... (código de manejo de fallos como lo tenías)
            if result: 
                 _, _, model_t_fail, ds_id_fail, _, _, exec_host_f, exec_worker_f = result # Asumiendo misma estructura de retorno
                 print(f"  FALLO - Dataset: {ds_id_fail:<15} | Modelo: {model_t_fail:<20} | Host: {exec_host_f} ({exec_worker_f})")
            else:
                 print(f"  FALLO - Un trabajo de entrenamiento no devolvió un resultado válido (probablemente None).")

    print(f"\nEntrenamiento completado. {successful_trainings} de {len(all_training_task_refs)} entrenamientos fueron exitosos.")
    
    if parsed_results_for_graphing:
        generar_graficas_entrenamiento(parsed_results_for_graphing, BASE_MODEL_DIR)

    if successful_trainings < len(all_training_task_refs):
        print("ADVERTENCIA: No todos los entrenamientos fueron exitosos.")
    print("Script de entrenamiento finalizado.")