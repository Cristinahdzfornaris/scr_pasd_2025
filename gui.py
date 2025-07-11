# gui.py
import streamlit as st
import requests
import os
import pandas as pd
import json
import time
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import ray 
# Importar las utilidades de monitoreo
from monitoring_utils import get_cluster_nodes_status, get_actor_status, get_inference_stats 

# --- Configuración de la Página y URLs de las APIs ---
st.set_page_config(layout="wide", page_title="Plataforma Distribuida de ML")

# URLs leídas de variables de entorno puestas por docker-compose
MANAGEMENT_API_URL = os.environ.get("MANAGEMENT_API_URL", "http://localhost:9000")
INFERENCE_API_URL = os.environ.get("INFERENCE_API_URL", "http://localhost:8000")
RAY_DASHBOARD_URL = os.environ.get("RAY_DASHBOARD_URL", "http://localhost:8265")

# --- Conexión a Ray (una sola vez por sesión) ---
if 'ray_initialized' not in st.session_state:
    try:
        ray_address = os.environ.get("RAY_ADDRESS", "auto")
        ray.init(address=ray_address, namespace="mi_plataforma", ignore_reinit_error=True)
        st.session_state['ray_initialized'] = True
    except Exception as e:
        st.session_state['ray_initialized'] = False
        st.session_state['ray_init_error'] = e # Guardar el error para mostrarlo

# --- Funciones de Ayuda con Caché de Streamlit ---
@st.cache_data(ttl=15) # Cachear la lista de modelos por 15 segundos
def get_available_models():
    try:
        response = requests.get(f"{INFERENCE_API_URL}/models")
        if response.status_code == 200:
            return response.json()
    except requests.exceptions.ConnectionError:
        return None # Devuelve None si la API no está disponible
    return {}

@st.cache_data(ttl=60)
def get_model_metrics(dataset, model_type):
    try:
        response = requests.get(f"{INFERENCE_API_URL}/models/{dataset}/{model_type}/metrics")
        if response.status_code == 200:
            return response.json()
    except requests.exceptions.ConnectionError:
        return {"error": "API no disponible"}
    return response.json() if response.status_code == 200 else {}

@st.cache_data(ttl=300)
def fetch_features_for_model(dataset, model):
    try:
        response = requests.get(f"{INFERENCE_API_URL}/models/{dataset}/{model}/features")
        if response.status_code == 200:
            return response.json().get("features", [])
    except requests.exceptions.ConnectionError:
        return None
    return []

# --- Diseño de la Interfaz ---
st.title("📊 Plataforma de Aprendizaje Supervisado Distribuido")
st.markdown("---")

# --- Pestañas Principales ---
tab_gestion, tab_prediccion, tab_metricas, tab_monitoreo = st.tabs([
    "🗂️ Gestión y Entrenamiento", 
    "🤖 Realizar Predicción", 
    "📉 Métricas de Entrenamiento",
    "📡 Dashboard de Monitoreo" 
])

# --- Pestaña 1: Gestión y Entrenamiento ---
with tab_gestion:
    st.header("Añadir o Eliminar Datasets y Modelos")
    
    with st.expander("🚀 Entrenar con uno o más Datasets", expanded=True):
        with st.form("multi_training_form"):
            st.markdown("##### 1. Sube uno o más archivos CSV")
            uploaded_files = st.file_uploader("Sube tus archivos CSV", type=["csv"], accept_multiple_files=True)
            
            st.markdown("##### 2. Configura cada Dataset")
            datasets_to_train_config = []
            if uploaded_files:
                for i, uploaded_file in enumerate(uploaded_files):
                    st.markdown(f"**Archivo: `{uploaded_file.name}`**")
                    cols = st.columns(2)
                    default_ds_name = os.path.splitext(uploaded_file.name)[0]
                    dataset_name = cols[0].text_input("Nombre del Dataset", value=default_ds_name, key=f"ds_name_{i}")
                    target_column = cols[1].text_input("Nombre de la Columna Objetivo", key=f"target_{i}")
                    
                    if dataset_name and target_column:
                        datasets_to_train_config.append({
                            "dataset_name": dataset_name, "target_column": target_column, "file": uploaded_file
                        })
                    st.markdown("---")
            else:
                st.info("Sube al menos un archivo CSV para configurar el entrenamiento.")

            st.markdown("##### 3. Selecciona los Modelos a Entrenar")
            model_options = ["logistic_regression", "decision_tree", "random_forest"]
            selected_models = st.multiselect("Elige los modelos:", options=model_options, default=model_options)

            submitted = st.form_submit_button("Iniciar Entrenamiento para Todos los Datasets Subidos")
            if submitted:
                if not datasets_to_train_config:
                    st.error("Debes subir y configurar al menos un dataset.")
                elif not selected_models:
                    st.error("Debes seleccionar al menos un modelo para entrenar.")
                else:
                    files_to_send = [('files', (cfg["file"].name, cfg["file"].getvalue(), 'text/csv')) for cfg in datasets_to_train_config]
                    configs_json = json.dumps([{"dataset_name": cfg["dataset_name"], "target_column": cfg["target_column"], "filename": cfg["file"].name} for cfg in datasets_to_train_config])
                    form_data = {'configs': configs_json, 'models_to_train': selected_models}
                    
                    with st.spinner(f"Enviando {len(datasets_to_train_config)} datasets para entrenamiento..."):
                        try:
                            response = requests.post(f"{MANAGEMENT_API_URL}/datasets/train_batch", files=files_to_send, data=form_data)
                            if response.status_code == 200:
                                st.success("¡Éxito! Trabajos de entrenamiento lanzados.")
                                st.json(response.json())
                                st.info("El entrenamiento se ejecuta en segundo plano. Refresca la lista de modelos en unos minutos.")
                            else:
                                st.error(f"Error de la API de Gestión ({response.status_code}):"); st.json(response.json())
                        except requests.exceptions.ConnectionError:
                            st.error(f"No se pudo conectar a la API de Gestión en {MANAGEMENT_API_URL}.")
    
    with st.expander("🗑️ Eliminar un Dataset y sus Modelos"):
        models_data_to_delete = get_available_models()
        if not models_data_to_delete:
            st.info("No hay datasets para eliminar.")
        else:
            dataset_to_delete = st.selectbox("Selecciona el dataset a eliminar", options=list(models_data_to_delete.keys()), key="delete_ds_select")
            st.warning(f"¡Atención! Esto eliminará TODOS los modelos asociados al dataset '{dataset_to_delete}'.", icon="⚠️")
            if st.button(f"Eliminar Dataset '{dataset_to_delete}'", type="primary"):
                if dataset_to_delete:
                    with st.spinner(f"Enviando solicitud para eliminar '{dataset_to_delete}'..."):
                        try:
                            response = requests.delete(f"{MANAGEMENT_API_URL}/models/{dataset_to_delete}")
                            if response.status_code == 200:
                                st.success(f"Solicitud de eliminación para '{dataset_to_delete}' enviada."); st.cache_data.clear(); time.sleep(1); st.rerun()
                            else:
                                st.error(f"Error al eliminar ({response.status_code}):"); st.json(response.json())
                        except requests.exceptions.ConnectionError:
                            st.error(f"No se pudo conectar a la API de Gestión en {MANAGEMENT_API_URL}.")

# --- Pestaña 2: Realizar Predicción ---
with tab_prediccion:
    st.header("Probar un Modelo Desplegado")

    if st.button("Refrescar lista de modelos", key="refresh_pred"): st.cache_data.clear()

    models_data = get_available_models()
    if models_data is None:
        st.error("La API de Inferencia no está disponible. Verifica que el servicio esté corriendo.")
    elif not models_data:
        st.warning("No hay modelos disponibles para predicción. Entrena uno en la pestaña de 'Gestión'.")
    else:
        dataset_options = list(models_data.keys())
        selected_dataset = st.selectbox("1. Selecciona un Dataset", dataset_options, key="p_ds")
        
        if selected_dataset and models_data.get(selected_dataset):
            model_options = models_data[selected_dataset].get("available_models", [])
            selected_model = st.selectbox("2. Selecciona un Modelo", model_options, key="p_model")
            
            if selected_model:
                st.subheader(f"3. Ingresa los valores de las características")
                feature_names = fetch_features_for_model(selected_dataset, selected_model)
                if feature_names is None:
                    st.error("No se pudo conectar a la API para obtener las características.")
                elif not feature_names:
                    st.warning("Este modelo no tiene nombres de características definidos. La predicción podría no ser confiable.")
                else:
                    with st.form("prediction_form"):
                        feature_inputs = {feature: st.number_input(label=feature, key=f"feat_{feature}", value=0.0, format="%.4f") for feature in feature_names}
                        submitted = st.form_submit_button("Predecir")
                        if submitted:
                            with st.spinner("Enviando petición de predicción..."):
                                predict_response = requests.post(f"{INFERENCE_API_URL}/predict/{selected_dataset}/{selected_model}", json={"features": feature_inputs})
                            if predict_response.status_code == 200:
                                st.success("Predicción recibida:"); st.json(predict_response.json())
                            else:
                                st.error(f"Error en la API de Inferencia ({predict_response.status_code}):"); st.json(predict_response.json())

# --- Pestaña 3: Métricas de Entrenamiento ---
with tab_metricas:
    st.header("Visualización de Métricas de Entrenamiento")
    if st.button("Refrescar datos de métricas", key="refresh_metrics"): st.cache_data.clear()

    models_data_metrics = get_available_models()
    if models_data_metrics:
        plot_data, all_metrics = [], []
        for dataset, models in models_data_metrics.items():
            for model_type in models.get("available_models", []):
                metrics = get_model_metrics(dataset, model_type)
                if metrics and 'accuracy' in metrics:
                    all_metrics.append({'dataset': dataset, 'model_type': model_type, **metrics})
                    plot_data.append({'dataset': dataset, 'model_type': model_type, 'accuracy': metrics['accuracy'], 'f1_score_macro': metrics.get('classification_report', {}).get('macro avg', {}).get('f1-score', 0)})
        
        if plot_data:
            df_plot = pd.DataFrame(plot_data)
            st.subheader("Comparativa de Precisión (Accuracy)"); fig1, ax1 = plt.subplots(); sns.barplot(data=df_plot, x="dataset", y="accuracy", hue="model_type", ax=ax1); st.pyplot(fig1)
            st.subheader("Comparativa de F1-Score (Macro Average)"); fig2, ax2 = plt.subplots(); sns.barplot(data=df_plot, x="dataset", y="f1_score_macro", hue="model_type", ax=ax2, palette="crest"); st.pyplot(fig2)
            st.subheader("Detalles de Métricas y Matriz de Confusión"); df_all_metrics = pd.DataFrame(all_metrics)
            selected_metric_row = st.selectbox("Selecciona un modelo entrenado para ver detalles", options=df_all_metrics.index, format_func=lambda i: f"{df_all_metrics.loc[i, 'dataset']} / {df_all_metrics.loc[i, 'model_type']}")
            if selected_metric_row is not None:
                row = df_all_metrics.loc[selected_metric_row]
                st.json(row.get('classification_report', 'No disponible'))
                cm = row.get('confusion_matrix'); class_names = [k for k in row['classification_report'].keys() if k not in ['accuracy', 'macro avg', 'weighted avg']]
                if cm and class_names:
                    fig_cm, ax_cm = plt.subplots(); sns.heatmap(np.array(cm), annot=True, fmt='d', cmap='Blues', ax=ax_cm, xticklabels=class_names, yticklabels=class_names); ax_cm.set_ylabel("Verdadero"); ax_cm.set_xlabel("Predicho"); st.pyplot(fig_cm)
    else: st.info("No hay modelos para mostrar métricas.")

# --- Pestaña 4: Dashboard de Monitoreo ---
with tab_monitoreo:
    st.header("Dashboard de Monitoreo del Sistema")
    
    if st.button("Refrescar Datos del Dashboard", key="refresh_monitor"):
        # Limpiar la caché de las funciones de monitoreo si usan @st.cache_data
        # get_cluster_nodes_status.clear()
        # get_actor_status.clear()
        st.rerun()

    # --- ¡LA COMPROBACIÓN MÁS IMPORTANTE DEL SCRIPT! ---
    if not st.session_state.get('ray_initialized', False):
        # Si la conexión a Ray falló, muestra el error y no intentes hacer nada más con Ray.
        st.error("Fallo al conectar con el clúster de Ray. El monitoreo del clúster está desactivado.")
        if 'ray_init_error' in st.session_state:
            st.exception(st.session_state.get('ray_init_error'))
    else:
        # --- SOLO SI LA CONEXIÓN A RAY ES EXITOSA, LLAMA A LAS FUNCIONES ---
        
        # Sección de Estado del Clúster Ray
        with st.expander("Estado del Clúster Ray", expanded=True):
            st.subheader("Nodos del Clúster")
            try:
                # Llama a la función DENTRO del bloque condicional
                node_data = get_cluster_nodes_status() 
                if node_data:
                    st.table(pd.DataFrame(node_data))
                else:
                    st.warning("No se pudieron obtener los datos de los nodos del clúster.")
            except Exception as e:
                st.error(f"Error al obtener el estado de los nodos: {e}")

            st.subheader("Estado del Actor de Registro (`ModelRegistryActor`)")
            try:
                # Llama a la función DENTRO del bloque condicional
                actor_data = get_actor_status() 
                if actor_data:
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Estado", actor_data.get("Estado", "N/A"))
                    col2.metric("Reinicios", actor_data.get("Reinicios", "N/A"))
                    col3.metric("Vivo", actor_data.get("Vivo", "❓"))
            except Exception as e:
                st.error(f"Error al obtener el estado del actor: {e}")
    with st.expander("Estadísticas de Inferencia (API)", expanded=True):
        st.subheader("Rendimiento del Servicio de Predicción")
        inference_stats = get_inference_stats(f"{INFERENCE_API_URL}/metrics")
        if "error" in inference_stats: st.error(inference_stats["error"])
        else:
            col1, col2 = st.columns(2)
            col1.metric("Total de Peticiones", f"{inference_stats.get('total_requests', 0)}")
            col2.metric("Latencia Promedio", f"{inference_stats.get('average_latency_ms', 0):.2f} ms")
            st.markdown("**Desglose por Modelo:**"); details_df = pd.DataFrame(inference_stats.get("details_by_model", []))
            st.table(details_df)

    st.markdown(f"Para una vista más detallada, visita el [Ray Dashboard]({RAY_DASHBOARD_URL}) oficial.", unsafe_allow_html=True)