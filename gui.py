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

# --- Configuración de la Página y URLs ---
st.set_page_config(layout="wide", page_title="Plataforma Distribuida de ML")

MANAGEMENT_API_URL = os.environ.get("MANAGEMENT_API_URL", "http://localhost:9000")
INFERENCE_API_URL = os.environ.get("INFERENCE_API_URL", "http://localhost:8000")
RAY_DASHBOARD_URL = os.environ.get("RAY_DASHBOARD_URL", "http://localhost:8265")

# --- Funciones de Ayuda ---
@st.cache_data(ttl=10) # Cachear por 10 segundos
def get_available_models():
    """Obtiene la lista de modelos disponibles desde la API de Inferencia."""
    try:
        response = requests.get(f"{INFERENCE_API_URL}/models")
        if response.status_code == 200:
            return response.json()
    except requests.exceptions.ConnectionError:
        st.error(f"No se pudo conectar a la API de Inferencia en {INFERENCE_API_URL}")
        return None
    return {}

@st.cache_data(ttl=60)
def get_model_metrics(dataset, model_type):
    """Obtiene las métricas de un modelo específico."""
    try:
        response = requests.get(f"{INFERENCE_API_URL}/models/{dataset}/{model_type}/metrics")
        if response.status_code == 200:
            return response.json()
    except requests.exceptions.ConnectionError:
        return {"error": f"No se pudo conectar para obtener métricas de {dataset}/{model_type}."}
    return {}

# --- Diseño de la Interfaz ---
st.title("📊 Plataforma de Aprendizaje Supervisado Distribuido")

# --- Sección de Monitoreo (se mantiene igual) ---
# ... (puedes añadir aquí la lógica para consultar la API del Ray Dashboard si quieres) ...
st.markdown("---")

# --- Pestañas Principales ---
tab_gestion, tab_prediccion, tab_metricas = st.tabs(["🗂️ Gestión y Entrenamiento", "🤖 Realizar Predicción", "📉 Métricas y Visualización"])

with tab_gestion:
    st.header("Añadir o Eliminar Datasets")
    
    with st.expander("🚀 Entrenar con un Nuevo Dataset", expanded=True):
        with st.form("training_form"):
            dataset_name = st.text_input("Nombre del Dataset (e.g., 'california_housing')", key="t_ds_name")
            uploaded_file = st.file_uploader("Sube tu archivo CSV", type=["csv"], key="t_file")
            target_column = st.text_input("Nombre de la Columna Objetivo (debe estar en el CSV)", key="t_target")
            submitted = st.form_submit_button("Iniciar Entrenamiento")

            if submitted and dataset_name and uploaded_file and target_column:
                files = {'file': (uploaded_file.name, uploaded_file.getvalue(), 'text/csv')}
                data = {'target_column': target_column}
                
                with st.spinner(f"Enviando dataset '{dataset_name}' a la API de Gestión para entrenamiento..."):
                    try:
                        response = requests.post(f"{MANAGEMENT_API_URL}/datasets/{dataset_name}/train", files=files, data=data)
                        if response.status_code == 200:
                            st.success("¡Éxito! Trabajo de entrenamiento lanzado.")
                            st.json(response.json())
                            st.info("El entrenamiento se ejecuta en segundo plano. Monitorea el estado en los logs o refresca la lista de modelos en unos minutos.")
                        else:
                            st.error(f"Error de la API de Gestión ({response.status_code}):")
                            try: st.json(response.json())
                            except: st.text(response.text)
                    except requests.exceptions.ConnectionError:
                        st.error(f"No se pudo conectar a la API de Gestión en {MANAGEMENT_API_URL}.")
    
    with st.expander("🗑️ Eliminar un Dataset y sus Modelos"):
        models_data = get_available_models()
        if not models_data:
            st.info("No hay datasets para eliminar.")
        else:
            dataset_to_delete = st.selectbox("Selecciona el dataset a eliminar", options=list(models_data.keys()), key="delete_ds_select")
            st.warning(f"¡Atención! Esto eliminará TODOS los modelos asociados al dataset '{dataset_to_delete}'.", icon="⚠️")
            if st.button(f"Eliminar Dataset '{dataset_to_delete}'", type="primary"):
                if dataset_to_delete:
                    with st.spinner(f"Enviando solicitud para eliminar '{dataset_to_delete}'..."):
                        try:
                            response = requests.delete(f"{MANAGEMENT_API_URL}/models/{dataset_to_delete}")
                            if response.status_code == 200:
                                st.success(f"Solicitud de eliminación para '{dataset_to_delete}' enviada.")
                                st.cache_data.clear()
                                time.sleep(1)
                                st.rerun()
                            else:
                                st.error(f"Error al eliminar ({response.status_code}):")
                                st.json(response.json())
                        except requests.exceptions.ConnectionError:
                            st.error(f"No se pudo conectar a la API de Gestión en {MANAGEMENT_API_URL}.")

with tab_prediccion:
    st.header("Probar un Modelo Desplegado")
    if st.button("Refrescar lista de modelos"):
        st.cache_data.clear()

    models_data = get_available_models()
    if not models_data:
        st.warning("No hay modelos disponibles para predicción.")
    else:
        dataset_options = list(models_data.keys())
        if not dataset_options:
             st.warning("No hay datasets con modelos disponibles.")
        else:
            selected_dataset = st.selectbox("1. Selecciona un Dataset", dataset_options, key="p_ds")
            if selected_dataset:
                model_options = models_data[selected_dataset].get("available_models", [])
                if not model_options:
                    st.warning(f"No hay modelos disponibles para el dataset '{selected_dataset}'.")
                else:
                    selected_model = st.selectbox("2. Selecciona un Modelo", model_options, key="p_model")
                    st.subheader(f"3. Ingresa las características para predecir")
                    st.info("Ingresa un JSON con el diccionario de 'features'. Ejemplo: {\"sepal length (cm)\": 5.1, ...}")
                    features_json_str = st.text_area("JSON de Características", value='{}', height=150)
                    
                    if st.button("Predecir", key="b_predict"):
                        if not features_json_str:
                            st.error("El JSON de características no puede estar vacío.")
                        else:
                            try:
                                features_dict = json.loads(features_json_str)
                                request_body = {"features": features_dict}
                                with st.spinner("Enviando petición de predicción..."):
                                    predict_response = requests.post(f"{INFERENCE_API_URL}/predict/{selected_dataset}/{selected_model}", json=request_body)
                                if predict_response.status_code == 200:
                                    st.success("Predicción recibida:")
                                    st.json(predict_response.json())
                                else:
                                    st.error(f"Error en la API de Inferencia ({predict_response.status_code}):")
                                    st.json(predict_response.json())
                            except json.JSONDecodeError: st.error("El JSON de características no es válido.")
                            except requests.exceptions.ConnectionError: st.error(f"No se pudo conectar a la API de Inferencia en {INFERENCE_API_URL}.")

with tab_metricas:
    st.header("Visualización de Métricas de Entrenamiento")
    models_data_metrics = get_available_models()

    if not models_data_metrics:
        st.warning("No hay modelos disponibles para mostrar métricas.")
    else:
        plot_data = []
        for dataset, models in models_data_metrics.items():
            for model_type in models.get("available_models", []):
                metrics = get_model_metrics(dataset, model_type)
                if metrics and 'accuracy' in metrics:
                    plot_data.append({
                        "dataset": dataset,
                        "model_type": model_type,
                        "accuracy": metrics['accuracy'],
                        "f1_score_macro": metrics.get('classification_report', {}).get('macro avg', {}).get('f1-score', 0)
                    })

        if not plot_data:
            st.info("No se pudieron cargar métricas para los modelos disponibles.")
        else:
            df_plot = pd.DataFrame(plot_data)

            # --- GRÁFICA 1: Precisión con Matplotlib/Seaborn ---
            st.subheader("Comparativa de Precisión (Accuracy)")
            fig1, ax1 = plt.subplots(figsize=(10, 6))
            sns.barplot(data=df_plot, x="dataset", y="accuracy", hue="model_type", ax=ax1, palette="viridis")
            ax1.set_title("Precisión de Modelos por Dataset")
            ax1.set_ylabel("Precisión (Test Set)")
            ax1.set_xlabel("Dataset")
            ax1.set_ylim(0, 1.05)
            ax1.legend(title="Tipo de Modelo")
            plt.xticks(rotation=15)
            st.pyplot(fig1)

            # --- GRÁFICA 2: F1-Score con Matplotlib/Seaborn ---
            st.subheader("Comparativa de F1-Score (Macro Average)")
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            sns.barplot(data=df_plot, x="dataset", y="f1_score_macro", hue="model_type", ax=ax2, palette="crest")
            ax2.set_title("F1-Score (Macro Avg) por Modelo y Dataset")
            ax2.set_ylabel("F1-Score (Macro Avg)")
            ax2.set_xlabel("Dataset")
            ax2.set_ylim(0, 1.05)
            ax2.legend(title="Tipo de Modelo")
            plt.xticks(rotation=15)
            st.pyplot(fig2)

            # --- Sección para ver métricas detalladas y matriz de confusión ---
            st.subheader("Detalles de Métricas y Matriz de Confusión por Modelo")
            selected_ds_metrics = st.selectbox("Selecciona un Dataset para ver detalles", list(models_data_metrics.keys()), key="m_ds")
            if selected_ds_metrics:
                selected_model_metrics = st.selectbox("Selecciona un Modelo", models_data_metrics[selected_ds_metrics].get("available_models", []), key="m_model")
                if selected_model_metrics:
                    st.markdown(f"**Métricas para `{selected_ds_metrics} / {selected_model_metrics}`**")
                    metrics_to_show = get_model_metrics(selected_ds_metrics, selected_model_metrics)
                    if "error" not in metrics_to_show:
                        st.json(metrics_to_show.get('classification_report', 'No disponible'))
                        
                        # Mostrar la matriz de confusión
                        cm = metrics_to_show.get('confusion_matrix')
                        if cm:
                            # Necesitamos los nombres de las clases. Los podríamos obtener de 'classification_report'
                            class_names = [k for k in metrics_to_show['classification_report'].keys() if k not in ['accuracy', 'macro avg', 'weighted avg']]
                            
                            fig_cm, ax_cm = plt.subplots()
                            sns.heatmap(np.array(cm), annot=True, fmt='d', cmap='Blues', ax=ax_cm,
                                        xticklabels=class_names, yticklabels=class_names)
                            ax_cm.set_title("Matriz de Confusión")
                            ax_cm.set_ylabel("Clase Verdadera")
                            ax_cm.set_xlabel("Clase Predicha")
                            st.pyplot(fig_cm)
                    else:
                        st.error("No se pudieron cargar las métricas detalladas.")