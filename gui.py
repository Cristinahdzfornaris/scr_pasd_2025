# gui.py
import streamlit as st
import requests
import os
import pandas as pd

# URLs de las APIs (leídas de variables de entorno puestas por docker-compose)
MANAGEMENT_API_URL = os.environ.get("MANAGEMENT_API_URL", "http://localhost:9000")
INFERENCE_API_URL = os.environ.get("INFERENCE_API_URL", "http://localhost:8000")

st.set_page_config(layout="wide")
st.title("📊 Plataforma de Aprendizaje Supervisado Distribuido")

# --- Funciones de Ayuda ---
@st.cache_data(ttl=30)
def get_available_models():
    try:
        response = requests.get(f"{INFERENCE_API_URL}/models")
        if response.status_code == 200:
            return response.json()
    except requests.exceptions.ConnectionError:
        return None
    return {} # Devolver dict vacío en caso de error

# --- Pestañas de la Interfaz ---
tab_entrenamiento, tab_prediccion = st.tabs(["🚀 Nuevo Entrenamiento", "🤖 Realizar Predicción"])

with tab_entrenamiento:
    st.header("Entrenar Nuevos Modelos con un Dataset")
    st.markdown("Sube un archivo CSV, especifica un nombre para el dataset y el nombre de la columna objetivo.")

    with st.form("training_form"):
        dataset_name = st.text_input("Nombre del Dataset (e.g., 'carros_usados')", key="t_ds_name")
        uploaded_file = st.file_uploader("Sube tu archivo CSV", type=["csv"], key="t_file")
        target_column = st.text_input("Nombre de la Columna Objetivo", key="t_target")
        submitted = st.form_submit_button("Iniciar Entrenamiento")

        if submitted and dataset_name and uploaded_file and target_column:
            files = {'file': (uploaded_file.name, uploaded_file.getvalue(), 'text/csv')}
            data = {'target_column': target_column}
            
            with st.spinner(f"Enviando dataset '{dataset_name}' a la API de Gestión..."):
                try:
                    response = requests.post(f"{MANAGEMENT_API_URL}/datasets/{dataset_name}/train", files=files, data=data)
                    if response.status_code == 200:
                        st.success("¡Éxito! Trabajo de entrenamiento lanzado.")
                        st.json(response.json())
                        st.info("El entrenamiento se ejecuta en segundo plano. Refresca la lista de modelos en la pestaña de predicción en unos minutos.")
                    else:
                        st.error(f"Error de la API de Gestión ({response.status_code}):")
                        st.json(response.json())
                except requests.exceptions.ConnectionError:
                    st.error(f"No se pudo conectar a la API de Gestión en {MANAGEMENT_API_URL}.")

with tab_prediccion:
    st.header("Probar un Modelo Desplegado")

    if st.button("Refrescar lista de modelos"):
        st.cache_data.clear()

    models_data = get_available_models()

    if not models_data:
        st.warning("No hay modelos disponibles o la API no responde. Entrena un modelo en la pestaña de 'Nuevo Entrenamiento'.")
    else:
        col1, col2 = st.columns([1, 2])
        with col1:
            selected_dataset = st.selectbox("1. Selecciona un Dataset", list(models_data.keys()))
            if selected_dataset:
                selected_model = st.selectbox("2. Selecciona un Modelo", models_data[selected_dataset].get("available_models", []))
        
        with col2:
            if selected_dataset and selected_model:
                st.subheader(f"3. Ingresa las características para predecir")
                # Aquí necesitaríamos obtener los feature_names, pero por simplicidad
                # pedimos un JSON por ahora. Una GUI real tendría campos dinámicos.
                st.info("Para esta demo, por favor ingresa un JSON con el diccionario de 'features'.")
                features_json_str = st.text_area("JSON de Características", value='{"feature1": 0.0, "feature2": 0.0}', height=150)
                
                if st.button("Predecir"):
                    try:
                        features_dict = json.loads(features_json_str)
                        request_body = {"features": features_dict}
                        
                        with st.spinner("Enviando petición de predicción..."):
                            predict_response = requests.post(
                                f"{INFERENCE_API_URL}/predict/{selected_dataset}/{selected_model}",
                                json=request_body
                            )
                        
                        if predict_response.status_code == 200:
                            st.success("Predicción recibida:")
                            st.json(predict_response.json())
                        else:
                            st.error(f"Error en la API de Inferencia ({predict_response.status_code}):")
                            st.json(predict_response.json())

                    except json.JSONDecodeError:
                        st.error("El JSON de características no es válido.")
                    except requests.exceptions.ConnectionError:
                        st.error(f"No se pudo conectar a la API de Inferencia en {INFERENCE_API_URL}.")