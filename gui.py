# gui.py
import streamlit as st
import requests
import os
import pandas as pd
import ray  # Importante para conectar con el clúster

# --- Configuración ---
API_URL = "http://api-service:8000"  # Usamos el nombre del servicio de Docker Compose
GRAPHS_DIR = "/app/models_output/training_graphs"
RAY_ADDRESS = "ray://ray-head:10001" # Dirección del clúster de Ray para obtener el estado

st.set_page_config(layout="wide")
st.title("📊 Plataforma de Aprendizaje Supervisado Distribuido")

# --- Funciones de Interacción con la API y Ray ---

@st.cache_data(ttl=60) # Cachear por 60 segundos
def get_available_models():
    """Obtiene la lista de modelos desde la API."""
    try:
        response = requests.get(f"{API_URL}/models")
        if response.status_code == 200:
            return response.json()
    except requests.exceptions.ConnectionError:
        return None
    return None

@st.cache_data(ttl=10) # Cachear estado del clúster solo por 10 segundos para ver cambios
def get_ray_cluster_status():
    """Se conecta a Ray y devuelve el estado de los nodos."""
    try:
        # Se conecta al clúster de Ray si no está ya conectado
        if not ray.is_initialized():
            ray.init(address=RAY_ADDRESS, ignore_reinit_error=True)
        
        nodes = ray.nodes()
        if not nodes:
            return None, "No se encontraron nodos en el clúster de Ray."

        node_data = []
        for node in nodes:
            # Extrae la información relevante de cada nodo
            node_info = {
                "Node ID": node.get("NodeID"),
                "Estado": "Vivo" if node.get("Alive") else "Muerto",
                "Dirección": f"{node.get('NodeManagerAddress')}:{node.get('NodeManagerPort')}",
                "Hostname": node.get('NodeManagerHostname'),
                "Recursos (CPU)": node.get("Resources", {}).get("CPU", 0),
                # Convierte la memoria de bytes a Gigabytes para que sea más legible
                "Memoria Total (GB)": round(node.get("Resources", {}).get("memory", 0) / (1024**3), 2)
            }
            node_data.append(node_info)
        
        # Convierte la lista de diccionarios en un DataFrame de Pandas
        return pd.DataFrame(node_data), None
    except Exception as e:
        # Maneja errores de conexión o cualquier otro problema
        return None, f"Error al conectar o consultar el clúster de Ray: {e}"


# --- Diseño de la Interfaz ---

# Obtiene los datos de los modelos disponibles al cargar la página
models_data = get_available_models()

# Crea las pestañas de la interfaz
tab1, tab2, tab3 = st.tabs(["📈 Visualización del Entrenamiento", "🤖 Realizar Predicción", "📋 Estado del Sistema"])

with tab1:
    st.header("Métricas y Gráficas del Entrenamiento")
    st.write("Aquí se muestran las gráficas generadas después del proceso de entrenamiento.")

    if not os.path.exists(GRAPHS_DIR):
        st.warning(f"El directorio de gráficas `{GRAPHS_DIR}` no fue encontrado. Ejecuta el entrenamiento primero.")
    else:
        graph_files = [f for f in os.listdir(GRAPHS_DIR) if f.endswith('.png')]
        if not graph_files:
            st.info("No se encontraron gráficas. El entrenamiento podría no haber generado ninguna imagen.")
        else:
            # Muestra cada gráfica encontrada
            for graph_file in sorted(graph_files):
                st.image(os.path.join(GRAPHS_DIR, graph_file), use_column_width=True)

with tab2:
    st.header("Probar un Modelo en Producción")

    if not models_data:
        st.error("No se pudo conectar a la API o no hay modelos disponibles. Asegúrate de que los servicios de entrenamiento y API estén en ejecución.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            dataset = st.selectbox("Selecciona un Dataset:", list(models_data.keys()))
            if dataset:
                model_type = st.selectbox("Selecciona un Modelo:", list(models_data[dataset].keys()))

        if dataset and model_type:
            st.subheader(f"Características para {model_type} en {dataset}")
            
            required_features = models_data[dataset][model_type]['features_required']
            
            # Formulario para introducir los datos de la predicción
            with st.form("prediction_form"):
                feature_inputs = {}
                for feature in required_features:
                    feature_inputs[feature] = st.number_input(f"Valor para `{feature}`:", value=0.0, format="%.4f")
                
                submitted = st.form_submit_button("Predecir")

                if submitted:
                    payload = {"features": feature_inputs}
                    try:
                        # Envía la petición a la API
                        response = requests.post(f"{API_URL}/predict/{dataset}/{model_type}", json=payload)
                        if response.status_code == 200:
                            prediction = response.json()
                            st.success(f"**Resultado de la Predicción:** `{prediction['prediction']}`")
                        else:
                            st.error(f"Error en la API: {response.status_code} - {response.text}")
                    except requests.exceptions.ConnectionError as e:
                        st.error(f"No se pudo conectar a la API. ¿Está funcionando? Error: {e}")

with tab3:
    st.header("Estado del Clúster de Ray")
    
    # Botón para forzar la actualización de los datos
    if st.button("Actualizar Estado del Clúster"):
        # Limpia la caché para obtener datos frescos al presionar el botón
        st.cache_data.clear()

    # Llama a la función para obtener el estado del clúster
    df_nodes, error_message = get_ray_cluster_status()

    if error_message:
        st.error(error_message)
    elif df_nodes is not None and not df_nodes.empty:
        st.success("Conexión con el clúster de Ray exitosa.")
        st.write("A continuación se muestra el estado de los nodos que componen el clúster:")
        
        # Muestra la tabla con los datos de los nodos
        st.dataframe(df_nodes, use_container_width=True)

        # Muestra un resumen con métricas clave
        total_cpus = df_nodes["Recursos (CPU)"].sum()
        total_mem = df_nodes["Memoria Total (GB)"].sum()
        st.metric(label="Nodos Vivos Totales", value=len(df_nodes))
        st.metric(label="CPUs Totales en el Clúster", value=f"{total_cpus}")
        st.metric(label="Memoria Total en el Clúster (GB)", value=f"{total_mem:.2f} GB")
    else:
        st.warning("No se pudo obtener el estado del clúster o el clúster está vacío.")