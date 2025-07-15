# gui.py (Versión 6.1 - Funcionalidad Completa Restaurada)

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
import socket
from itertools import cycle
from monitoring_utils import get_cluster_nodes_status, get_actor_status, get_aggregated_inference_stats
# ==============================================================================
# --- CLASE DE CLIENTE RESILIENTE (SIN CAMBIOS) ---
# ==============================================================================

class ResilientClient:
    """
    Un cliente HTTP que realiza descubrimiento de servicios, balanceo de carga
    round-robin y reintentos del lado del cliente.
    """
    def __init__(self, service_name, port):
        self.service_name = service_name
        self.port = port
        self.servers = []
        self.server_cycler = cycle([])
        self.discover()

    def discover(self):
        """Descubre o redescubre los servidores. Puede ser llamado para refrescar."""
        try:
            ips = [info[4][0] for info in socket.getaddrinfo(self.service_name, self.port, socket.AF_INET)]
            unique_ips = sorted(list(set(ips)))
            self.servers = [f"http://{ip}:{self.port}" for ip in unique_ips]
            if self.servers:
                print(f"Servidores descubiertos para '{self.service_name}': {self.servers}")
                self.server_cycler = cycle(self.servers)
            else:
                st.error(f"CRÍTICO: No se pudo descubrir ninguna réplica para el servicio '{self.service_name}'.")
                self.server_cycler = cycle([])
        except socket.gaierror:
            st.error(f"CRÍTICO: El nombre de servicio '{self.service_name}' no se pudo resolver. ¿Está corriendo?")
            self.servers = []
            self.server_cycler = cycle([])

    def make_request(self, method, path, **kwargs):
        """
        Realiza una petición HTTP, probando cada servidor en orden hasta que
        uno responda o todos fallen.
        """
        if not self.servers:
            raise Exception(f"No hay servidores disponibles para el servicio '{self.service_name}'.")
        for _ in range(len(self.servers)):
            server_url = next(self.server_cycler)
            full_url = f"{server_url}{path}"
            try:
                print(f"Cliente Resiliente: Intentando petición a: {full_url}")
                response = requests.request(method, full_url, **kwargs)
                response.raise_for_status() 
                print(f"Cliente Resiliente: Petición exitosa a: {full_url}")
                return response
            except requests.exceptions.RequestException as e:
                print(f"Cliente Resiliente: Fallo al contactar a {server_url}: {e}. Probando siguiente réplica...")
                st.toast(f"Réplica {server_url} no responde, reintentando...", icon="⚠️")
        st.error(f"El servicio '{self.service_name}' no está disponible. Todas las réplicas han fallado.")
        self.discover()
        raise Exception(f"El servicio '{self.service_name}' no está disponible. Todas las réplicas fallaron.")

# ==============================================================================
# --- INICIALIZACIÓN Y CONFIGURACIÓN ---
# ==============================================================================

# --- Intenta importar las utilidades de monitoreo ---
try:
    ### CORREGIDO ### - Asegúrate de que esta función también sea resiliente si depende de una API
    from monitoring_utils import get_cluster_nodes_status, get_actor_status, get_inference_stats
    MONITORING_ENABLED = True
except ImportError:
    MONITORING_ENABLED = False
    print("ADVERTENCIA: monitoring_utils.py no encontrado. La pestaña de monitoreo estará desactivada.")

# --- Configuración de la Página y Clientes Resilientes ---
st.set_page_config(layout="wide", page_title="Plataforma Distribuida de ML")

# Cliente para el servicio de Gestión
MANAGEMENT_API_SERVICE_NAME = os.environ.get("MANAGEMENT_API_SERVICE_NAME", "management-api")
MANAGEMENT_API_PORT = int(os.environ.get("MANAGEMENT_API_PORT", 9000))
management_client = ResilientClient(MANAGEMENT_API_SERVICE_NAME, MANAGEMENT_API_PORT)

# Cliente para el servicio de Inferencia
INFERENCE_API_SERVICE_NAME = os.environ.get("INFERENCE_API_SERVICE_NAME", "api-service")
INFERENCE_API_PORT = int(os.environ.get("INFERENCE_API_PORT", 8000))
inference_client = ResilientClient(INFERENCE_API_SERVICE_NAME, INFERENCE_API_PORT)

RAY_DASHBOARD_URL = os.environ.get("RAY_DASHBOARD_URL", f"http://localhost:8265")

# --- Conexión a Ray (solo para la pestaña de monitoreo) ---
if MONITORING_ENABLED and 'ray_initialized' not in st.session_state:
    try:
        ray.init(address="auto", namespace="mi_plataforma", ignore_reinit_error=True)
        st.session_state['ray_initialized'] = True
    except Exception as e:
        st.session_state['ray_initialized'] = False
        st.session_state['ray_init_error'] = e

# --- Inicialización del Estado de la Sesión ---
if 'analyzed_files' not in st.session_state:
    st.session_state['analyzed_files'] = {}
if 'configs' not in st.session_state:
    st.session_state['configs'] = {}

# ==============================================================================
# --- FUNCIONES DE AYUDA (USANDO CLIENTES RESILIENTES) ---
# ==============================================================================

@st.cache_data(ttl=10)
def get_available_models():
    try:
        response = inference_client.make_request("GET", "/models", timeout=5)
        return response.json()
    except Exception: return {}

@st.cache_data(ttl=30)
def get_model_metrics(dataset, model_type):
    try:
        response = inference_client.make_request("GET", f"/models/{dataset}/{model_type}/metrics", timeout=10)
        return response.json()
    except Exception: return {}

@st.cache_data(ttl=60)
def fetch_features_for_model(dataset, model):
    try:
        response = inference_client.make_request("GET", f"/models/{dataset}/{model}/features", timeout=10)
        return response.json().get("features", [])
    except Exception: return None

# ==============================================================================
# --- DISEÑO DE LA INTERFAZ ---
# ==============================================================================

st.title("📊 Plataforma de Aprendizaje Supervisado Distribuido")
st.markdown("---")

tab_definitions = ["🗂️ Gestión y Entrenamiento", "🤖 Realizar Predicción", "📉 Métricas y Comparativa"]
if MONITORING_ENABLED:
    tab_definitions.append("📡 Dashboard de Monitoreo")

created_tabs = st.tabs(tab_definitions)
tab_gestion = created_tabs[0]
tab_prediccion = created_tabs[1]
tab_metricas = created_tabs[2]
tab_monitoreo = created_tabs[3] if len(created_tabs) > 3 else None

# ==============================================================================
# --- PESTAÑA 1: GESTIÓN Y ENTRENAMIENTO ---
# ==============================================================================
with tab_gestion:
    st.header("Flujo de Trabajo de Entrenamiento")
    st.subheader("1. Sube tus Datasets")
    uploaded_files = st.file_uploader(
        "Selecciona uno o más archivos CSV", type=["csv"], accept_multiple_files=True, label_visibility="collapsed"
    )

    if uploaded_files:
        for file in uploaded_files:
            if file.name not in st.session_state.analyzed_files:
                with st.spinner(f"Analizando `{file.name}`..."):
                    files_payload = {'file': (file.name, file, 'text/csv')}
                    try:
                        response = management_client.make_request("POST", "/datasets/analyze", files=files_payload, timeout=30)
                        st.session_state.analyzed_files[file.name] = response.json()
                        st.session_state.configs[file.name] = {
                            'dataset_name': os.path.splitext(file.name)[0],
                            'target_column': response.json().get('columns', [''])[0],
                            'selected_models': ["logistic_regression", "decision_tree"]
                        }
                    except Exception as e:
                        st.error(f"Error de conexión al analizar `{file.name}`: {e}")
                        st.session_state.analyzed_files[file.name] = {"error": str(e)}

    st.markdown("---")
    st.subheader("2. Configura los Parámetros de Entrenamiento")

    if not st.session_state.analyzed_files:
        st.info("Sube un archivo CSV para empezar a configurar.")
    else:
        files_to_configure = list(st.session_state.analyzed_files.keys())
        for filename in files_to_configure:
            if filename not in st.session_state.analyzed_files: continue
            file_info = st.session_state.analyzed_files[filename]
            config = st.session_state.configs.get(filename)
            if not config: continue

            with st.container(border=True):
                st.markdown(f"**Configuración para `{filename}`**")
                if "error" in file_info:
                    st.error(f"No se pudo procesar: {file_info['error']}")
                    continue

                col_ds, col_target = st.columns(2)
                config['dataset_name'] = col_ds.text_input("Nombre del Dataset", value=config.get('dataset_name', ''), key=f"ds_{filename}")
                columns_list = file_info.get("columns", [])
                try:
                    current_index = columns_list.index(config.get('target_column'))
                except (ValueError, TypeError): current_index = 0
                config['target_column'] = col_target.selectbox("Columna Objetivo", options=columns_list, index=current_index, key=f"target_{filename}")
                config['selected_models'] = st.multiselect("Modelos a Entrenar", options=["logistic_regression", "decision_tree", "random_forest"], default=config.get('selected_models', []), key=f"models_{filename}")
                
                if st.button(f"🗑️ Eliminar Dataset y Modelos", key=f"delete_{filename}", type="secondary"):
                    with st.spinner(f"Eliminando '{config['dataset_name']}'..."):
                        try:
                            management_client.make_request("DELETE", f"/models/{config['dataset_name']}", timeout=30)
                            st.session_state.analyzed_files.pop(filename, None)
                            st.session_state.configs.pop(filename, None)
                            st.success(f"Dataset '{config['dataset_name']}' eliminado.")
                            st.cache_data.clear(); st.rerun()
                        except Exception as e: st.error(f"Error al eliminar: {e}")

    st.markdown("---")
    st.subheader("3. Elige tu Modo de Entrenamiento")

    ### RESTAURADO ### - Lógica de entrenamiento individual y por lotes
    if not st.session_state.configs:
        st.warning("Primero configura al menos un dataset en el paso 2.")
    else:
        # MODO GRANULAR
        with st.container(border=True):
            st.markdown("##### Entrenamiento Individual")
            dataset_names = [cfg['dataset_name'] for cfg in st.session_state.configs.values() if 'dataset_name' in cfg]
            dataset_to_train_individually = st.selectbox("Selecciona un dataset para entrenar:", options=dataset_names)
            
            if st.button("🚀 Entrenar este Dataset Individualmente"):
                filename_to_train, config_to_train = None, None
                for fname, cfg in st.session_state.configs.items():
                    if cfg.get('dataset_name') == dataset_to_train_individually:
                        filename_to_train = fname
                        config_to_train = cfg
                        break
                
                if filename_to_train and config_to_train:
                    if not config_to_train.get('selected_models'):
                        st.error("Este dataset no tiene modelos seleccionados.")
                    else:
                        original_file = next((f for f in uploaded_files if f.name == filename_to_train), None)
                        if original_file:
                             with st.spinner(f"Lanzando entrenamiento para `{config_to_train['dataset_name']}`..."):
                                files_payload = [('files', (filename_to_train, original_file.getvalue(), 'text/csv'))]
                                config_payload = {'configs': json.dumps([{"dataset_name": config_to_train['dataset_name'], "target_column": config_to_train['target_column'], "filename": filename_to_train}]), 'models_to_train': config_to_train['selected_models']}
                                try:
                                    management_client.make_request("POST", "/datasets/train_batch", files=files_payload, data=config_payload, timeout=120)
                                    st.success(f"¡Trabajo para '{config_to_train['dataset_name']}' lanzado!"); st.balloons(); st.cache_data.clear()
                                except Exception as e: st.error(f"Error al lanzar entrenamiento: {e}")
                        else: st.error(f"El archivo '{filename_to_train}' ya no está disponible.")

        # MODO POR LOTES
        with st.container(border=True):
            st.markdown("##### Entrenamiento por Lotes")
            if st.button("🚀 Entrenar TODOS los Datasets Configurados", type="primary"):
                with st.spinner(f"Lanzando lote de entrenamiento..."):
                    success_count = 0
                    for filename, config in st.session_state.configs.items():
                        if not config.get('selected_models'): continue
                        original_file = next((f for f in uploaded_files if f.name == filename), None)
                        if original_file:
                            files_payload = [('files', (filename, original_file.getvalue(), 'text/csv'))]
                            config_payload = {'configs': json.dumps([{"dataset_name": config['dataset_name'], "target_column": config['target_column'], "filename": filename}]), 'models_to_train': config['selected_models']}
                            try:
                                management_client.make_request("POST", "/datasets/train_batch", files=files_payload, data=config_payload, timeout=120)
                                success_count += 1
                                st.toast(f"Trabajo para '{config['dataset_name']}' lanzado.", icon="🚀")
                            except Exception as e: st.error(f"Fallo al lanzar '{config['dataset_name']}': {e}")
                    st.success(f"¡Lote completado! {success_count} trabajos lanzados."); st.balloons(); st.cache_data.clear()

# ==============================================================================
# --- PESTAÑA 2: REALIZAR PREDICCIÓN ---
# ==============================================================================
with tab_prediccion:
    st.header("🔮 Probar un Modelo Desplegado")
    models_data = get_available_models()
    if not models_data:
        st.warning("No hay modelos disponibles. Entrena uno en la pestaña de 'Gestión'.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            dataset_options = list(models_data.keys())
            selected_dataset = st.selectbox("1. Selecciona un Dataset", dataset_options, key="p_ds")
        if selected_dataset and models_data.get(selected_dataset):
            with col2:
                model_options = models_data[selected_dataset].get("available_models", [])
                selected_model = st.selectbox("2. Selecciona un Modelo", model_options, key="p_model")
            if selected_model:
                st.subheader(f"3. Ingresa los valores de las características")
                feature_names = fetch_features_for_model(selected_dataset, selected_model)
                if feature_names is not None:
                    with st.form("prediction_form"):
                        # Se corrigió la lógica para usar un diccionario directamente
                        feature_inputs = {feat: st.number_input(label=feat, key=f"feat_{feat}", value=0.0, format="%.4f") for feat in feature_names}
                        if st.form_submit_button("Realizar Predicción", type="primary"):
                            with st.spinner("Enviando predicción..."):
                                try:
                                    # El backend espera un diccionario de features, no una lista
                                    response = inference_client.make_request("POST", f"/predict/{selected_dataset}/{selected_model}", json={"features": feature_inputs}, timeout=15)
                                    st.success("Predicción recibida:")
                                    st.json(response.json())
                                except Exception as e: st.error(f"Error en la predicción: {e}")

# ==============================================================================
# --- PESTAÑA 3: MÉTRICAS Y COMPARATIVA ---
# ==============================================================================
with tab_metricas:
    st.header("📉 Métricas y Comparativa de Modelos")
    if st.button("Refrescar Datos de Métricas"): st.cache_data.clear(); st.rerun()

    models_data_metrics = get_available_models()
    if not models_data_metrics:
        st.info("No hay modelos entrenados para mostrar métricas.")
    else:
        ### RESTAURADO ### - Lógica completa de métricas y visualización
        plot_data = []; all_metrics_data = []
        for dataset, models in models_data_metrics.items():
            for model_type in models.get("available_models", []):
                metrics = get_model_metrics(dataset, model_type)
                if metrics and 'accuracy' in metrics:
                    all_metrics_data.append({'dataset': dataset, 'model_type': model_type, **metrics})
                    plot_data.append({'dataset': dataset, 'model_type': model_type, 'accuracy': metrics['accuracy'], 'f1_score_macro': metrics.get('classification_report', {}).get('macro avg', {}).get('f1-score', 0)})
        
        if plot_data:
            df_plot = pd.DataFrame(plot_data)
            st.subheader("Comparativa de Rendimiento General")
            col1, col2 = st.columns(2)
            with col1:
                fig1, ax1 = plt.subplots()
                sns.barplot(data=df_plot, x="accuracy", y="dataset", hue="model_type", ax=ax1, orient='h')
                ax1.set_title("Precisión (Accuracy)"); ax1.set_xlabel("Accuracy"); ax1.set_ylabel("Dataset")
                st.pyplot(fig1)

            with col2:
                fig2, ax2 = plt.subplots()
                sns.barplot(data=df_plot, x="f1_score_macro", y="dataset", hue="model_type", ax=ax2, orient='h', palette="viridis")
                ax2.set_title("F1-Score (Macro Avg)"); ax2.set_xlabel("F1-Score"); ax2.set_ylabel("")
                st.pyplot(fig2)

            st.markdown("---")
            st.subheader("Análisis Detallado por Modelo")
            df_all_metrics = pd.DataFrame(all_metrics_data)
            if not df_all_metrics.empty:
                selected_idx = st.selectbox("Selecciona un modelo para inspeccionar:", options=df_all_metrics.index, format_func=lambda i: f"{df_all_metrics.loc[i, 'dataset']} / {df_all_metrics.loc[i, 'model_type']}")
                if selected_idx is not None:
                    row = df_all_metrics.loc[selected_idx]
                    c1, c2 = st.columns(2)
                    with c1:
                        st.text("Reporte de Clasificación:")
                        st.json(row.get('classification_report', {}))
                    with c2:
                        cm = row.get('confusion_matrix')
                        class_report = row.get('classification_report', {})
                        class_names = [k for k in class_report.keys() if k not in ['accuracy', 'macro avg', 'weighted avg']]
                        if cm and class_names:
                            fig_cm, ax_cm = plt.subplots()
                            sns.heatmap(np.array(cm), annot=True, fmt='d', cmap='Blues', ax=ax_cm, xticklabels=class_names, yticklabels=class_names)
                            ax_cm.set_title("Matriz de Confusión"); ax_cm.set_ylabel("Verdadero"); ax_cm.set_xlabel("Predicho")
                            st.pyplot(fig_cm)

# ==============================================================================
# --- PESTAÑA 4: DASHBOARD DE MONITOREO ---
# ==============================================================================
if tab_monitoreo is not None:
    with tab_monitoreo:
        st.header("Dashboard de Monitoreo del Sistema")
        if st.button("Refrescar Datos del Dashboard", key="refresh_monitor"): st.rerun()

        if not st.session_state.get('ray_initialized', False):
            st.error("Fallo al conectar con el clúster de Ray.")
            if 'ray_init_error' in st.session_state: st.exception(st.session_state.get('ray_init_error'))
        else:
            with st.expander("Estado del Clúster Ray", expanded=True):
                st.subheader("Nodos del Clúster")
                try:
                    node_data = get_cluster_nodes_status()
                    if node_data: st.table(pd.DataFrame(node_data))
                    else: st.warning("No se pudieron obtener datos de los nodos.")
                except Exception as e: st.error(f"Error obteniendo estado de nodos: {e}")

            with st.expander("Estadísticas de Inferencia (API)", expanded=True):
                st.subheader("Rendimiento del Servicio de Predicción")
                try:
                        # --- LLAMADA A LA NUEVA FUNCIÓN DE AGREGACIÓN ---
                        # Le pasamos el inference_client para que sepa a quiénes preguntar
                        inference_stats = get_aggregated_inference_stats(inference_client)

                        if "error" in inference_stats:
                            st.error(inference_stats["error"])
                        else:
                            col1, col2 = st.columns(2)
                            col1.metric("Total de Peticiones", f"{inference_stats.get('total_requests', 0)}")
                            col2.metric("Latencia Promedio", f"{inference_stats.get('average_latency_ms', 0):.2f} ms")
                            details_df = pd.DataFrame(inference_stats.get("details_by_model", []))
                            if not details_df.empty:
                                st.markdown("**Desglose por Modelo:**")
                                st.table(details_df)
                except Exception as e:
                    st.error(f"No se pudieron obtener las estadísticas de inferencia: {e}")

