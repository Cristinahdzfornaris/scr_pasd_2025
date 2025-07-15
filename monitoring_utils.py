# monitoring_utils.py (VERSIÓN CORREGIDA Y SIMPLIFICADA)
import ray
import requests
import pandas as pd
from prometheus_client.parser import text_string_to_metric_families

def get_cluster_nodes_status():
    """Consulta el estado de los nodos del clúster de Ray."""
    if not ray.is_initialized(): return None
    
    nodes = ray.nodes()
    status_list = []
    for node in nodes:
        resources = node.get("Resources", {})
        status_list.append({
            "Node ID": node.get("NodeID", "N/A")[:12],
            "Estado": "VIVO" if node.get("Alive") else "MUERTO",
            "IP": node.get("NodeManagerAddress", "N/A"),
            "CPUs (Total)": resources.get("CPU", 0),
            "Memoria (GB)": f"{node.get('memory_total_bytes', 0) / (1024**3):.2f}",
            "Object Store (GB)": f"{node.get('object_store_memory_bytes', 0) / (1024**3):.2f}",
        })
    return status_list

def get_actor_status(actor_name="model_registry", namespace="mi_plataforma"):
    """
    Verifica el estado de un actor nombrado de una manera más robusta,
    sin usar ray.state.
    """
    if not ray.is_initialized():
        return {"Estado": "Ray no inicializado", "Vivo": "❓"}
    
    try:
        # El simple hecho de poder obtener el actor significa que está vivo.
        # Si está muerto o no existe, ray.get_actor() lanzará un ValueError.
        actor = ray.get_actor(actor_name, namespace=namespace)
        
        # Podemos hacer una llamada simple a un método (ping) para confirmar que responde.
        # Añadamos un método 'ping' al actor para esto.
        # Por ahora, asumimos que si get_actor tiene éxito, está vivo.
        return {
            "Nombre": actor_name,
            "Estado": "ALIVE",
            # No podemos obtener fácilmente el Node ID o los reinicios sin ray.state,
            # así que simplificamos la salida.
            "Reinicios": "N/A (API simplificada)",
            "Vivo": "✅"
        }
    except ValueError:
        # Esto ocurre si el actor no se encuentra (puede estar muerto o nunca se creó).
        return {"Estado": "❌ No Encontrado / Muerto", "Vivo": "❌"}

# La función get_inference_stats se mantiene igual
def get_inference_stats(client):
    """
    Obtiene las estadísticas de inferencia desde el endpoint /metrics
    utilizando un ResilientClient para alta disponibilidad.
    """
    try:
        # En lugar de requests.get(metrics_url), usamos nuestro cliente inteligente.
        # El endpoint para las métricas de Prometheus es siempre '/metrics'.
        response = client.make_request("GET", "/metrics", timeout=5)
        
        stats = {
            "total_requests": 0,
            "average_latency_ms": 0.0,
            "details_by_model": []
        }
        
        model_data = {}
        # El resto de la lógica para parsear las métricas es la misma.
        for family in text_string_to_metric_families(response.text):
            if family.name == "inference_latency_seconds_count":
                for sample in family.samples:
                    model_key = tuple(sorted(sample.labels.items()))
                    if model_key not in model_data:
                        model_data[model_key] = {"count": 0, "sum": 0.0}
                    model_data[model_key]["count"] += sample.value
            elif family.name == "inference_latency_seconds_sum":
                 for sample in family.samples:
                    model_key = tuple(sorted(sample.labels.items()))
                    if model_key not in model_data:
                        model_data[model_key] = {"count": 0, "sum": 0.0}
                    model_data[model_key]["sum"] += sample.value

        total_latency_sum = sum(data["sum"] for data in model_data.values())
        stats["total_requests"] = int(sum(data["count"] for data in model_data.values()))

        for model_key, data in model_data.items():
            labels_dict = dict(model_key)
            model_name = f"{labels_dict.get('dataset', 'N/A')}/{labels_dict.get('model_type', 'N/A')}"
            stats["details_by_model"].append({
                "Modelo": model_name, "Peticiones": int(data["count"]),
                "Latencia Media (ms)": (data["sum"] / data["count"]) * 1000 if data["count"] > 0 else 0
            })
        if stats["total_requests"] > 0:
            stats["average_latency_ms"] = (total_latency_sum / stats["total_requests"]) * 1000
        return stats
    except Exception as e:
        return {"error": f"No se pudo obtener las métricas de inferencia: {e}"}

def get_aggregated_inference_stats(client):
    """
    Obtiene métricas de TODAS las réplicas de un servicio y las agrega.
    """
    # Usamos la lista de servidores del cliente resiliente
    all_servers = client.servers
    if not all_servers:
        return {"error": "No se encontraron servidores para el servicio de inferencia."}

    aggregated_stats = {
        "total_requests": 0,
        "total_latency_sum": 0.0,
        "details_by_model": {} # Usamos un dict para agregar por modelo
    }
    
    # Iteramos sobre cada servidor descubierto
    for server_url in all_servers:
        try:
            # Hacemos una petición directa a cada réplica
            response = requests.get(f"{server_url}/metrics", timeout=2)
            response.raise_for_status()
            
            # Parseamos la respuesta de esta réplica específica
            for family in text_string_to_metric_families(response.text):
                if family.name == "inference_latency_seconds_count":
                    for sample in family.samples:
                        # Creamos una clave única para cada modelo/dataset
                        model_key = f"{sample.labels.get('dataset', 'N/A')}/{sample.labels.get('model_type', 'N/A')}"
                        if model_key not in aggregated_stats["details_by_model"]:
                            aggregated_stats["details_by_model"][model_key] = {"count": 0, "sum": 0.0}
                        aggregated_stats["details_by_model"][model_key]["count"] += sample.value
                elif family.name == "inference_latency_seconds_sum":
                    for sample in family.samples:
                        model_key = f"{sample.labels.get('dataset', 'N/A')}/{sample.labels.get('model_type', 'N/A')}"
                        if model_key not in aggregated_stats["details_by_model"]:
                            aggregated_stats["details_by_model"][model_key] = {"count": 0, "sum": 0.0}
                        aggregated_stats["details_by_model"][model_key]["sum"] += sample.value

        except requests.exceptions.RequestException as e:
            print(f"No se pudo obtener métricas de la réplica {server_url}: {e}")
            # Continuamos con la siguiente réplica si una falla

    # Ahora calculamos los totales a partir de los datos agregados
    total_reqs = sum(data["count"] for data in aggregated_stats["details_by_model"].values())
    total_latency = sum(data["sum"] for data in aggregated_stats["details_by_model"].values())
    
    final_stats = {
        "total_requests": int(total_reqs),
        "average_latency_ms": (total_latency / total_reqs) * 1000 if total_reqs > 0 else 0,
        "details_by_model": [
            {
                "Modelo": model_key,
                "Peticiones": int(data["count"]),
                "Latencia Media (ms)": (data["sum"] / data["count"]) * 1000 if data["count"] > 0 else 0
            } for model_key, data in aggregated_stats["details_by_model"].items()
        ]
    }
    
    return final_stats