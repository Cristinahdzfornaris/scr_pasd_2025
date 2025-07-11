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
def get_inference_stats(metrics_url: str):
    try:
        response = requests.get(metrics_url, timeout=5)
        response.raise_for_status()
        
        stats = {
            "total_requests": 0,
            "average_latency_ms": 0.0,
            "details_by_model": []
        }
        
        model_data = {}
        for family in text_string_to_metric_families(response.text):
            if family.name == "inference_latency_seconds":
                for sample in family.samples:
                    model_key = tuple(sorted(sample.labels.items()))
                    if model_key not in model_data:
                        model_data[model_key] = {"count": 0, "sum": 0.0}
                    if sample.name.endswith("_count"): model_data[model_key]["count"] += sample.value
                    if sample.name.endswith("_sum"): model_data[model_key]["sum"] += sample.value

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
    except requests.exceptions.RequestException as e:
        return {"error": f"No se pudo conectar a la API de métricas: {e}"}