# wait_for_services.py
import socket
import time
import sys

def wait_for_service(host, port, service_name):
    """Intenta conectarse a un servicio host:port hasta que tenga éxito o se agote el tiempo."""
    print(f'WAIT_SCRIPT: Esperando a {service_name} en {host}:{port}...')
    
    # Intentar por ~90 segundos (45 intentos * 2s de espera + timeout)
    for i in range(45):
        try:
            # Usar 'with' asegura que el socket se cierre automáticamente
            with socket.create_connection((host, port), timeout=2):
                print(f'WAIT_SCRIPT: ¡{service_name} está accesible!')
                return True
        except (socket.timeout, ConnectionRefusedError, socket.gaierror) as e:
            # ConnectionRefusedError es común si el servidor aún no está escuchando
            print(f'WAIT_SCRIPT: Intento {i+1}/45, {service_name} aún no está listo ({type(e).__name__})...')
            time.sleep(2)
        except Exception as e_generic:
             print(f'WAIT_SCRIPT: Intento {i+1}/45: Error de socket inesperado ({type(e_generic).__name__}: {e_generic})...')
             time.sleep(2)

    print(f'WAIT_SCRIPT ERROR: {service_name} no estuvo disponible después del tiempo de espera.')
    return False

if __name__ == "__main__":
    # Verificar Management API
    if not wait_for_service('management-api', 9000, 'Management API'):
        sys.exit(1) # Salir con error si no se puede conectar
    
    # Verificar Inference API
    if not wait_for_service('api-service', 8000, 'Inference API'):
        sys.exit(1)
    
    print("WAIT_SCRIPT: Todas las APIs dependientes están listas.")