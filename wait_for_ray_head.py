# wait_for_ray_head.py
import socket
import time
import sys

host = "ray-head"
port = 6379
retries = 45  # Unos 90 segundos de intentos
wait_seconds_between_retries = 2
socket_timeout = 2

print(f'WAIT_SCRIPT: Iniciando espera para {host}:{port}...')

for i in range(retries):
    s = None  # Inicializar s en cada iteración del bucle
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(socket_timeout)
        print(f"WAIT_SCRIPT: Intento {i+1}/{retries}: Conectando a {host}:{port}...")
        s.connect((host, port))
        # Si llegamos aquí, la conexión fue exitosa
        print(f'WAIT_SCRIPT: Intento {i+1}/{retries}: {host}:{port} está ACCESIBLE!')
        sys.exit(0)  # Salir con código 0 (éxito)
    except socket.timeout:
        print(f'WAIT_SCRIPT: Intento {i+1}/{retries}: Timeout conectando a {host}:{port}. Reintentando en {wait_seconds_between_retries}s...')
    except ConnectionRefusedError:
        print(f'WAIT_SCRIPT: Intento {i+1}/{retries}: Conexión rechazada por {host}:{port}. Reintentando en {wait_seconds_between_retries}s...')
    except socket.gaierror:
        print(f'WAIT_SCRIPT: Intento {i+1}/{retries}: Error resolviendo nombre de host {host}. Reintentando en {wait_seconds_between_retries}s...')
    except Exception as e:
        print(f'WAIT_SCRIPT: Intento {i+1}/{retries}: Error inesperado ({type(e).__name__}: {e}). Reintentando en {wait_seconds_between_retries}s...')
    finally:
        if s:
            s.close() # Asegurarse de cerrar el socket
    
    time.sleep(wait_seconds_between_retries)

print(f'WAIT_SCRIPT: ERROR: {host}:{port} no estuvo disponible después de {retries} intentos.')
sys.exit(1) # Salir con código 1 (error)