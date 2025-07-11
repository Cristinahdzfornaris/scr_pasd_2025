FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# No se necesita un CMD específico ya que todos los servicios lo anulan en docker-compose.yml