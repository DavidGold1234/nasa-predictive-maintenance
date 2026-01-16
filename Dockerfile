FROM python:3.11-slim

# Evita crear archivos .pyc y buffer de logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Instalar dependencias del sistema (opcional, útil para pandas, psycopg2, etc.)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements
COPY requirements.txt .

# Instalar dependencias Python
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copiar código
COPY src/ ./src/

# Comando por defecto
CMD ["python", "src/main.py"]