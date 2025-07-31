# Usa una imagen oficial de Python como base
FROM python:3.10-slim

# Establece el directorio de trabajo dentro del contenedor
WORKDIR /app

# Instala las dependencias del sistema necesarias para compilar dlib
RUN apt-get update && apt-get install -y cmake build-essential

# Copia el archivo de requerimientos y los instala
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia el resto del código de tu aplicación al contenedor
COPY . .

# Expone el puerto que usará Gunicorn
EXPOSE 10000

# El comando que se ejecutará para iniciar tu aplicación
CMD ["gunicorn", "--workers", "2", "--bind", "0.0.0.0:10000", "app:app"]