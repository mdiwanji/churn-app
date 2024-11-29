# Base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt /app/

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application files, including tests
COPY . /app

# Set environment variable PYTHONPATH to /app
ENV PYTHONPATH /app

# Exécuter le script pour entraîner le modèle et générer rf_model.pkl
RUN python train.py

# Exposer le port utilisé par Flask
EXPOSE 5000

# Commande pour démarrer l'application Flask
CMD ["python", "app.py", "--host=0.0.0.0", "--port=5000"]
