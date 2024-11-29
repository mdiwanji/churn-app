# Base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy application files
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Exposer le port utilisé par Flask
EXPOSE 5000

# Exécuter le script pour entraîner le modèle et générer churn_model_clean.pkl
RUN python train.py

# Commande pour démarrer l'application Flask
CMD ["python", "app.py"]
