from flask import Flask, request, jsonify
from joblib import load
import os

app = Flask(__name__)

# Define the base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Construct paths for model and scaler
model_path = os.path.join(BASE_DIR, 'rf_model.pkl')
# Load model
model = load(model_path)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Récupérer les données JSON de la requête
        data = request.get_json(force=True)

        # Vérifier que toutes les caractéristiques requises sont présentes
        required_features = ['Age', 'Total_Purchase', 'Account_Manager', 'Years', 'Num_Sites']
        if not all(feature in data for feature in required_features):
            missing = [feature for feature in required_features if feature not in data]
            return jsonify({'error': f'The following information are missing : {missing}'}), 400

        # Extraire les caractéristiques du JSON
        input_data = [data[feature] for feature in required_features]

        # Convertir les données en format attendu par le modèle
        input_array = [input_data]

        # Faire la prédiction
        prediction = model.predict(input_array)
        prediction_proba = model.predict_proba(input_array)

        # Préparer la réponse
        response = {
            'prediction': int(prediction[0]),
            'probability': {
                '0': round(prediction_proba[0][0], 2),
                '1': round(prediction_proba[0][1], 2)
            }
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
