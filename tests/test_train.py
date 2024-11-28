import os
import pandas as pd
from joblib import load

# Define the features used by the model
SELECTED_FEATURES = ['Age', 'Total_Purchase', 'Account_Manager', 'Years', 'Num_Sites']

def load_model(model_path='rf_model.pkl'):
    """Load the pre-trained model from a file."""
    model = load(model_path)
    return model

def prepare_data(input_data):
    """Prepare input data for prediction."""
    # Ensure input_data is a DataFrame
    if isinstance(input_data, dict):
        input_data = pd.DataFrame([input_data])
    elif isinstance(input_data, list):
        input_data = pd.DataFrame(input_data)
    elif not isinstance(input_data, pd.DataFrame):
        raise ValueError("Input data must be a dict, list of dicts, or a pandas DataFrame.")

    # Check if all required features are present
    missing_features = [feature for feature in SELECTED_FEATURES if feature not in input_data.columns]
    if missing_features:
        raise ValueError(f"Missing features: {missing_features}")

    # Select and reorder the required features
    input_data = input_data[SELECTED_FEATURES]

    return input_data

def make_prediction(model, input_data):
    """Make predictions using the loaded model."""
    predictions = model.predict(input_data)
    probabilities = model.predict_proba(input_data)
    return predictions, probabilities

def main():
    # Load the model
    model = load_model()

    # Example input data
    new_data = [
        {'Age': 30, 'Total_Purchase': 5000, 'Account_Manager': 0, 'Years': 2, 'Num_Sites': 8},
        {'Age': 45, 'Total_Purchase': 15000, 'Account_Manager': 1, 'Years': 10, 'Num_Sites': 15}
    ]

    # Prepare the data
    input_data = prepare_data(new_data)

    # Make predictions
    predictions, probabilities = make_prediction(model, input_data)

    # Display the results
    for i in range(len(input_data)):
        print(f"Client {i+1}:")
        print(f"  Input Data: {input_data.iloc[i].to_dict()}")
        print(f"  Prediction: {'Churn' if predictions[i] == 1 else 'No Churn'}")
        print(f"  Probability of Churn: {probabilities[i][1]:.2f}")
        print("---")

if __name__ == '__main__':
    main()
