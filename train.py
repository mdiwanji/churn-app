import os
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score
import pandas as pd
from joblib import dump

def load_data(csv_file):
    """Load data from a CSV file into a pandas DataFrame."""
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"CSV file not found at path: {csv_file}")
    df = pd.read_csv(csv_file)
    return df

def select_features(df, selected_features, target_column):
    """Select features and target from DataFrame."""
    X = df[selected_features]
    y = df[target_column]
    return X, y

def split_data(X, y, test_size=0.2, random_state=42):
    """Split the data into training and testing sets."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def train_model(X_train, y_train, random_state=42, n_estimators=200, max_depth=5):
    """Train the RandomForestClassifier model."""
    rf_classifier = RandomForestClassifier(
        random_state=random_state,
        n_estimators=n_estimators,
        max_depth=max_depth
    )
    rf_classifier.fit(X_train, y_train)
    return rf_classifier

def evaluate_model(model, X_test, y_test):
    """Evaluate the model's performance."""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    return accuracy, recall, f1

def save_model(model, filename):
    """Save the trained model to a file."""
    dump(model, filename)

def main():
    # Paths for model and data
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    csv_file = os.path.join(BASE_DIR, 'data/customer_churn.csv')

    # Load data
    df = load_data(csv_file)

    # Select features and target
    selected_features = ['Age', 'Total_Purchase', 'Account_Manager', 'Years', 'Num_Sites']
    X, y = select_features(df, selected_features, 'Churn')

    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Train model
    model = train_model(X_train, y_train)

    # Evaluate model
    accuracy, recall, f1 = evaluate_model(model, X_test, y_test)
    print(f"Accuracy: {accuracy}")
    print(f"Recall: {recall}")
    print(f"F1-score: {f1}")

    # Save model
    filename = 'rf_model.pkl'
    save_model(model, filename)

if __name__ == '__main__':
    main()
