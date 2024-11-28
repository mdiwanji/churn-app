import os
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score
import pandas as pd
import matplotlib.pyplot as plt
from joblib import dump

# Download the CSV file from the GitHub URL
# Paths for model and scaler
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_file = os.path.join(BASE_DIR, 'data/customer_churn.csv')

# Load the CSV file into a pandas DataFrame
df = pd.read_csv(csv_file)

# Select specified features
selected_features = ['Age', 'Total_Purchase','Account_Manager', 'Years', 'Num_Sites']

# Prepare data
X = df[selected_features]
y = df['Churn']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the parameter grid for hyperparameter tuning
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10]
}

# Create a RandomForestClassifier
rf_classifier = RandomForestClassifier(random_state=42)

# Use GridSearchCV to find the best hyperparameters
grid_search = GridSearchCV(estimator=rf_classifier, param_grid=param_grid, cv=5, scoring='f1') #using f1 as scoring metric
grid_search.fit(X_train, y_train)

# Get the best model
best_rf_classifier = grid_search.best_estimator_

# Make predictions using the best model
y_pred = best_rf_classifier.predict(X_test)

# Evaluate the best model
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Best Hyperparameters: {grid_search.best_params_}")
print(f"Accuracy: {accuracy}")
print(f"Recall: {recall}")
print(f"F1-score: {f1}")

# Get the best model
best_rf_classifier = grid_search.best_estimator_

# Save the best model as a pickle file
filename = 'rf_model.pkl'
dump(best_rf_classifier, filename)