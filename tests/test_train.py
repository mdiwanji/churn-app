import os
import pytest
import pandas as pd
from train import (
    load_data,
    select_features,
    split_data,
    train_model,
    evaluate_model,
    save_model
)

@pytest.fixture
def csv_file():
    """Fixture for the data file path."""
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(BASE_DIR, 'data/customer_churn.csv')

@pytest.fixture
def df(csv_file):
    """Fixture to load the DataFrame."""
    return load_data(csv_file)

@pytest.fixture
def features_and_target(df):
    """Fixture to get features and target."""
    selected_features = ['Age', 'Total_Purchase', 'Account_Manager', 'Years', 'Num_Sites']
    X, y = select_features(df, selected_features, 'Churn')
    return X, y

def test_load_data(csv_file):
    """Test the load_data function."""
    df = load_data(csv_file)
    assert not df.empty, "DataFrame should not be empty"
    assert 'Churn' in df.columns, "'Churn' column should be in DataFrame"

def test_select_features(features_and_target):
    """Test the select_features function."""
    X, y = features_and_target
    assert not X.empty, "Features DataFrame should not be empty"
    assert not y.empty, "Target Series should not be empty"
    assert X.shape[0] == y.shape[0], "Features and target should have the same number of rows"

def test_split_data(features_and_target):
    """Test the split_data function."""
    X, y = features_and_target
    X_train, X_test, y_train, y_test = split_data(X, y)
    assert len(X_train) > 0, "X_train should not be empty"
    assert len(X_test) > 0, "X_test should not be empty"
    assert len(y_train) > 0, "y_train should not be empty"
    assert len(y_test) > 0, "y_test should not be empty"

def test_train_model(features_and_target):
    """Test the train_model function."""
    X, y = features_and_target
    X_train, _, y_train, _ = split_data(X, y)
    model = train_model(X_train, y_train)
    assert model is not None, "Model should not be None"
    assert hasattr(model, 'predict'), "Model should have a 'predict' method"

def test_evaluate_model(features_and_target):
    """Test the evaluate_model function."""
    X, y = features_and_target
    X_train, X_test, y_train, y_test = split_data(X, y)
    model = train_model(X_train, y_train)
    accuracy, recall, f1 = evaluate_model(model, X_test, y_test)
    assert 0 <= accuracy <= 1, "Accuracy should be between 0 and 1"
    assert 0 <= recall <= 1, "Recall should be between 0 and 1"
    assert 0 <= f1 <= 1, "F1-score should be between 0 and 1"

def test_save_model(tmpdir, features_and_target):
    """Test the save_model function."""
    X, y = features_and_target
    X_train, _, y_train, _ = split_data(X, y)
    model = train_model(X_train, y_train)
    filename = tmpdir.join('rf_model_test.pkl')
    save_model(model, str(filename))
    assert os.path.exists(str(filename)), "Model file should exist"
