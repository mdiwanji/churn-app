import pytest
from app import app
import json

@pytest.fixture
def client():
    with app.test_client() as client:
        yield client

def test_predict_valid_input(client):
    """Test du endpoint POST '/predict' avec des données valides."""
    data = {
        'Age': 35,
        'Total_Purchase': 10000,
        'Account_Manager': 1,
        'Years': 5,
        'Num_Sites': 8
    }
    response = client.post('/predict', json=data)
    assert response.status_code == 200
    response_json = response.get_json()
    assert 'prediction' in response_json
    assert 'probability' in response_json
    assert '0' in response_json['probability']
    assert '1' in response_json['probability']

def test_predict_missing_fields(client):
    """Test du endpoint POST '/predict' avec des champs manquants."""
    data = {
        'Age': 35,
        'Total_Purchase': 10000,
        'Account_Manager': 1,
        # 'Years' est manquant
        'Num_Sites': 8
    }
    response = client.post('/predict', json=data)
    assert response.status_code == 400
    response_json = response.get_json()
    assert 'error' in response_json

def test_predict_invalid_data(client):
    """Test du endpoint POST '/predict' avec des données invalides."""
    data = {
        'Age': 'trente-cinq',  # Type invalide
        'Total_Purchase': 10000,
        'Account_Manager': 1,
        'Years': 5,
        'Num_Sites': 8
    }
    response = client.post('/predict', json=data)
    assert response.status_code in [400, 500]
    response_json = response.get_json()
    assert 'error' in response_json
