import pytest
import json
import requests
from "C:\Users\Uporabnik\Desktop\notebooks\inteligVaje\src\serve\service.py" import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_predict_endpoint(client):

    test_data = {
        'day': 1,
        'month': 3,
        'year': 2022,
        'hour': 12,
        'minute': 30,
        'relative_humidity': 50,
        'dew_point': 10,
        'apparent_temperature': 25,
        'precipitation_probability': 0.1,
        'rain': 1,
        'surface_pressure': 1015,
        'bike_stands': 10,
        'temperature': 20
    }

    response = client.post('/predict', json=test_data)

    assert response.status_code == 200

    result = response.get_json()

    assert 'predictions' in result