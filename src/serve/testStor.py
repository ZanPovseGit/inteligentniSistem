import pytest
import json
import requests
from service import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_predict_endpoint(client):
    url = '/predictLas'
    test_data = {
    "target": 1.0,
    "rain": 0.2,
    "time": "2024-05-15T14:30:00"
    }

    response = client.post(url, json=test_data)

    assert response.status_code == 200

    result = response.get_json()

    assert 'prediction' in result