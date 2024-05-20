import pytest
import json
import requests
from serviceDruga import app
import pandas as pd

@pytest.fixture
def client():
    with app.test_client() as client:
        yield client

def test_predict_endpoint(client):
    try:
        url = '/predict'
        test_data = {
            "target": 1.0,
            "rain": 0.2,
            "time": "2024-05-15T14:30:00"
        }

        response = client.get(url)

        assert response.status_code == 200

        result = response.get_json()

        assert 'prediction' in result

    except Exception as e:
        pytest.fail(f"An unexpected error occurred: {str(e)}")
