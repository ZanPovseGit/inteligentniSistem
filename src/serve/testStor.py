import pytest
import json
import requests
from serviceNormal import appImp
from serviceNormal import DateTimeTransformer
import pandas as pd

@pytest.fixture
def client():
    appImp.config['TESTING'] = True
    with appImp.test_client() as client:
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