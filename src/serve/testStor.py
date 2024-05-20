import pytest
import json
import requests
from serviceNormal import appImp

class DateTimeTransformer:
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X['time'] = pd.to_datetime(X['time'])
        X['day'] = X['time'].dt.day
        X['month'] = X['time'].dt.month
        X['year'] = X['time'].dt.year
        X['hour'] = X['time'].dt.hour
        X['minute'] = X['time'].dt.minute
        X = X.drop('time', axis=1)
        return X

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