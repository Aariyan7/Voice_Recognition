from fastapi.testclient import TestClient
from main import app, API_KEY

client = TestClient(app)

def test_process_data_authorized():
    data = {"name": "test_user", "value": 42}
    headers = {"x-api-key": API_KEY}
    response = client.post("/process-data", json=data, headers=headers)
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert response.json()["received_input"] == data

def test_process_data_unauthorized():
    data = {"name": "test_user", "value": 42}
    headers = {"x-api-key": "WRONG_KEY"}
    response = client.post("/process-data", json=data, headers=headers)
    assert response.status_code == 401

def test_process_data_missing_key():
    data = {"name": "test_user", "value": 42}
    response = client.post("/process-data", json=data)
    assert response.status_code == 422 # FastAPI returns 422 for missing required header/params
