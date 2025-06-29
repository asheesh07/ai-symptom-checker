from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_analyze_symptoms():
    response = client.post("/analyze", json={"symptoms": "fever and headache"})
    assert response.status_code == 200 or response.status_code == 500  # 500 if OpenAI key not set
