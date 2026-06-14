import asyncio
from src.api.main import app
from fastapi.testclient import TestClient

client = TestClient(app)

response = client.post("/extract", data={"text": "Urgent shipment from Delhi to Pune."})
print(response.status_code)
print(response.json() if response.status_code == 200 else response.text)
