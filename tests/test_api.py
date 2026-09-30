"""
Technical Unit Tests: FastAPI Microservice (Week 3 - Part E)
============================================================
MENTOR NOTE FOR INTERNS:
API tests verify network endpoints, HTTP status codes, and JSON response schemas.
We test:
1. GET /health
2. GET /model/metadata
3. POST /predict (Raw transaction + auto-enrichment)
4. POST /predict/enriched (Enriched transaction)
5. POST /predict/batch (Batch scoring)
6. Error handling (HTTP 422 for invalid payloads)
"""

import pytest
from fastapi.testclient import TestClient

from src.api.app import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client):
    """Verifies GET /health returns 200 and healthy status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["artifacts_loaded"] is True


def test_model_metadata_endpoint(client):
    """Verifies GET /model/metadata returns governance info."""
    response = client.get("/model/metadata")
    assert response.status_code == 200
    data = response.json()
    assert "version" in data
    assert "metrics" in data
    assert data["features_count"] == 60


def test_predict_endpoint_auto_enrichment(client):
    """Verifies POST /predict scores a raw transaction with auto-enrichment."""
    payload = {
        "Transaction_ID": "FT-API-001",
        "Customer_ID": "FT-C00001",
        "Transaction_DateTime": 46023.25,
        "Transaction_Type": "Transfer",
        "Amount_NGN": 50000.0,
        "Channel": "Mobile App",
        "Device_Type": "Android",
        "Location": "Lagos",
        "International_Transaction": "No",
        "Transaction_Status": "Successful"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["Transaction_ID"] == "FT-API-001"
    assert data["Predicted_Risk_Flag"] in ["Yes", "No"]
    assert 0.0 <= data["Risk_Probability"] <= 1.0
    assert data["Risk_Tier"] in ["Low", "Medium", "High"]


def test_predict_enriched_endpoint(client):
    """Verifies POST /predict/enriched accepts full demographic payload."""
    payload = {
        "Transaction_ID": "FT-API-002",
        "Customer_ID": "FT-C00002",
        "Transaction_DateTime": 46023.25,
        "Transaction_Type": "Card Purchase",
        "Amount_NGN": 125000.0,
        "Channel": "POS",
        "Device_Type": "POS Terminal",
        "Location": "Abuja",
        "International_Transaction": "No",
        "Transaction_Status": "Successful",
        "Age": 42.0,
        "Tenure_Months": 30.0,
        "Digital_Engagement_Score": 82.0,
        "Gender": "Female",
        "Customer_Segment": "Premium",
        "Account_Type": "Current",
        "Monthly_Income_Band": "500k-999k",
        "Preferred_Channel": "Web",
        "Account_Status": "Active"
    }
    response = client.post("/predict/enriched", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["Transaction_ID"] == "FT-API-002"
    assert "Predicted_Risk_Flag" in data


def test_predict_batch_endpoint(client):
    """Verifies POST /predict/batch scores multiple transactions."""
    payload = {
        "transactions": [
            {
                "Transaction_ID": "FT-BATCH-1",
                "Customer_ID": "FT-C00001",
                "Transaction_DateTime": 46023.25,
                "Transaction_Type": "Transfer",
                "Amount_NGN": 10000.0,
                "Channel": "Mobile App",
                "Device_Type": "Android",
                "Location": "Lagos",
                "International_Transaction": "No",
                "Transaction_Status": "Successful"
            },
            {
                "Transaction_ID": "FT-BATCH-2",
                "Customer_ID": "FT-C00002",
                "Transaction_DateTime": 46023.25,
                "Transaction_Type": "Cash Withdrawal",
                "Amount_NGN": 20000.0,
                "Channel": "ATM",
                "Device_Type": "ATM Terminal",
                "Location": "Kano",
                "International_Transaction": "No",
                "Transaction_Status": "Successful"
            }
        ]
    }
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_processed"] == 2
    assert len(data["predictions"]) == 2


def test_predict_invalid_input_validation(client):
    """Verifies invalid input (negative amount) returns HTTP 422."""
    payload = {
        "Transaction_ID": "FT-API-BAD",
        "Customer_ID": "FT-C00001",
        "Transaction_DateTime": 46023.25,
        "Transaction_Type": "Transfer",
        "Amount_NGN": -5000.0,  # Negative amount rejected by Pydantic gt=0
        "Channel": "Mobile App",
        "Device_Type": "Android",
        "Location": "Lagos",
        "International_Transaction": "No",
        "Transaction_Status": "Successful"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
