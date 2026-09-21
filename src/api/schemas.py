"""
Pydantic Schemas for FinTrust Prediction API
=============================================
MENTOR NOTE FOR INTERNS:
In modern web-scale ML architectures (FastAPI, Flask, etc.), incoming HTTP requests
are validated at the network boundary using Pydantic.

This gives us:
1. Automatic type coercion and format validation before our ML code even touches the data.
2. Auto-generated OpenAPI / Swagger documentation (`/docs`).
3. Meaningful HTTP 422 Unprocessable Entity responses for invalid client requests.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class TransactionRequest(BaseModel):
    """Schema for an incoming transaction scoring request."""
    Transaction_ID: str = Field(..., description="Unique alphanumeric transaction identifier", example="FT-T100001")
    Customer_ID: str = Field(..., description="Unique customer identifier", example="FT-C00234")
    Transaction_DateTime: float = Field(..., description="Transaction timestamp (Excel serial date or days)", example=46023.15)
    Transaction_Type: str = Field(..., description="Type of transaction", example="Transfer")
    Amount_NGN: float = Field(..., gt=0, description="Transaction amount in Nigerian Naira (must be > 0)", example=25000.0)
    Channel: str = Field(..., description="Transaction channel", example="Mobile App")
    Device_Type: Optional[str] = Field("Android", description="User device type", example="Android")
    Location: Optional[str] = Field("Lagos", description="Geographic location of transaction", example="Lagos")
    International_Transaction: str = Field("No", description="'Yes' or 'No'", example="No")
    Transaction_Status: str = Field("Successful", description="Status of transaction", example="Successful")


class TransactionBatchRequest(BaseModel):
    """Schema for batch scoring request."""
    transactions: List[TransactionRequest]


class PredictionResponse(BaseModel):
    """Schema for single transaction prediction response."""
    Transaction_ID: str
    Customer_ID: str
    Predicted_Risk_Flag: str = Field(..., description="'Yes' (Flagged for Review) or 'No' (Normal)")
    Risk_Probability: float = Field(..., description="Predicted risk probability between 0.0 and 1.0")
    Risk_Tier: str = Field(..., description="Categorical risk tier: Low, Medium, or High")
    Operational_Action: str = Field(..., description="Recommended banking ops action")
    Scored_At: str


class BatchPredictionResponse(BaseModel):
    """Schema for batch prediction response."""
    total_processed: int
    predictions: List[PredictionResponse]


class HealthResponse(BaseModel):
    """Service health and model metadata."""
    status: str
    model_version: str
    decision_threshold: float
    artifacts_loaded: bool
