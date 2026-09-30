"""
Pydantic Schemas for FinTrust Prediction API (Week 3: Integration-Ready)
=======================================================================
MENTOR NOTE FOR INTERNS:
In enterprise ML serving, schemas define the strict API contract:
1. `TransactionRequest`: Raw transaction payload scored in real-time with customer store auto-enrichment.
2. `EnrichedTransactionRequest`: Directly supplies both transaction attributes AND customer demographics.
3. `PredictionResponse`: Standardized model output with risk flags, calibrated probability, tier, and recommendation.
4. `ModelMetadataResponse`: Model governance and audit contract (Part C).
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class TransactionRequest(BaseModel):
    """Schema for incoming raw transaction scoring (auto-enriches customer profile)."""
    Transaction_ID: str = Field(..., description="Unique transaction ID", json_schema_extra={"example": "FT-T100001"})
    Customer_ID: str = Field(..., description="Customer ID matching customer master cache", json_schema_extra={"example": "FT-C00001"})
    Transaction_DateTime: float = Field(..., description="Timestamp or Excel serial date", json_schema_extra={"example": 46023.15})
    Transaction_Type: str = Field(..., description="Type of transaction", json_schema_extra={"example": "Transfer"})
    Amount_NGN: float = Field(..., gt=0, description="Amount in Nigerian Naira (> 0)", json_schema_extra={"example": 25000.0})
    Channel: str = Field(..., description="Transaction channel", json_schema_extra={"example": "Mobile App"})
    Device_Type: Optional[str] = Field("Android", description="Device type", json_schema_extra={"example": "Android"})
    Location: Optional[str] = Field("Lagos", description="Geographic location", json_schema_extra={"example": "Lagos"})
    International_Transaction: str = Field("No", description="'Yes' or 'No'", json_schema_extra={"example": "No"})
    Transaction_Status: str = Field("Successful", description="Status", json_schema_extra={"example": "Successful"})


class EnrichedTransactionRequest(TransactionRequest):
    """Schema for explicitly enriched transaction scoring."""
    Age: float = Field(35.0, description="Customer age (18-100)", json_schema_extra={"example": 35.0})
    Tenure_Months: float = Field(24.0, description="Account tenure in months", json_schema_extra={"example": 24.0})
    Digital_Engagement_Score: float = Field(75.0, description="Engagement score (0-100)", json_schema_extra={"example": 75.0})
    Gender: str = Field("Female", description="Gender", json_schema_extra={"example": "Female"})
    Customer_Segment: str = Field("Everyday", description="Segment", json_schema_extra={"example": "Everyday"})
    Account_Type: str = Field("Savings", description="Account type", json_schema_extra={"example": "Savings"})
    Monthly_Income_Band: str = Field("250k-499k", description="Income band", json_schema_extra={"example": "250k-499k"})
    Preferred_Channel: str = Field("Mobile App", description="Preferred channel", json_schema_extra={"example": "Mobile App"})
    Account_Status: str = Field("Active", description="Account status", json_schema_extra={"example": "Active"})


class TransactionBatchRequest(BaseModel):
    """Schema for batch scoring request."""
    transactions: List[TransactionRequest]


class PredictionResponse(BaseModel):
    """Standardized prediction response schema."""
    Transaction_ID: str
    Customer_ID: str
    Predicted_Risk_Flag: str = Field(..., description="'Yes' (Flagged for Review) or 'No' (Legitimate)")
    Risk_Probability: float = Field(..., description="Risk probability score (0.0 to 1.0)")
    Risk_Tier: str = Field(..., description="Low, Medium, or High risk tier")
    Operational_Action: str = Field(..., description="Recommended operations action")
    Model_Version: str = Field(..., description="Model version identifier")
    Scored_At: str = Field(..., description="ISO 8601 scoring timestamp")


class BatchPredictionResponse(BaseModel):
    """Batch prediction response schema."""
    total_processed: int
    predictions: List[PredictionResponse]


class HealthResponse(BaseModel):
    """Service health and model status."""
    status: str
    model_version: str
    decision_threshold: float
    artifacts_loaded: bool


class ModelMetadataResponse(BaseModel):
    """Model governance metadata response (Part C)."""
    model_name: str
    model_family: str
    version: str
    author: str
    features_count: int
    classification_threshold: float
    metrics: Dict[str, Any]
