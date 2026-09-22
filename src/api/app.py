"""
FastAPI Prediction Service for FinTrust ML Pipeline
====================================================
MENTOR NOTE FOR INTERNS:
This service demonstrates how an ML model transitions from an offline experiment
into an online microservice reachable via HTTP endpoints.

Endpoints exposed:
- GET /         : Service root & links
- GET /health   : Health check & model metadata
- POST /predict : Real-time single transaction scoring
- POST /predict/batch : High-throughput batch scoring
"""

from typing import List
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
import pandas as pd

import sys
from pathlib import Path
# Ensure project root is in sys.path when running file directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PATHS, MODEL_CONFIG
from src.api.schemas import (
    TransactionRequest,
    TransactionBatchRequest,
    PredictionResponse,
    BatchPredictionResponse,
    HealthResponse
)
from src.models.predict import FinTrustPredictionPipeline
from src.validation.validator import DataValidationError

# Initialize FastAPI application
app = FastAPI(
    title="FinTrust Transaction Risk Scoring Service",
    description="Production-grade ML microservice for automated banking fraud risk detection.",
    version="1.0.0"
)

# Global pipeline instance (lazy loaded on startup)
pipeline: FinTrustPredictionPipeline = None


@app.on_event("startup")
def startup_event():
    """Load model and preprocessor artifacts into memory when the server boots up."""
    global pipeline
    try:
        pipeline = FinTrustPredictionPipeline(strict_validation=True)
    except FileNotFoundError:
        print("WARNING: Model artifacts not found on startup. Train model before calling /predict.")


@app.exception_handler(DataValidationError)
async def data_validation_exception_handler(request, exc: DataValidationError):
    """Custom exception handler mapping domain validation errors to HTTP 422."""
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "Data Validation Failed",
            "message": str(exc),
            "report": exc.report.summary() if hasattr(exc, "report") else None
        }
    )


@app.get("/", tags=["Info"])
def root():
    return {
        "service": "FinTrust Transaction Risk Scoring API",
        "version": "1.0.0",
        "documentation": "/docs",
        "status": "online"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    """Health check endpoint to monitor model readiness in production."""
    artifacts_ready = (
        PATHS.model_artifact_path.exists() and
        PATHS.preprocessor_artifact_path.exists()
    )
    return HealthResponse(
        status="healthy" if artifacts_ready else "degraded",
        model_version="1.0.0-random_forest",
        decision_threshold=MODEL_CONFIG.classification_threshold,
        artifacts_loaded=artifacts_ready
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict_transaction(request: TransactionRequest):
    """
    Score a single transaction in real time.
    Executes the full 7-stage ML workflow.
    """
    global pipeline
    if pipeline is None:
        try:
            pipeline = FinTrustPredictionPipeline(strict_validation=True)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Model pipeline not initialized: {str(e)}"
            )

    try:
        input_dict = request.dict()
        result_df = pipeline.run_pipeline(input_dict, verbose=False)
        record = result_df.iloc[0].to_dict()
        
        return PredictionResponse(
            Transaction_ID=record["Transaction_ID"],
            Customer_ID=record["Customer_ID"],
            Predicted_Risk_Flag=record["Predicted_Risk_Flag"],
            Risk_Probability=float(record["Risk_Probability"]),
            Risk_Tier=record["Risk_Tier"],
            Operational_Action=record["Operational_Action"],
            Scored_At=record["Scored_At"]
        )
    except DataValidationError as dve:
        raise dve
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(e)}"
        )


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Inference"])
def predict_batch_transactions(request: TransactionBatchRequest):
    """Score a batch of transactions with high throughput."""
    global pipeline
    if pipeline is None:
        try:
            pipeline = FinTrustPredictionPipeline(strict_validation=True)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Model pipeline not initialized: {str(e)}"
            )

    try:
        input_list = [t.dict() for t in request.transactions]
        result_df = pipeline.run_pipeline(input_list, verbose=False)
        
        predictions = []
        for _, row in result_df.iterrows():
            predictions.append(PredictionResponse(
                Transaction_ID=row["Transaction_ID"],
                Customer_ID=row["Customer_ID"],
                Predicted_Risk_Flag=row["Predicted_Risk_Flag"],
                Risk_Probability=float(row["Risk_Probability"]),
                Risk_Tier=row["Risk_Tier"],
                Operational_Action=row["Operational_Action"],
                Scored_At=row["Scored_At"]
            ))
            
        return BatchPredictionResponse(
            total_processed=len(predictions),
            predictions=predictions
        )
    except DataValidationError as dve:
        raise dve
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch inference error: {str(e)}"
        )
