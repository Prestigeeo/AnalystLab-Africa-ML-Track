"""
FastAPI Prediction Service for FinTrust ML Pipeline (Week 3: Integration-Ready)
==============================================================================
MENTOR NOTE FOR INTERNS:
This service demonstrates how an ML model transitions from an offline experiment
into an online microservice reachable via HTTP endpoints.

Endpoints exposed:
- GET /               : Service root & metadata links
- GET /health         : Health check & model readiness
- GET /model/metadata : Model governance, hyperparameters, and test metrics
- POST /predict       : Real-time transaction scoring with customer cache auto-enrichment
- POST /predict/enriched : Explicitly enriched transaction scoring
- POST /predict/batch : High-throughput batch scoring
"""

import sys
from pathlib import Path
from contextlib import asynccontextmanager
from typing import List, Optional
import json

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
import pandas as pd

# Ensure project root is in sys.path when running file directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PATHS, MODEL_CONFIG
from src.api.schemas import (
    TransactionRequest,
    EnrichedTransactionRequest,
    TransactionBatchRequest,
    PredictionResponse,
    BatchPredictionResponse,
    HealthResponse,
    ModelMetadataResponse
)
from src.models.predict import FinTrustPredictionPipeline
from src.validation.validator import DataValidationError

# Global pipeline instance
pipeline: Optional[FinTrustPredictionPipeline] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Modern lifespan context manager (replaces deprecated on_event).
    Loads model and preprocessor artifacts during server startup.
    """
    global pipeline
    try:
        pipeline = FinTrustPredictionPipeline(strict_validation=True)
    except FileNotFoundError:
        print("WARNING: Model artifacts not found on startup. Train model before scoring.")
    yield
    # Teardown logic if needed
    pipeline = None


# Initialize FastAPI application with lifespan
app = FastAPI(
    title="FinTrust Transaction Risk Scoring Service",
    description="Production-grade ML microservice for automated banking fraud risk triage.",
    version="2.0.0-week3",
    lifespan=lifespan
)


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
        "version": "2.0.0-week3",
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
        model_version="2.0.0-week3-random_forest",
        decision_threshold=MODEL_CONFIG.classification_threshold,
        artifacts_loaded=artifacts_ready
    )


@app.get("/model/metadata", response_model=ModelMetadataResponse, tags=["Governance"])
def get_model_metadata():
    """Returns model governance metadata, performance metrics, and threshold settings."""
    if not PATHS.metadata_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model metadata file not found. Train model first."
        )
    with open(PATHS.metadata_path, "r") as f:
        meta = json.load(f)
    return ModelMetadataResponse(
        model_name=meta.get("model_name", "FinTrust Classifier"),
        model_family=meta.get("model_family", "RandomForest"),
        version=meta.get("version", "2.0.0"),
        author=meta.get("author", "FinTrust MLE Track"),
        features_count=meta.get("features_count", 60),
        classification_threshold=meta.get("classification_threshold", 0.35),
        metrics=meta.get("metrics", {})
    )


def _get_pipeline() -> FinTrustPredictionPipeline:
    global pipeline
    if pipeline is None:
        try:
            pipeline = FinTrustPredictionPipeline(strict_validation=True)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Model pipeline not initialized: {str(e)}"
            )
    return pipeline


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict_transaction(request: TransactionRequest):
    """
    Score a raw transaction in real time with auto-enrichment from customer cache.
    Executes the full 7-stage ML workflow.
    """
    pipe = _get_pipeline()
    try:
        input_dict = request.model_dump() if hasattr(request, "model_dump") else request.dict()
        result_df = pipe.run_pipeline(input_dict, verbose=False)
        record = result_df.iloc[0].to_dict()
        
        return PredictionResponse(
            Transaction_ID=record["Transaction_ID"],
            Customer_ID=record["Customer_ID"],
            Predicted_Risk_Flag=record["Predicted_Risk_Flag"],
            Risk_Probability=float(record["Risk_Probability"]),
            Risk_Tier=record["Risk_Tier"],
            Operational_Action=record["Operational_Action"],
            Model_Version=str(record.get("Model_Version", "2.0.0-week3")),
            Scored_At=str(record["Scored_At"])
        )
    except DataValidationError as dve:
        raise dve
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(e)}"
        )


@app.post("/predict/enriched", response_model=PredictionResponse, tags=["Inference"])
def predict_enriched_transaction(request: EnrichedTransactionRequest):
    """
    Score a transaction with pre-supplied customer demographics.
    """
    pipe = _get_pipeline()
    try:
        input_dict = request.model_dump() if hasattr(request, "model_dump") else request.dict()
        result_df = pipe.run_pipeline(input_dict, verbose=False)
        record = result_df.iloc[0].to_dict()
        
        return PredictionResponse(
            Transaction_ID=record["Transaction_ID"],
            Customer_ID=record["Customer_ID"],
            Predicted_Risk_Flag=record["Predicted_Risk_Flag"],
            Risk_Probability=float(record["Risk_Probability"]),
            Risk_Tier=record["Risk_Tier"],
            Operational_Action=record["Operational_Action"],
            Model_Version=str(record.get("Model_Version", "2.0.0-week3")),
            Scored_At=str(record["Scored_At"])
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
    pipe = _get_pipeline()
    try:
        input_list = [t.model_dump() if hasattr(t, "model_dump") else t.dict() for t in request.transactions]
        result_df = pipe.run_pipeline(input_list, verbose=False)
        
        predictions = []
        for _, row in result_df.iterrows():
            predictions.append(PredictionResponse(
                Transaction_ID=row["Transaction_ID"],
                Customer_ID=row["Customer_ID"],
                Predicted_Risk_Flag=row["Predicted_Risk_Flag"],
                Risk_Probability=float(row["Risk_Probability"]),
                Risk_Tier=row["Risk_Tier"],
                Operational_Action=row["Operational_Action"],
                Model_Version=str(row.get("Model_Version", "2.0.0-week3")),
                Scored_At=str(row["Scored_At"])
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
