from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging

from api.routes import router
from config.settings import config

logger = logging.getLogger(__name__)

def create_app() -> FastAPI:
    """Create FastAPI application"""
    app = FastAPI(
        title="Advanced Sepsis Detection API",
        version="2.0.0",
        description="Advanced ensemble learning system for sepsis detection with CNN and XGBoost",
        docs_url="/docs",
        redoc_url="/redoc"
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include API routes
    app.include_router(router, prefix="/api/v1")

    @app.get("/")
    async def root():
        return {
            "message": "Advanced Sepsis Detection API v2.0", 
            "status": "running",
            "docs_url": "/docs",
            "health_check": "/api/v1/health"
        }

    @app.on_event("startup")
    async def startup_event():
        """Startup event"""
        logger.info("Starting Sepsis Detection API...")
        
    @app.on_event("shutdown")
    async def shutdown_event():
        """Shutdown event"""
        logger.info("Shutting down Sepsis Detection API...")

    return app

def start_api():
    """Start the API server"""
    logger.info(f"Starting API server on {config.API_HOST}:{config.API_PORT}")
    app = create_app()
    uvicorn.run(
        app, 
        host=config.API_HOST, 
        port=config.API_PORT, 
        log_level="info",
        reload=False
    )

# The FastAPI app instance for direct uvicorn use
app = create_app()

if __name__ == "__main__":
    start_api()

# ==============================================================================
# api/routes.py (COMPLETO)
# ==============================================================================

from fastapi import APIRouter, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
import pandas as pd
import numpy as np
import io
import joblib
from pathlib import Path
import logging
import json
import os
from typing import Optional

from api.models import PredictionRequest, PredictionResponse, TrainingStatus, ModelInfo

logger = logging.getLogger(__name__)

router = APIRouter()

# Global variables for model state
ensemble_model: Optional[object] = None
data_loader: Optional[object] = None
training_status = {"status": "idle", "progress": 0.0, "message": "Ready"}

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    model_loaded = ensemble_model is not None
    return {
        "status": "healthy",
        "model_loaded": model_loaded,
        "training_status": training_status,
        "timestamp": pd.Timestamp.now().isoformat()
    }

@router.post("/predict", response_model=PredictionResponse)
async def predict_sepsis(request: PredictionRequest):
    """Make sepsis prediction for individual patient"""
    if ensemble_model is None:
        raise HTTPException(status_code=400, detail="Model not loaded. Please train or load a model first.")
    
    try:
        # Validate input dimensions
        if not request.features or not request.features[0]:
            raise HTTPException(status_code=400, detail="Empty features provided")
        
        # Convert input to numpy array
        features_array = np.array(request.features)
        
        # Reshape to expected format: (1, time_window, num_features)
        if len(features_array.shape) == 2:
            features = features_array.reshape(1, features_array.shape[0], features_array.shape[1])
        else:
            raise HTTPException(status_code=400, detail="Invalid feature dimensions")
        
        logger.info(f"Making prediction for features shape: {features.shape}")
        
        # Make prediction
        prediction = ensemble_model.predict(features)[0]
        
        # Calculate risk level
        if prediction >= 0.8:
            risk_level = "HIGH"
        elif prediction >= 0.5:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
        
        # Get base model predictions for confidence calculation
        base_preds = []
        try:
            for model in ensemble_model.base_models:
                if hasattr(model, 'name') and model.name == "CNN_Medical":
                    pred = model.predict_proba(features)
                else:
                    feat_2d = features.reshape(features.shape[0], -1)
                    pred = model.predict_proba(feat_2d)
                base_preds.append(pred[0])
        except Exception as e:
            logger.warning(f"Error getting base predictions: {e}")
            base_preds = [prediction] * 5  # Fallback
        
        # Calculate confidence (inverse of standard deviation)
        confidence = max(0.0, 1.0 - np.std(base_preds)) if len(base_preds) > 1 else 0.8
        
        # Create explanation
        explanation = {
            "base_predictions": {f"model_{i}": float(p) for i, p in enumerate(base_preds)},
            "ensemble_mean": float(np.mean(base_preds)),
            "prediction_std": float(np.std(base_preds)),
            "model_count": len(base_preds)
        }
        
        response = PredictionResponse(
            sepsis_probability=float(prediction),
            risk_level=risk_level,
            confidence=float(confidence),
            explanation=explanation
        )
        
        logger.info(f"Prediction completed: {prediction:.3f}, Risk: {risk_level}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@router.post("/batch_predict")
async def batch_predict_sepsis(file: UploadFile = File(...)):
    """Batch prediction from CSV file"""
    if ensemble_model is None:
        raise HTTPException(status_code=400, detail="Model not loaded")
    
    try:
        # Read uploaded file
        contents = await file.read()
        
        # Try to parse CSV
        try:
            df = pd.read_csv(io.BytesIO(contents))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid CSV file: {str(e)}")
        
        logger.info(f"Processing batch file with {len(df)} rows")
        
        predictions = []
        errors = []
        
        for idx, row in df.iterrows():
            try:
                # Convert row to features (you may need to adjust this based on your data format)
                row_values = row.values
                
                # Assume data comes as flattened features that need reshaping
                # This is a simplified example - you'd need to adapt based on your actual data format
                time_window = 6  # From config
                features_per_timestep = len(row_values) // time_window
                
                if len(row_values) % time_window != 0:
                    # Pad if necessary
                    padding_needed = time_window * features_per_timestep + features_per_timestep - len(row_values)
                    if padding_needed > 0:
                        row_values = np.append(row_values, [0] * padding_needed)
                    features_per_timestep = len(row_values) // time_window
                
                features = row_values.reshape(1, time_window, features_per_timestep)
                pred = ensemble_model.predict(features)[0]
                
                predictions.append({
                    "row_index": int(idx),
                    "patient_id": row.get('Patient_ID', f"patient_{idx}"),
                    "sepsis_probability": float(pred),
                    "risk_level": "HIGH" if pred >= 0.8 else "MEDIUM" if pred >= 0.5 else "LOW"
                })
                
            except Exception as e:
                errors.append({
                    "row_index": int(idx),
                    "error": str(e)
                })
                logger.warning(f"Error processing row {idx}: {e}")
        
        result = {
            "predictions": predictions,
            "total_processed": len(predictions),
            "total_errors": len(errors),
            "errors": errors[:10]  # Limit error details
        }
        
        logger.info(f"Batch prediction completed: {len(predictions)} successful, {len(errors)} errors")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

@router.post("/train")
async def train_model(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Start model training in background"""
    global training_status
    
    if training_status["status"] == "training":
        raise HTTPException(status_code=400, detail="Training already in progress")
    
    # Validate file
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")
    
    try:
        # Save uploaded file
        temp_dir = Path("temp")
        temp_dir.mkdir(exist_ok=True)
        file_path = temp_dir / f"train_{file.filename}"
        
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        logger.info(f"Training file saved: {file_path}")
        
        # Start background training
        background_tasks.add_task(train_pipeline_background, str(file_path))
        
        training_status.update({
            "status": "initiated", 
            "progress": 0.0, 
            "message": "Training job queued"
        })
        
        return {"message": "Training started", "status": "initiated", "file": file.filename}
        
    except Exception as e:
        logger.error(f"Error starting training: {e}")
        raise HTTPException(status_code=500, detail=f"Error starting training: {str(e)}")

@router.get("/training_status", response_model=TrainingStatus)
async def get_training_status():
    """Get current training status"""
    return TrainingStatus(**training_status)

@router.post("/load_model")
async def load_pretrained_model():
    """Load pre-trained model from disk"""
    global ensemble_model, data_loader
    
    try:
        from config.settings import config
        
        model_path = Path(config.MODELS_DIR) / "advanced_sepsis_model.pkl"
        scaler_path = Path(config.MODELS_DIR) / "advanced_scaler.pkl"
        
        if not model_path.exists():
            raise HTTPException(status_code=404, detail="No pre-trained model found")
        
        # Load model
        ensemble_model = joblib.load(model_path)
        logger.info(f"Model loaded from {model_path}")
        
        # Load scaler if available
        if scaler_path.exists():
            data_loader = joblib.load(scaler_path)
            logger.info(f"Scaler loaded from {scaler_path}")
        
        return {
            "message": "Model loaded successfully", 
            "status": "ready",
            "model_path": str(model_path),
            "scaler_available": scaler_path.exists()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")

@router.get("/model_info", response_model=ModelInfo)
async def get_model_info():
    """Get detailed model information"""
    if ensemble_model is None:
        raise HTTPException(status_code=400, detail="Model not loaded")
    
    try:
        # Get base model names
        base_models = []
        if hasattr(ensemble_model, 'base_models'):
            base_models = [getattr(model, 'name', str(type(model).__name__)) for model in ensemble_model.base_models]
        
        # Get meta model name
        meta_model = "Unknown"
        if hasattr(ensemble_model, 'meta_model'):
            meta_model = getattr(ensemble_model.meta_model, 'name', str(type(ensemble_model.meta_model).__name__))
        
        # Get feature count
        feature_count = 0
        if hasattr(ensemble_model, 'dataset_manager') and hasattr(ensemble_model.dataset_manager, 'feature_metadata'):
            metadata = ensemble_model.dataset_manager.feature_metadata
            feature_count = len(metadata.get('meta_features', []))
        
        # Get training metrics
        training_metrics = {}
        if hasattr(ensemble_model, 'base_results'):
            training_metrics = ensemble_model.base_results
        
        return ModelInfo(
            model_type="Advanced Sepsis Ensemble",
            base_models=base_models,
            meta_model=meta_model,
            feature_count=feature_count,
            training_metrics=training_metrics
        )
        
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting model info: {str(e)}")

@router.get("/visualizations")
async def list_visualizations():
    """List available visualization files"""
    from config.settings import config
    
    viz_dir = Path(config.CHARTS_DIR)
    if not viz_dir.exists():
        return {"visualizations": [], "message": "No visualizations directory found"}
    
    # Find HTML files
    html_files = list(viz_dir.glob("*.html"))
    
    visualizations = []
    for f in html_files:
        try:
            stats = f.stat()
            visualizations.append({
                "name": f.stem,
                "filename": f.name,
                "path": str(f.relative_to(Path.cwd())),
                "size_mb": round(stats.st_size / 1024 / 1024, 2),
                "created": pd.Timestamp.fromtimestamp(stats.st_mtime).isoformat()
            })
        except Exception as e:
            logger.warning(f"Error getting stats for {f}: {e}")
    
    return {
        "visualizations": visualizations,
        "total_count": len(visualizations)
    }

@router.get("/download_chart/{chart_name}")
async def download_chart(chart_name: str):
    """Download specific visualization chart"""
    from config.settings import config
    
    # Sanitize filename
    safe_name = "".join(c for c in chart_name if c.isalnum() or c in ('-', '_'))
    chart_path = Path(config.CHARTS_DIR) / f"{safe_name}.html"
    
    if not chart_path.exists():
        raise HTTPException(status_code=404, detail="Chart not found")
    
    return FileResponse(
        chart_path, 
        media_type="text/html", 
        filename=f"{safe_name}.html"
    )

@router.post("/explain_prediction")
async def explain_prediction(request: PredictionRequest):
    """Get SHAP explanation for a prediction"""
    if ensemble_model is None:
        raise HTTPException(status_code=400, detail="Model not loaded")
    
    try:
        # Convert input format
        features = np.array(request.features).reshape(1, -1, len(request.features[0]))
        
        # Get base model predictions
        base_preds = []
        model_names = []
        
        for model in ensemble_model.base_models:
            try:
                if hasattr(model, 'name'):
                    model_names.append(model.name)
                    if model.name == "CNN_Medical":
                        pred = model.predict_proba(features)
                    else:
                        feat_2d = features.reshape(features.shape[0], -1)
                        pred = model.predict_proba(feat_2d)
                    base_preds.append(pred[0])
                else:
                    model_names.append("Unknown")
                    base_preds.append(0.5)  # Fallback
            except Exception as e:
                logger.warning(f"Error getting prediction from model: {e}")
                model_names.append("Error")
                base_preds.append(0.5)
        
        # Simple explanation (without SHAP for now to avoid complexity)
        base_preds_array = np.array(base_preds)
        mean_pred = np.mean(base_preds_array)
        std_pred = np.std(base_preds_array)
        
        # Calculate contributions as deviation from mean
        contributions = base_preds_array - mean_pred
        
        explanation = {
            "prediction_breakdown": {
                name: {
                    "prediction": float(pred),
                    "contribution": float(contrib),
                    "relative_importance": float(abs(contrib) / (std_pred + 1e-6))
                }
                for name, pred, contrib in zip(model_names, base_preds, contributions)
            },
            "ensemble_stats": {
                "mean_prediction": float(mean_pred),
                "std_prediction": float(std_pred),
                "final_prediction": float(ensemble_model.predict(features)[0])
            },
            "interpretation": {
                "most_influential": model_names[np.argmax(np.abs(contributions))],
                "agreement_level": "High" if std_pred < 0.1 else "Medium" if std_pred < 0.2 else "Low"
            }
        }
        
        return {"explanation": explanation}
        
    except Exception as e:
        logger.error(f"Error in explanation: {e}")
        raise HTTPException(status_code=500, detail=f"Explanation error: {str(e)}")

@router.delete("/cleanup")
async def cleanup_temp_files():
    """Clean up temporary files"""
    try:
        temp_dir = Path("temp")
        if temp_dir.exists():
            files_removed = 0
            for file in temp_dir.glob("*"):
                try:
                    file.unlink()
                    files_removed += 1
                except Exception as e:
                    logger.warning(f"Could not remove {file}: {e}")
            
            return {"message": f"Cleanup completed", "files_removed": files_removed}
        else:
            return {"message": "No temp directory found"}
    except Exception as e:
        logger.error(f"Cleanup error: {e}")
        raise HTTPException(status_code=500, detail=f"Cleanup error: {str(e)}")

# Background training function
async def train_pipeline_background(filepath: str):
    """Background training function for API"""
    global ensemble_model, data_loader, training_status
    
    try:
        # Import here to avoid circular imports
        from ensemble.trainer import train_pipeline
        
        training_status.update({
            "status": "training",
            "progress": 0.1,
            "message": "Loading data..."
        })
        
        # Train with small sample for API demo
        ensemble, results = train_pipeline(filepath, nrows=5000)
        
        # Update global variables
        ensemble_model = ensemble
        
        training_status.update({
            "status": "completed",
            "progress": 1.0,
            "message": "Training completed successfully",
            "metrics": results
        })
        
        logger.info("Background training completed successfully")
        
        # Cleanup temp file
        try:
            Path(filepath).unlink()
        except Exception as e:
            logger.warning(f"Could not remove temp file {filepath}: {e}")
        
    except Exception as e:
        logger.error(f"Background training error: {e}")
        training_status.update({
            "status": "error",
            "progress": 0.0,
            "message": f"Training failed: {str(e)}"
        })
