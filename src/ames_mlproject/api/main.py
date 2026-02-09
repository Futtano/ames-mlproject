"""
FastAPI Web API for the Ames Housing price prediction project.
Provides industrial-grade endpoints for inference and system health.
"""

from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

from ames_mlproject.api.schemas import PredictionRequest, PredictionResponse
from ames_mlproject.config import get_config
from ames_mlproject.core.logging import logger
from ames_mlproject.pipelines.predict import PredictionPipeline

# Load configuration
config = get_config()

# Initialize FastAPI app
app = FastAPI(
    title="Ames Housing Price Prediction API",
    description="Productized ML API for predicting housing prices in Ames, Iowa.",
    version="1.0.0",
)

# Setup templates (nested in api/templates)
BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Serve the web frontend."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring."""
    return {"status": "healthy"}


@app.post("/api/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Predict the sale price of a house based on its features.

    The request is automatically validated against the Pydantic schema.
    """
    try:
        # Convert Pydantic model to dictionary (using aliases for internal compatibility)
        input_data = request.model_dump(by_alias=True)

        # Initialize prediction pipeline
        pipeline = PredictionPipeline()

        # Make prediction
        prediction = pipeline.predict(input_data)

        return PredictionResponse(SalePrice=float(prediction[0]))

    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) from e


if __name__ == "__main__":
    uvicorn.run(
        "ames_mlproject.api.main:app",
        host=config.api.host,
        port=config.api.port,
        reload=config.api.debug,
    )
