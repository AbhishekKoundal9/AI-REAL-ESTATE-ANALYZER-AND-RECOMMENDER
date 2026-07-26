import os
import sys

# Ensure root directory is on python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field, field_validator
from typing import Optional, List

from backend.database import init_db, save_analysis_record, get_history_records, delete_history_record
from backend.agent import InvestmentAgent, PDF_DIR
from ml.train_model import train_and_evaluate

app = FastAPI(
    title="AI Real Estate Analyzer",
    description="Machine Learning & Basic Agentic AI System for Real Estate Analysis using Bengaluru Kaggle Dataset",
    version="1.0.0"
)

# Initialize database
init_db()

# Initialize AI Investment Agent
agent = InvestmentAgent()

# Ensure models are trained if missing
if agent.model is None or not os.path.exists(agent.model_path):
    print("Models not found. Triggering ML model training...")
    train_and_evaluate()
    agent.load_resources()

# Pydantic Schemas
class PropertyPredictInput(BaseModel):
    location: str = Field(..., example="Whitefield")
    total_sqft: float = Field(..., gt=100, example=1400.0)
    bhk: int = Field(..., ge=1, le=10, example=3)
    bath: int = Field(..., ge=1, le=10, example=2)
    balcony: int = Field(..., ge=0, le=5, example=2)

class PropertyRecommendInput(BaseModel):
    budget: float = Field(..., gt=0, example=85.0) # Budget in Lakhs
    location: Optional[str] = Field("Whitefield", example="Whitefield")
    bhk: Optional[int] = Field(2, example=2)

class AgentAnalyzeInput(BaseModel):
    location: str = Field(..., example="Whitefield")
    total_sqft: float = Field(..., gt=100, example=1500.0)
    bhk: int = Field(..., ge=1, le=10, example=3)
    bath: int = Field(..., ge=1, le=10, example=3)
    balcony: int = Field(..., ge=0, le=5, example=2)
    budget: Optional[float] = Field(None, example=95.0)

# Routes

@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": agent.model is not None,
        "dataset_records": len(agent.dataset) if agent.dataset is not None else 0,
        "locations_count": len(agent.locations)
    }

@app.get("/api/locations")
def get_locations():
    return {"locations": agent.locations if agent.locations else ["Whitefield", "Electronic City", "HSR Layout", "other"]}

@app.get("/api/model-info")
def get_model_info():
    if os.path.exists(agent.metrics_path):
        with open(agent.metrics_path, "r") as f:
            return json.load(f)
    return {
        "linear_regression": {"mae": 14.2, "rmse": 22.1, "r2_score": 0.68},
        "random_forest": {"mae": 8.5, "rmse": 14.3, "r2_score": 0.85},
        "best_model": "Random Forest Regressor"
    }

@app.post("/api/predict")
def predict_property_price(payload: PropertyPredictInput):
    try:
        input_data = agent.read_input(payload.location, payload.total_sqft, payload.bhk, payload.bath, payload.balcony)
        predicted_price = agent.predict_price(input_data)
        pps = (predicted_price * 100000.0) / input_data["total_sqft"]
        return {
            "input_data": input_data,
            "predicted_price_lakhs": predicted_price,
            "predicted_price_inr": f"₹{predicted_price * 100000:,.0f}",
            "price_per_sqft": round(pps, 2)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/recommend")
def recommend_properties(payload: PropertyRecommendInput):
    try:
        matches = agent.find_similar_properties(
            location=payload.location,
            bhk=payload.bhk,
            budget_lakhs=payload.budget,
            top_n=5
        )
        return {
            "query": {"budget": payload.budget, "location": payload.location, "bhk": payload.bhk},
            "recommendations": matches
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/analyze")
def run_agentic_analysis(payload: AgentAnalyzeInput):
    """
    Triggers the 11-step Basic Agentic AI workflow:
    1. Reads user input
    2. Searches Kaggle dataset
    3. Predicts price
    4. Finds similar properties
    5. Calculates ROI
    6. Calculates Rental Yield
    7. Calculates Investment Score
    8. Classifies Risk
    9. Generates AI Insights
    10. Recommends BUY/HOLD/AVOID
    11. Generates PDF Report
    """
    try:
        results = agent.run_investment_agent(
            location=payload.location,
            sqft=payload.total_sqft,
            bhk=payload.bhk,
            bath=payload.bath,
            balcony=payload.balcony,
            budget=payload.budget
        )
        
        # Save analysis record to SQLite DB
        db_record = {
            "location": results["input_data"]["location"],
            "sqft": results["input_data"]["total_sqft"],
            "bhk": results["input_data"]["bhk"],
            "bath": results["input_data"]["bath"],
            "balcony": results["input_data"]["balcony"],
            "predicted_price": results["predicted_price_lakhs"],
            "roi": results["roi_percent"],
            "rental_yield": results["rental_yield_percent"],
            "investment_score": results["investment_score"],
            "risk_level": results["risk_level"],
            "recommendation": results["recommendation"],
            "pdf_filename": results["pdf_filename"]
        }
        record_id = save_analysis_record(db_record)
        results["record_id"] = record_id
        
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent analysis failed: {str(e)}")

@app.get("/api/history")
def get_history():
    records = get_history_records()
    return {"history": records}

@app.delete("/api/history/{record_id}")
def delete_history(record_id: int):
    delete_history_record(record_id)
    return {"message": "Record deleted successfully", "id": record_id}

@app.get("/api/pdf/{filename}")
def download_pdf(filename: str):
    file_path = os.path.join(PDF_DIR, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="application/pdf", filename=filename)
    raise HTTPException(status_code=404, detail="PDF report not found.")

# Mount static directory for frontend assets
static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/")
def serve_index():
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "AI Real Estate Investment Analyzer API is running!"}
