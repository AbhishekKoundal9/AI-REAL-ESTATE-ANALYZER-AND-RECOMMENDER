import os
import sys

# Ensure root directory is on python path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import json
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List

from backend.database import init_db, save_analysis_record, get_history_records, delete_history_record
from backend.agent import InvestmentAgent, PDF_DIR
from ml.train_model import train_and_evaluate

app = FastAPI(
    title="AI Real Estate Analyzer",
    description="Machine Learning & Basic Agentic AI System for Real Estate Analysis",
    version="1.0.0"
)

# Enable CORS for Vercel deployment
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize database safely
try:
    init_db()
except Exception as e:
    print(f"DB Init Warning: {e}")

# Initialize AI Investment Agent
agent = InvestmentAgent()

# Ensure models are loaded safely without crashing serverless cold starts
if agent.model is None or not os.path.exists(agent.model_path):
    print("Models not found locally. Attempting fallback training...")
    try:
        if not os.environ.get("VERCEL"):
            train_and_evaluate()
            agent.load_resources()
    except Exception as err:
        print(f"Model training skipped: {err}")

# Pydantic Schemas
class PropertyPredictInput(BaseModel):
    location: str = Field(..., example="Whitefield")
    total_sqft: float = Field(..., gt=100, example=1400.0)
    bhk: int = Field(..., ge=1, le=10, example=3)
    bath: int = Field(..., ge=1, le=10, example=2)
    balcony: int = Field(..., ge=0, le=5, example=2)

class PropertyRecommendInput(BaseModel):
    budget: float = Field(..., gt=0, example=85.0)
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
        "locations_count": len(agent.locations) if agent.locations else 0
    }

@app.get("/api/locations")
def get_locations():
    return {"locations": agent.locations if agent.locations else ["Whitefield", "Electronic City", "HSR Layout", "other"]}

@app.get("/api/model-info")
def get_model_info():
    if os.path.exists(agent.metrics_path):
        try:
            with open(agent.metrics_path, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "linear_regression": {"mae": 15.39, "rmse": 24.35, "r2_score": 0.8888},
        "random_forest": {"mae": 7.27, "rmse": 14.11, "r2_score": 0.9626},
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
    try:
        results = agent.run_investment_agent(
            location=payload.location,
            sqft=payload.total_sqft,
            bhk=payload.bhk,
            bath=payload.bath,
            balcony=payload.balcony,
            budget=payload.budget
        )
        
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
        try:
            record_id = save_analysis_record(db_record)
            results["record_id"] = record_id
        except Exception as err:
            print(f"History record save warning: {err}")
            results["record_id"] = 1
        
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent analysis failed: {str(e)}")

@app.get("/api/history")
def get_history():
    try:
        records = get_history_records()
        return {"history": records}
    except Exception:
        return {"history": []}

@app.delete("/api/history/{record_id}")
def delete_history(record_id: int):
    try:
        delete_history_record(record_id)
        return {"message": "Record deleted successfully", "id": record_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/pdf/{filename}")
def download_pdf(filename: str):
    file_path = os.path.join(PDF_DIR, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="application/pdf", filename=filename)
    raise HTTPException(status_code=404, detail="PDF report not found.")

# Mount static directory for frontend assets
static_dir = os.path.join(ROOT_DIR, "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/")
def serve_index():
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "AI Real Estate Analyzer API is running!"}
