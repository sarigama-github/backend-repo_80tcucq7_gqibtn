import os
from datetime import datetime
from typing import Optional, List

import requests
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from database import create_document, get_documents, db

app = FastAPI(title="CropGuard Atlas API", description="Detect crop issues, see soil insights and live weather.")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ======================
# Schemas
# ======================
class AnalysisRequest(BaseModel):
    crop: Optional[str] = Field(None, description="Crop type e.g., wheat, rice")
    lat: Optional[float] = Field(None, ge=-90, le=90)
    lon: Optional[float] = Field(None, ge=-180, le=180)
    source: str = Field(..., description="'field' or 'satellite'")


class AnalysisResult(BaseModel):
    id: str
    crop: Optional[str]
    source: str
    lat: Optional[float]
    lon: Optional[float]
    size_kb: float
    probable_diseases: List[str]
    confidence: float
    recommendations: List[str]
    created_at: datetime


class SoilProfile(BaseModel):
    lat: float
    lon: float
    soil_type: str
    ph: float
    nitrogen: str
    phosphorus: str
    potassium: str
    organic_matter_pct: float


# ======================
# Utility helpers (lightweight heuristics)
# ======================

def simple_soil_profile(lat: float, lon: float) -> SoilProfile:
    # Very simple geo-heuristic buckets
    zone = "temperate"
    if abs(lat) < 15:
        zone = "tropical"
    elif abs(lat) > 45:
        zone = "cool"

    if zone == "tropical":
        soil_type = "Loamy Sand"
        ph = 6.2
        om = 2.1
        n, p, k = ("Medium", "Low", "Medium")
    elif zone == "cool":
        soil_type = "Silty Loam"
        ph = 6.8
        om = 4.0
        n, p, k = ("Medium", "Medium", "High")
    else:
        soil_type = "Clay Loam"
        ph = 6.5
        om = 3.0
        n, p, k = ("Low", "Medium", "Medium")

    return SoilProfile(
        lat=lat,
        lon=lon,
        soil_type=soil_type,
        ph=ph,
        nitrogen=n,
        phosphorus=p,
        potassium=k,
        organic_matter_pct=om,
    )


def heuristic_disease_labels(size_kb: float, source: str) -> List[str]:
    # Mock prediction based on file size and source
    labels_field = [
        "Leaf Spot",
        "Blight",
        "Powdery Mildew",
        "Rust",
    ]
    labels_sat = [
        "Water Stress",
        "Nitrogen Deficiency",
        "Fungal Hotspots",
        "Weed Infestation",
    ]
    bucket = int(size_kb // 50) % 4
    return [labels_field, labels_sat][0 if source == "field" else 1][bucket:bucket + 2]


def recommendation_for(labels: List[str]) -> List[str]:
    recs = []
    for l in labels:
        if "Nitrogen" in l:
            recs.append("Apply nitrogen-rich fertilizer; consider split application before rainfall.")
        elif "Water" in l:
            recs.append("Increase irrigation frequency; check for clogged drip lines.")
        elif "Fungal" in l or "Blight" in l or "Mildew" in l or "Rust" in l or "Leaf Spot" in l:
            recs.append("Use a broad-spectrum fungicide and remove heavily infected leaves.")
        elif "Weed" in l:
            recs.append("Apply pre-emergent herbicide and perform mechanical weeding.")
        else:
            recs.append("Scout the field and send a sample to lab for confirmation.")
    return recs


# ======================
# Routes
# ======================
@app.get("/")
def read_root():
    return {"name": "CropGuard Atlas", "status": "ok"}


@app.get("/api/weather")
def get_weather(lat: float, lon: float):
    # Using Open-Meteo (no API key) for current weather and next hours
    try:
        url = (
            "https://api.open-meteo.com/v1/forecast?" \
            f"latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m" \
            "&hourly=temperature_2m,precipitation_probability,wind_speed_10m&forecast_days=1&timezone=auto"
        )
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
        return {"provider": "open-meteo", "data": data}
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Weather provider error: {str(e)}")


@app.get("/api/soil")
def get_soil(lat: float, lon: float):
    profile = simple_soil_profile(lat, lon)
    return profile


@app.post("/api/analyze/image", response_model=AnalysisResult)
async def analyze_image(
    file: UploadFile = File(...),
    source: str = Form("field"),  # 'field' or 'satellite'
    crop: Optional[str] = Form(None),
    lat: Optional[float] = Form(None),
    lon: Optional[float] = Form(None),
):
    if source not in ("field", "satellite"):
        raise HTTPException(status_code=400, detail="source must be 'field' or 'satellite'")

    # Read a chunk to estimate size; we don't persist the file itself
    content = await file.read()
    size_kb = round(len(content) / 1024, 2)

    labels = heuristic_disease_labels(size_kb, source)
    conf = min(0.95, 0.5 + (size_kb % 100) / 200)  # mock confidence
    recs = recommendation_for(labels)

    doc = {
        "crop": crop,
        "source": source,
        "lat": lat,
        "lon": lon,
        "size_kb": size_kb,
        "probable_diseases": labels,
        "confidence": conf,
    }
    try:
        inserted_id = create_document("analysis", doc)
    except Exception:
        # If DB not configured, still return response without storing
        inserted_id = "no-db"

    return AnalysisResult(
        id=inserted_id,
        crop=crop,
        source=source,
        lat=lat,
        lon=lon,
        size_kb=size_kb,
        probable_diseases=labels,
        confidence=conf,
        recommendations=recs,
        created_at=datetime.utcnow(),
    )


@app.get("/api/analysis")
def list_analysis(limit: int = 10):
    try:
        docs = get_documents("analysis", {}, limit)
        # convert ObjectId to string if present
        for d in docs:
            if "_id" in d:
                d["id"] = str(d.pop("_id"))
        return {"items": docs}
    except Exception:
        return {"items": []}


@app.get("/test")
def test_database():
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": None,
        "database_name": None,
        "connection_status": "Not Connected",
        "collections": []
    }

    try:
        if db is not None:
            response["database"] = "✅ Available"
            response["connection_status"] = "Connected"
            try:
                collections = db.list_collection_names()
                response["collections"] = collections[:10]
                response["database"] = "✅ Connected & Working"
            except Exception as e:
                response["database"] = f"⚠️  Connected but Error: {str(e)[:50]}"
        else:
            response["database"] = "⚠️  Available but not initialized"
    except Exception as e:
        response["database"] = f"❌ Error: {str(e)[:50]}"

    # Check environment variables
    response["database_url"] = "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set"
    response["database_name"] = "✅ Set" if os.getenv("DATABASE_NAME") else "❌ Not Set"

    return response


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
