from pydantic import BaseModel, Field
from typing import Optional, List

class Analysis(BaseModel):
    """
    Crop analysis results from image uploads.
    Collection: "analysis"
    """
    crop: Optional[str] = Field(None, description="Crop type")
    source: str = Field(..., description="'field' or 'satellite'")
    lat: Optional[float] = Field(None, ge=-90, le=90)
    lon: Optional[float] = Field(None, ge=-180, le=180)
    size_kb: float = Field(..., ge=0)
    probable_diseases: List[str] = Field(default_factory=list)
    confidence: float = Field(..., ge=0, le=1)
