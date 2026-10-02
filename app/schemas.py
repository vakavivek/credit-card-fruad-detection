from typing import Optional
from pydantic import BaseModel

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool

class PredictionResponse(BaseModel):
    message: str
    output_file: str
    total_transactions: int
    fraud_transactions: int