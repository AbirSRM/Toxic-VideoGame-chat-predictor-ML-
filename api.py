import torch
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from infer import predict, load_model
from config import Config
import os

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="ONLYPosiChat Toxicity API",
    description="Auto-scaling hardware-aware chat toxicity classification API.",
    version="1.0.0"
)

class ChatRequest(BaseModel):
    text: str
    threshold: float = 0.5

class ChatResponse(BaseModel):
    tier_used: str
    label: str
    toxic_prob: float
    predicted_class: int

# Global state to keep track of the selected tier
state = {
    "active_tier": "low"
}

@app.on_event("startup")
async def startup_event():
    """Detects hardware capabilities and sets the active model tier."""
    logger.info("Initializing ONLYPosiChat Server...")
    
    # 1. Hardware Detection
    gpu_available = torch.cuda.is_available()
    
    if gpu_available:
        logger.info(f"GPU Detected: {torch.cuda.get_device_name(0)}")
        state["active_tier"] = "high"
    else:
        logger.info("No GPU detected. Running on CPU.")
        state["active_tier"] = "low"
        
    logger.info(f"Selected model tier: {state['active_tier'].upper()}")
    
    # 2. Pre-load the model to avoid latency on the first request
    # NOTE: If models are not found locally, the API will still start, 
    # but the first inference request will fail. Ensure models are trained or mounted.
    try:
        model_path = f"./models/{state['active_tier']}_tier/final_model"
        if os.path.exists(model_path):
            logger.info(f"Pre-loading {state['active_tier']} tier model...")
            load_model(state["active_tier"])
            logger.info("Model loaded successfully.")
        else:
            logger.warning(f"Model path {model_path} not found. Ensure models are mounted or trained.")
    except Exception as e:
        logger.error(f"Error during model pre-load: {e}")

@app.post("/api/v1/analyze", response_model=ChatResponse)
async def analyze_chat(request: ChatRequest):
    """
    Analyzes a chat message for toxicity.
    Uses the dynamically selected hardware tier.
    """
    tier = state["active_tier"]
    
    if not request.text or request.text.strip() == "":
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
        
    try:
        result = predict(request.text, tier=tier, threshold=request.threshold)
        return ChatResponse(
            tier_used=tier,
            label=result["label"],
            toxic_prob=result["toxic_prob"],
            predicted_class=result["predicted_class"]
        )
    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
