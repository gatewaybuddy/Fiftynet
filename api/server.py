"""
FastAPI server for Fiftynet model inference.

Provides REST endpoints for text generation and model information.
"""

from pathlib import Path
from typing import Optional
import time

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import torch

from model import FFTNet
from fftnet.utils.storage import load_model
from fftnet.utils.sampling import sample_top_k_top_p
from tokenizer import SimpleTokenizer


# Pydantic models for request/response validation
class GenerateRequest(BaseModel):
    """Request model for text generation."""
    prompt: str = Field(..., description="Input text prompt", min_length=1, max_length=10000)
    max_new_tokens: int = Field(50, description="Number of tokens to generate", ge=1, le=500)
    temperature: float = Field(1.0, description="Sampling temperature", ge=0.0, le=2.0)
    top_k: int = Field(0, description="Top-k sampling (0 to disable)", ge=0, le=1000)
    top_p: float = Field(1.0, description="Nucleus sampling threshold", ge=0.0, le=1.0)


class GenerateResponse(BaseModel):
    """Response model for text generation."""
    generated_text: str
    prompt: str
    tokens_generated: int
    inference_time: float


class ModelInfo(BaseModel):
    """Model information response."""
    model_name: str
    vocab_size: int
    dim: int
    num_blocks: int
    device: str
    ready: bool


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    uptime: float


# Global state
class ModelState:
    """Global model state."""
    model: Optional[FFTNet] = None
    tokenizer: Optional[SimpleTokenizer] = None
    config: Optional[dict] = None
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    start_time: float = time.time()
    ready: bool = False


state = ModelState()


# Create FastAPI app
app = FastAPI(
    title="Fiftynet API",
    description="REST API for Fiftynet frequency-domain transformer model",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def load_model_state(model_path: str, tokenizer_path: str) -> None:
    """Load model and tokenizer into global state."""
    try:
        model_file = Path(model_path)
        if not model_file.exists():
            model_file = Path("weights") / model_path

        if not model_file.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        state.model, state.config = load_model(model_file)
        state.model.to(state.device)
        state.model.eval()

        state.tokenizer = SimpleTokenizer.load(tokenizer_path)
        state.ready = True

        print(f"Model loaded successfully from {model_file}")
        print(f"Device: {state.device}")
        print(f"Config: {state.config}")

    except Exception as e:
        print(f"Error loading model: {e}")
        state.ready = False
        raise


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    import os

    model_path = os.getenv("MODEL_PATH", "trained")
    tokenizer_path = os.getenv("TOKENIZER_PATH", "tokenizer.json")

    try:
        load_model_state(model_path, tokenizer_path)
    except Exception as e:
        print(f"Warning: Could not load model on startup: {e}")
        print("Model must be loaded via POST /load-model endpoint")


@app.get("/", response_model=dict)
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Fiftynet API",
        "version": "1.0.0",
        "status": "ready" if state.ready else "not_ready",
        "endpoints": {
            "generate": "POST /generate - Generate text from prompt",
            "model_info": "GET /model-info - Get model information",
            "health": "GET /health - Health check",
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if state.ready else "not_ready",
        model_loaded=state.ready,
        uptime=time.time() - state.start_time,
    )


@app.get("/model-info", response_model=ModelInfo)
async def model_info():
    """Get information about the loaded model."""
    if not state.ready or state.model is None or state.config is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return ModelInfo(
        model_name="FFTNet",
        vocab_size=state.config.get("vocab_size", 0),
        dim=state.config.get("dim", 0),
        num_blocks=state.config.get("num_blocks", 0),
        device=str(state.device),
        ready=state.ready,
    )


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """
    Generate text from a prompt.

    Args:
        request: Generation parameters

    Returns:
        Generated text with metadata
    """
    if not state.ready or state.model is None or state.tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        start_time = time.time()

        # Tokenize input
        input_ids = state.tokenizer.encode(request.prompt)
        if len(input_ids) == 0:
            raise HTTPException(status_code=400, detail="Prompt resulted in empty token sequence")

        input_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(state.device)

        # Generate tokens
        with torch.no_grad():
            generated = input_tensor
            for _ in range(request.max_new_tokens):
                logits = state.model(generated)
                next_token_logits = logits[:, -1, :]

                # Sample next token
                if request.temperature == 0 or (request.top_k == 0 and request.top_p == 1.0):
                    next_token = next_token_logits.argmax(dim=-1, keepdim=True)
                else:
                    next_token = sample_top_k_top_p(
                        next_token_logits,
                        k=request.top_k,
                        p=request.top_p,
                        temperature=request.temperature
                    ).unsqueeze(-1)

                generated = torch.cat([generated, next_token], dim=1)

        # Decode output
        output_tokens = generated[0].tolist()
        generated_text = state.tokenizer.decode(output_tokens)

        inference_time = time.time() - start_time

        return GenerateResponse(
            generated_text=generated_text,
            prompt=request.prompt,
            tokens_generated=len(output_tokens) - len(input_ids),
            inference_time=inference_time,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


@app.post("/load-model")
async def load_model_endpoint(
    model_path: str = "trained",
    tokenizer_path: str = "tokenizer.json",
    background_tasks: BackgroundTasks = None
):
    """
    Load or reload a model.

    Args:
        model_path: Path to model weights
        tokenizer_path: Path to tokenizer

    Returns:
        Success message
    """
    try:
        load_model_state(model_path, tokenizer_path)
        return {"status": "success", "message": f"Model loaded from {model_path}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    # Run server with uvicorn
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # Set to True for development
        log_level="info",
    )
