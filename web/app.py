"""
Phase 10 — backend FastAPI.

Démarre avec : uvicorn web.app:app --reload   (depuis la racine du projet)

Endpoints :
  GET  /health
  GET  /emotions
  POST /generate_base
  POST /generate_steered

Design :
  - Modèle, vecteurs et classifieur chargés une seule fois au démarrage (lifespan).
  - asyncio.Semaphore(1) : une seule inférence GPU/MPS à la fois.
  - asyncio.to_thread() : inference bloquante sortie du thread async.
  - CORS ouvert pour Phase 11 (front HTML/JS en localhost).
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from transformers import pipeline

from src.model_loader import ModelWrapper
from src.steering import generate_base as _generate_base
from src.steering import generate_steered as _generate_steered

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

VECTORS_DIR = Path("vectors")
LAYER_IDX   = 20

EMOTIONS: dict[str, dict] = {
    "joy": {
        "label":         "Joy",
        "alpha_default": 2.0,
        "description":   "Warm, joyful, enthusiastic tone",
    },
    "anger": {
        "label":         "Anger",
        "alpha_default": 2.0,
        "description":   "Angry, frustrated, hostile tone",
    },
}

# ----------------------------------------------------------------------
# État global (initialisé dans lifespan)
# ----------------------------------------------------------------------

_wrapper:   ModelWrapper | None       = None
_vectors:   dict[str, torch.Tensor]   = {}
_classifier                           = None
_semaphore: asyncio.Semaphore | None  = None


# ----------------------------------------------------------------------
# Lifespan — chargement unique au démarrage
# ----------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _wrapper, _vectors, _classifier, _semaphore

    print("[startup] Chargement du modèle...")
    _wrapper = ModelWrapper()

    print("[startup] Chargement des vecteurs...")
    for emotion in EMOTIONS:
        path = VECTORS_DIR / f"{emotion}_vector.pt"
        if not path.exists():
            raise FileNotFoundError(
                f"Vecteur manquant : {path}. "
                "Lancez d'abord : python -m src.extract_vectors"
            )
        _vectors[emotion] = torch.load(path, weights_only=True)

    print("[startup] Chargement du classifieur...")
    _classifier = pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        top_k=None,
        device="cpu",
    )

    _semaphore = asyncio.Semaphore(1)
    print("[startup] Prêt.")

    yield

    print("[shutdown] Nettoyage.")


# ----------------------------------------------------------------------
# Application
# ----------------------------------------------------------------------

app = FastAPI(title="Emotion Steering Demo", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----------------------------------------------------------------------
# Helpers internes
# ----------------------------------------------------------------------

def _score(text: str) -> dict[str, float]:
    """Scores j-hartmann sur les 6 classes. Tourne sur CPU."""
    results = _classifier(text[:512], truncation=True)
    return {r["label"]: round(r["score"], 4) for r in results[0]}


async def _run(fn, *args, **kwargs):
    """
    Exécute une fonction bloquante (PyTorch) dans un thread séparé,
    protégée par le sémaphore MPS.
    """
    async with _semaphore:
        return await asyncio.to_thread(fn, *args, **kwargs)


# ----------------------------------------------------------------------
# Schémas Pydantic
# ----------------------------------------------------------------------

class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = Field(default=120, ge=20, le=400)


class SteerRequest(BaseModel):
    prompt: str
    emotion: str
    alpha: float = Field(default=2.0, ge=0.1, le=10.0)
    max_new_tokens: int = Field(default=120, ge=20, le=400)


class GenerateResponse(BaseModel):
    text:   str
    scores: dict[str, float]


# ----------------------------------------------------------------------
# Endpoints
# ----------------------------------------------------------------------

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "device": str(_wrapper.device) if _wrapper else "not loaded",
        "emotions": list(_vectors.keys()),
    }


@app.get("/emotions")
async def emotions():
    return [
        {"name": name, **meta}
        for name, meta in EMOTIONS.items()
    ]


@app.get("/")
async def root():
    return FileResponse(Path(__file__).parent / "index.html")


@app.post("/generate_base", response_model=GenerateResponse)
async def generate_base(req: GenerateRequest):
    text   = await _run(_generate_base, _wrapper, req.prompt, req.max_new_tokens)
    scores = await asyncio.to_thread(_score, text)  # CPU-bound — hors sémaphore MPS intentionnellement
    return GenerateResponse(text=text, scores=scores)


@app.post("/generate_steered", response_model=GenerateResponse)
async def generate_steered(req: SteerRequest):
    if req.emotion not in _vectors:
        raise HTTPException(
            status_code=400,
            detail=f"Émotion inconnue : {req.emotion!r}. "
                   f"Disponibles : {list(_vectors.keys())}",
        )
    vector = _vectors[req.emotion]
    text   = await _run(
        _generate_steered,
        _wrapper, req.prompt, vector, req.alpha, LAYER_IDX, req.max_new_tokens,
    )
    scores = await asyncio.to_thread(_score, text)  # CPU-bound — hors sémaphore MPS intentionnellement
    return GenerateResponse(text=text, scores=scores)
