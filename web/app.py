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

from src.eval_latent import latent_score as _latent_score
from src.eval_latent import llm_judge_score as _llm_judge_score
from src.model_loader import ModelWrapper
from src.steering import generate_base as _generate_base
from src.steering import generate_steered as _generate_steered

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

VECTORS_DIR  = Path("vectors")
LAYER_IDX    = 20
MAX_RETRIES  = 3   # tentatives max avant de rendre un refus tel quel

# Préfixes caractéristiques d'un refus RLHF
_REFUSAL_PREFIXES = (
    "i apologize",
    "i'm not able",
    "i am not able",
    "i cannot",
    "i can't",
    "as an ai",
    "i'm an ai",
    "i am an ai",
    "i'm unable",
    "i am unable",
    "i'm sorry, but",
    "i'm sorry, i",
)


def _is_refusal(text: str) -> bool:
    t = text.lower().strip()
    return any(t.startswith(p) for p in _REFUSAL_PREFIXES)

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


class AnalyzeRequest(BaseModel):
    text:    str
    emotion: str


class AnalyzeResponse(BaseModel):
    llm_judge: float | None


class GenerateResponse(BaseModel):
    text:     str
    scores:   dict[str, float]
    latent:   float | None = None   # cosine alignment avec le vecteur émotion à la couche 20
    attempts: int = 1               # nombre de tentatives avant un output non-refus


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

    # Auto-retry : si le modèle génère un refus RLHF, on réessaie silencieusement.
    # temperature=0.7 rend chaque run stochastique — un refus n'implique pas
    # que les suivants le seront aussi.
    text     = ""
    attempts = 0
    for attempt in range(1, MAX_RETRIES + 1):
        attempts = attempt
        text = await _run(
            _generate_steered,
            _wrapper, req.prompt, vector, req.alpha, LAYER_IDX, req.max_new_tokens,
        )
        if not _is_refusal(text):
            break

    latent = await _run(_latent_score, _wrapper, text, vector, LAYER_IDX)
    scores = await asyncio.to_thread(_score, text)
    return GenerateResponse(text=text, scores=scores, latent=latent, attempts=attempts)


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest):
    """LLM judge : le modèle évalue son propre output, temperature=0.1."""
    if req.emotion not in _vectors:
        raise HTTPException(
            status_code=400,
            detail=f"Émotion inconnue : {req.emotion!r}.",
        )
    judge = await _run(_llm_judge_score, _wrapper, req.text, req.emotion)
    return AnalyzeResponse(llm_judge=judge)
