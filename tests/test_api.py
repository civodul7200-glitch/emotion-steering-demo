"""
Tests d'intégration — endpoints FastAPI.

Le modèle, les vecteurs et le classifieur sont tous mockés :
aucun téléchargement ni calcul GPU n'est nécessaire.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from unittest.mock import MagicMock, patch

import pytest
import torch
from fastapi.testclient import TestClient

import web.app as app_module
from web.app import app, EMOTIONS


# ---------------------------------------------------------------------------
# Fixture — client avec dépendances mockées
# ---------------------------------------------------------------------------

FAKE_HIDDEN_DIM = 1536


def make_classifier_mock():
    """Simule pipeline('text-classification', ...) — retourne des scores fixes."""
    def _classify(text, truncation=False):
        return [[
            {"label": "joy",      "score": 0.70},
            {"label": "anger",    "score": 0.05},
            {"label": "fear",     "score": 0.05},
            {"label": "sadness",  "score": 0.05},
            {"label": "disgust",  "score": 0.05},
            {"label": "neutral",  "score": 0.05},
            {"label": "surprise", "score": 0.05},
        ]]
    mock = MagicMock(side_effect=_classify)
    return mock


FAKE_GENERATED_TEXT = "This is a generated story continuation."


@pytest.fixture()
def client():
    """
    TestClient avec lifespan et fonctions de génération mockées.
    On patche generate_base et generate_steered pour retourner du texte
    directement — les tests steering sont couverts dans test_steering.py.
    """
    wrapper_mock = MagicMock()
    wrapper_mock.device = torch.device("cpu")

    vectors = {
        emotion: torch.zeros(FAKE_HIDDEN_DIM)
        for emotion in EMOTIONS
    }

    classifier_mock = make_classifier_mock()

    @asynccontextmanager
    async def mock_lifespan(a):
        import asyncio
        app_module._wrapper    = wrapper_mock
        app_module._vectors    = vectors
        app_module._classifier = classifier_mock
        app_module._semaphore  = asyncio.Semaphore(1)
        yield
        app_module._wrapper    = None
        app_module._vectors    = {}
        app_module._classifier = None
        app_module._semaphore  = None

    app.router.lifespan_context = mock_lifespan

    with patch("web.app._generate_base",    return_value=FAKE_GENERATED_TEXT), \
         patch("web.app._generate_steered", return_value=FAKE_GENERATED_TEXT), \
         patch("web.app._latent_score",     return_value=0.42), \
         patch("web.app._llm_judge_score",  return_value=0.75):
        with TestClient(app) as c:
            yield c


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

class TestHealth:
    def test_returns_ok(self, client):
        r = client.get("/health")
        assert r.status_code == 200

    def test_device_field_present(self, client):
        data = client.get("/health").json()
        assert "device" in data
        assert data["status"] == "ok"

    def test_emotions_listed(self, client):
        data = client.get("/health").json()
        assert set(data["emotions"]) == set(EMOTIONS.keys())


# ---------------------------------------------------------------------------
# /emotions
# ---------------------------------------------------------------------------

class TestEmotions:
    def test_returns_list(self, client):
        r = client.get("/emotions")
        assert r.status_code == 200
        assert isinstance(r.json(), list)

    def test_all_emotions_present(self, client):
        names = {e["name"] for e in client.get("/emotions").json()}
        assert names == set(EMOTIONS.keys())

    def test_emotion_has_required_fields(self, client):
        for emotion in client.get("/emotions").json():
            assert "name" in emotion
            assert "label" in emotion
            assert "alpha_default" in emotion


# ---------------------------------------------------------------------------
# /generate_base
# ---------------------------------------------------------------------------

class TestGenerateBase:
    def test_success(self, client):
        r = client.post("/generate_base", json={"prompt": "Tell me a story."})
        assert r.status_code == 200
        data = r.json()
        assert "text" in data
        assert "scores" in data

    def test_text_is_string(self, client):
        data = client.post("/generate_base", json={"prompt": "Hello"}).json()
        assert isinstance(data["text"], str)

    def test_scores_have_seven_classes(self, client):
        data = client.post("/generate_base", json={"prompt": "Hello"}).json()
        assert len(data["scores"]) == 7

    def test_score_values_are_floats(self, client):
        data = client.post("/generate_base", json={"prompt": "Hello"}).json()
        for v in data["scores"].values():
            assert isinstance(v, float)

    def test_max_new_tokens_respected(self, client):
        r = client.post("/generate_base", json={"prompt": "Hello", "max_new_tokens": 50})
        assert r.status_code == 200

    def test_max_new_tokens_too_low(self, client):
        r = client.post("/generate_base", json={"prompt": "Hello", "max_new_tokens": 5})
        assert r.status_code == 422

    def test_max_new_tokens_too_high(self, client):
        r = client.post("/generate_base", json={"prompt": "Hello", "max_new_tokens": 999})
        assert r.status_code == 422

    def test_empty_prompt_accepted(self, client):
        r = client.post("/generate_base", json={"prompt": ""})
        assert r.status_code == 200


# ---------------------------------------------------------------------------
# /generate_steered
# ---------------------------------------------------------------------------

class TestGenerateSteered:
    def test_success_joy(self, client):
        r = client.post("/generate_steered", json={
            "prompt": "Continue this story.", "emotion": "joy", "alpha": 2.0
        })
        assert r.status_code == 200
        data = r.json()
        assert "text" in data
        assert "scores" in data

    def test_success_anger(self, client):
        r = client.post("/generate_steered", json={
            "prompt": "Continue this story.", "emotion": "anger", "alpha": 2.0
        })
        assert r.status_code == 200

    def test_unknown_emotion_returns_400(self, client):
        r = client.post("/generate_steered", json={
            "prompt": "Hello", "emotion": "surprise", "alpha": 2.0
        })
        assert r.status_code == 400
        assert "surprise" in r.json()["detail"]

    def test_alpha_too_low_returns_422(self, client):
        r = client.post("/generate_steered", json={
            "prompt": "Hello", "emotion": "joy", "alpha": 0.0
        })
        assert r.status_code == 422

    def test_alpha_too_high_returns_422(self, client):
        r = client.post("/generate_steered", json={
            "prompt": "Hello", "emotion": "joy", "alpha": 99.0
        })
        assert r.status_code == 422

    def test_scores_sum_near_one(self, client):
        data = client.post("/generate_steered", json={
            "prompt": "Hello", "emotion": "joy", "alpha": 2.0
        }).json()
        total = sum(data["scores"].values())
        assert abs(total - 1.0) < 0.01

    def test_response_schema(self, client):
        data = client.post("/generate_steered", json={
            "prompt": "Hello", "emotion": "joy", "alpha": 2.0
        }).json()
        assert {"text", "scores", "latent"} <= set(data.keys())

    def test_latent_field_is_float(self, client):
        data = client.post("/generate_steered", json={
            "prompt": "Hello", "emotion": "joy", "alpha": 2.0
        }).json()
        assert isinstance(data["latent"], float)

    def test_final_refusal_flag_and_empty_scores(self, client):
        """Quand tous les retries produisent un refus, final_refusal=True
        et scores doit être vide — un texte de refus ne doit pas alimenter
        les métriques émotionnelles."""
        refusal_text = "I'm sorry, but I cannot continue that story."
        with patch("web.app._generate_steered", return_value=refusal_text):
            data = client.post("/generate_steered", json={
                "prompt": "Hello", "emotion": "joy", "alpha": 2.0
            }).json()
        assert data["final_refusal"] is True
        assert data["scores"] == {}
        assert data["latent"] is None


# ---------------------------------------------------------------------------
# /analyze
# ---------------------------------------------------------------------------

class TestAnalyze:
    def test_success(self, client):
        r = client.post("/analyze", json={"text": FAKE_GENERATED_TEXT, "emotion": "joy"})
        assert r.status_code == 200
        assert "llm_judge" in r.json()

    def test_llm_judge_is_float(self, client):
        data = client.post("/analyze", json={"text": FAKE_GENERATED_TEXT, "emotion": "joy"}).json()
        assert isinstance(data["llm_judge"], float)

    def test_unknown_emotion_returns_400(self, client):
        r = client.post("/analyze", json={"text": "some text", "emotion": "surprise"})
        assert r.status_code == 400
