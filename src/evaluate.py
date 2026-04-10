"""
Phase 7 — évaluation avec classifieur externe.

Classifieur : j-hartmann/emotion-english-distilroberta-base
  6 classes : anger, disgust, fear, joy, neutral, sadness, surprise

Pour chaque prompt fixe :
  - générer version base
  - générer version steered (joy, puis anger)
  - scorer les trois
  - calculer delta = score_steered(cible) - score_base(cible)

Sortie : tableau prompt / score_base / score_steered / delta
"""
from __future__ import annotations

from pathlib import Path

import torch
from transformers import pipeline

from src.load_model import ModelWrapper
from src.steering import generate_base, generate_steered

VECTORS_DIR = Path("vectors")
LAYER_IDX   = 20

EVAL_PROMPTS = [
    "Continue this story: She opened the envelope slowly and read the first line.",
    "Continue this story: He walked into the office and everyone turned to look at him.",
    "Continue this story: The phone rang at 3am and she recognized the number.",
    "Write a short scene about two people who haven't spoken in years.",
    "Continue this story: He found the old photograph at the bottom of the drawer.",
]


def score_emotion(classifier, text: str) -> dict[str, float]:
    """Retourne {label: score} pour les 6 classes."""
    results = classifier(text[:512], truncation=True)  # limite à 512 tokens
    return {r["label"]: r["score"] for r in results[0]}


def evaluate() -> None:
    # --- chargement classifieur (CPU, ~80 MB) ---
    print("[eval] Chargement du classifieur j-hartmann...")
    classifier = pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        top_k=None,
        device="cpu",     # petit modèle, CPU suffisant
    )
    print("[eval] Classifieur prêt.\n")

    # --- chargement du LLM et des vecteurs ---
    wrapper     = ModelWrapper()
    joy_vector  = torch.load(VECTORS_DIR / "joy_vector.pt",   weights_only=True)
    anger_vector= torch.load(VECTORS_DIR / "anger_vector.pt", weights_only=True)

    alpha = 2.0

    rows_joy   = []
    rows_anger = []

    for prompt in EVAL_PROMPTS:
        print(f"→ {prompt[:60]}...")

        base        = generate_base(wrapper, prompt, max_new_tokens=120)
        steered_joy = generate_steered(wrapper, prompt, joy_vector,   alpha, LAYER_IDX, max_new_tokens=120)
        steered_ang = generate_steered(wrapper, prompt, anger_vector, alpha, LAYER_IDX, max_new_tokens=120)

        s_base = score_emotion(classifier, base)
        s_joy  = score_emotion(classifier, steered_joy)
        s_ang  = score_emotion(classifier, steered_ang)

        rows_joy.append({
            "prompt":         prompt[:50],
            "base_joy":       round(s_base["joy"],   3),
            "steered_joy":    round(s_joy["joy"],    3),
            "delta_joy":      round(s_joy["joy"] - s_base["joy"], 3),
        })
        rows_anger.append({
            "prompt":         prompt[:50],
            "base_anger":     round(s_base["anger"],  3),
            "steered_anger":  round(s_ang["anger"],   3),
            "delta_anger":    round(s_ang["anger"] - s_base["anger"], 3),
        })

    # --- affichage ---
    print("\n" + "=" * 65)
    print("VECTEUR JOY — delta sur classe 'joy'")
    print("=" * 65)
    print(f"{'Prompt':50} {'base':>6} {'steer':>6} {'Δ':>7}")
    print("-" * 65)
    for r in rows_joy:
        flag = "✓" if r["delta_joy"] > 0 else "✗"
        print(f"{r['prompt']:50} {r['base_joy']:6.3f} {r['steered_joy']:6.3f} {r['delta_joy']:+7.3f} {flag}")
    mean_delta = sum(r["delta_joy"] for r in rows_joy) / len(rows_joy)
    print(f"\n  delta moyen joy   : {mean_delta:+.3f}")

    print("\n" + "=" * 65)
    print("VECTEUR ANGER — delta sur classe 'anger'")
    print("=" * 65)
    print(f"{'Prompt':50} {'base':>6} {'steer':>6} {'Δ':>7}")
    print("-" * 65)
    for r in rows_anger:
        flag = "✓" if r["delta_anger"] > 0 else "✗"
        print(f"{r['prompt']:50} {r['base_anger']:6.3f} {r['steered_anger']:6.3f} {r['delta_anger']:+7.3f} {flag}")
    mean_delta = sum(r["delta_anger"] for r in rows_anger) / len(rows_anger)
    print(f"\n  delta moyen anger : {mean_delta:+.3f}")


if __name__ == "__main__":
    evaluate()
