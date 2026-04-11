"""
Phase 7 — évaluation par grille alpha × prompt × émotion.

Pour chaque prompt de EVAL_PROMPTS :
  - score base (N_RUNS runs, seeds 42+i)
  - pour chaque alpha dans ALPHAS et chaque émotion :
      score steered (N_RUNS runs, seeds 42+i)

Sortie : courbe score(α) par prompt.
Documente quantitativement non-monotonicité et seuils d'activation.

N_RUNS = 5, soit 275 générations au total (~90–120 min sur MPS).
Note : torch.manual_seed ne contrôle pas le sampler MPS — pas de seed fixé.
5 runs non-seedés donnent une distribution fiable malgré la stochasticité.
"""
from __future__ import annotations

from pathlib import Path

import torch
from transformers import pipeline

from src.model_loader import ModelWrapper
from src.steering import generate_base, generate_steered

VECTORS_DIR = Path("vectors")
LAYER_IDX   = 20
N_RUNS      = 5          # runs par condition — MPS ne respecte pas manual_seed,
                         # 5 runs donnent une distribution fiable sans seed fixé
ALPHAS      = [1.0, 1.5, 2.0, 2.5, 3.0]

EVAL_PROMPTS = [
    "Continue this story: She opened the envelope slowly and read the first line.",
    "Continue this story: He walked into the office and everyone turned to look at him.",
    "Continue this story: The phone rang at 3am and she recognized the number.",
    "Write a short scene about two people who haven't spoken in years.",
    "Continue this story: He found the old photograph at the bottom of the drawer.",
]


def score_emotion(classifier, text: str) -> dict[str, float]:
    """Retourne {label: score} pour les 7 classes."""
    results = classifier(text[:512], truncation=True)
    return {r["label"]: r["score"] for r in results[0]}


def _mean(vals: list[float]) -> float:
    return sum(vals) / len(vals)


def evaluate() -> None:
    # --- chargement classifieur (CPU, ~80 MB) ---
    print("[eval] Chargement du classifieur j-hartmann...")
    classifier = pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        top_k=None,
        device="cpu",
    )
    print("[eval] Classifieur prêt.\n")

    # --- chargement du LLM et des vecteurs ---
    wrapper      = ModelWrapper()
    joy_vector   = torch.load(VECTORS_DIR / "joy_vector.pt",   weights_only=True)
    anger_vector = torch.load(VECTORS_DIR / "anger_vector.pt", weights_only=True)
    vectors      = {"joy": joy_vector, "anger": anger_vector}

    # MPS ne respecte pas torch.manual_seed pour le sampling — pas de seed fixé.
    # N_RUNS=5 compense en échantillonnant la distribution stochastique.
    total = len(EVAL_PROMPTS) * (N_RUNS + len(vectors) * len(ALPHAS) * N_RUNS)
    done  = 0

    for p_idx, prompt in enumerate(EVAL_PROMPTS):
        short = prompt[:65]
        print(f"\n{'='*72}")
        print(f"PROMPT {p_idx+1}/{len(EVAL_PROMPTS)} : {short}...")
        print(f"{'='*72}")

        # --- base ---
        base_by_emotion: dict[str, list[float]] = {}
        for run in range(N_RUNS):
            text = generate_base(wrapper, prompt, max_new_tokens=120)
            for emotion, score in score_emotion(classifier, text).items():
                base_by_emotion.setdefault(emotion, []).append(score)
            done += 1
            print(f"  [base run {run+1}/{N_RUNS}]  ({done}/{total} générations)")

        base_ref: dict[str, float] = {
            em: _mean(scores) for em, scores in base_by_emotion.items()
        }
        print(f"\n  BASE  joy: {base_ref.get('joy', 0):.3f}   "
              f"anger: {base_ref.get('anger', 0):.3f}   (mean {N_RUNS} runs)")

        # --- grille alpha par émotion ---
        for emotion_name, vector in vectors.items():
            target = emotion_name

            # collecte
            alpha_data: list[dict] = []
            for alpha in ALPHAS:
                run_scores: list[float] = []
                run_dominant: list[str] = []

                for run in range(N_RUNS):
                    text = generate_steered(
                        wrapper, prompt, vector, alpha, LAYER_IDX,
                        max_new_tokens=120,
                    )
                    s = score_emotion(classifier, text)
                    run_scores.append(s.get(target, 0.0))
                    run_dominant.append(max(s, key=s.__getitem__))
                    done += 1
                    print(f"  [{emotion_name} α={alpha} run {run+1}/{N_RUNS}]  "
                          f"({done}/{total} générations)")

                dominant = max(set(run_dominant), key=run_dominant.count)
                alpha_data.append({
                    "alpha":     alpha,
                    "mean":      _mean(run_scores),
                    "min":       min(run_scores),
                    "max":       max(run_scores),
                    "dominant":  dominant,
                })

            # pic
            peak = max(alpha_data, key=lambda r: r["mean"])

            # affichage
            print(f"\n  {emotion_name.upper()} VECTOR — score '{target}' par alpha")
            for i, r in enumerate(alpha_data):
                drift = (
                    f"  (→ {r['dominant']})"
                    if r["dominant"] != target else ""
                )
                trend = ""
                if i > 0:
                    diff = r["mean"] - alpha_data[i - 1]["mean"]
                    if diff < -0.05:
                        trend = " ↓"
                    elif diff > 0.05:
                        trend = " ↑"
                peak_mark = "  ← pic" if r["alpha"] == peak["alpha"] else ""
                print(
                    f"  α={r['alpha']:.1f}  {r['mean']:.3f}  "
                    f"[{r['min']:.3f}–{r['max']:.3f}]"
                    f"{drift}{trend}{peak_mark}"
                )

            delta_peak = peak["mean"] - base_ref.get(target, 0.0)
            print(f"  → Δ au pic : {delta_peak:+.3f} vs base  "
                  f"(α={peak['alpha']:.1f}, score={peak['mean']:.3f})")

    print(f"\n{'='*72}")
    print("Évaluation terminée.")
    print(f"{'='*72}\n")


if __name__ == "__main__":
    evaluate()
