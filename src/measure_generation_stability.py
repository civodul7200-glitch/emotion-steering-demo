"""
Mesure de la stochasticité de la génération sur les cas documentés du golden set.

Pour chaque combinaison (prompt, émotion, alpha), lance N runs indépendants
sans auto-retry — pour mesurer le taux de refus brut et la variance des scores.

Métriques collectées par run :
  - refus détecté (bool)
  - scores Hartmann sur les 7 classes
  - score latent (cosine à la couche 20)

Agrégat par combinaison :
  - refusal_rate = n_refus / N
  - target_mean, target_std, target_min, target_max  (sur les runs non-refus)
  - latent_mean, latent_std

Verdicts :
  STABLE    std < 0.10  et  refusal_rate ≤ 0.10
  VARIABLE  std < 0.20  et  refusal_rate ≤ 0.30
  INSTABLE  std ≥ 0.20  ou  refusal_rate > 0.30
  FRAGILE   refusal_rate ≥ 0.60  (trop de refus pour estimer la variance)

Les résultats sont écrits dans data/generation_stability.json.

Usage :
    python -m src.measure_generation_stability
    python -m src.measure_generation_stability --n-runs 20
    python -m src.measure_generation_stability --dry-run
"""
from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime
from pathlib import Path

import torch
from transformers import pipeline

from src.eval_latent import latent_score as _latent_score
from src.model_loader import ModelWrapper
from src.steering import GENERATION_TEMPERATURE, generate_steered

# ---------------------------------------------------------------------------
# Détection de refus (identique à web/app.py)
# ---------------------------------------------------------------------------

_REFUSAL_PREFIXES = (
    "i apologize", "i'm not able", "i am not able",
    "i cannot", "i can't", "as an ai", "i'm an ai",
    "i am an ai", "i'm unable", "i am unable",
    "i'm sorry, but", "i'm sorry, i",
)


def _is_refusal(text: str) -> bool:
    t = text.lower().strip()
    return any(t.startswith(p) for p in _REFUSAL_PREFIXES)


# ---------------------------------------------------------------------------
# Matrice de test
# Chaque entrée est une combinaison (prompt, émotion, alpha) documentée
# dans le golden set. Le champ golden_id pointe vers l'entrée source.
# ---------------------------------------------------------------------------

CASES: list[dict] = [
    {
        "key":       "env-joy-2.0",
        "golden_id": "env-joy-calibration",
        "prompt":    "Continue this story: She opened the envelope slowly and read the first line.",
        "emotion":   "joy",
        "alpha":     2.0,
        "note":      "Sweet spot documenté 83.8%",
    },
    {
        "key":       "env-anger-1.5",
        "golden_id": "env-anger-nonmonotone",
        "prompt":    "Continue this story: She opened the envelope slowly and read the first line.",
        "emotion":   "anger",
        "alpha":     1.5,
        "note":      "Peak non-monotone documenté 92.7%",
    },
    {
        "key":       "env-anger-2.0",
        "golden_id": "env-anger-nonmonotone",
        "prompt":    "Continue this story: She opened the envelope slowly and read the first line.",
        "emotion":   "anger",
        "alpha":     2.0,
        "note":      "Dégradation documentée 84.5%",
    },
    {
        "key":       "call-joy-1.5",
        "golden_id": "call-01",
        "prompt":    "Continue this story: He finally got the call he had been waiting for.",
        "emotion":   "joy",
        "alpha":     1.5,
        "note":      "Meilleure cohérence narrative documentée 95.4%",
    },
    {
        "key":       "call-anger-2.0",
        "golden_id": "call-01",
        "prompt":    "Continue this story: He finally got the call he had been waiting for.",
        "emotion":   "anger",
        "alpha":     2.0,
        "note":      "Optimal anger sur ce prompt documenté 93.5%",
    },
    {
        "key":       "park-joy-2.0",
        "golden_id": "park-01",
        "prompt":    "Describe a walk through a park on a sunny afternoon.",
        "emotion":   "joy",
        "alpha":     2.0,
        "note":      "Threshold effect documenté 92.6% — prompt original du golden set",
    },
    {
        "key":       "park-anger-2.0",
        "golden_id": "park-01",
        "prompt":    "Describe a walk through a park on a sunny afternoon.",
        "emotion":   "anger",
        "alpha":     2.0,
        "note":      "Inversion documentée 90.1% malgré le premise positif",
    },
]

VECTORS_DIR      = Path("vectors")
OUTPUT_PATH      = Path("data/generation_stability.json")
LAYER_IDX        = 20
MAX_NEW_TOKENS   = 120


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

def load_classifier():
    print("[stability] Chargement du classifieur Hartmann...")
    return pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        top_k=None,
        device="cpu",
    )


def hartmann_scores(classifier, text: str) -> dict[str, float]:
    results = classifier(text[:512], truncation=True)
    return {r["label"]: round(r["score"], 4) for r in results[0]}


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------

def verdict(target_std: float | None, refusal_rate: float) -> str:
    if refusal_rate >= 0.60:
        return "FRAGILE"
    if target_std is None:
        return "FRAGILE"   # tous des refus, pas de variance mesurable
    if refusal_rate > 0.30 or target_std >= 0.20:
        return "INSTABLE"
    if refusal_rate > 0.10 or target_std >= 0.10:
        return "VARIABLE"
    return "STABLE"


# ---------------------------------------------------------------------------
# Mesure d'un cas
# ---------------------------------------------------------------------------

def measure_case(
    case: dict,
    wrapper: ModelWrapper,
    vectors: dict[str, torch.Tensor],
    classifier,
    n_runs: int,
) -> dict:
    emotion   = case["emotion"]
    alpha     = case["alpha"]
    prompt    = case["prompt"]
    vector    = vectors[emotion]

    print(f"\n  [{case['key']}]  {emotion} α={alpha}  ({n_runs} runs)")
    print(f"  prompt : {prompt[:70]}{'...' if len(prompt) > 70 else ''}")

    runs = []
    for i in range(n_runs):
        text = generate_steered(
            wrapper, prompt, vector, alpha, LAYER_IDX, MAX_NEW_TOKENS
        )
        is_ref = _is_refusal(text)

        if is_ref:
            runs.append({"refusal": True})
            print(f"    run {i + 1:2d}  REFUS")
        else:
            scores  = hartmann_scores(classifier, text)
            latent  = _latent_score(wrapper, text, vector, LAYER_IDX)
            target  = scores.get(emotion, 0.0)
            runs.append({
                "refusal": False,
                "scores":  scores,
                "latent":  latent,
                "target":  target,
            })
            print(f"    run {i + 1:2d}  {emotion}={target:.3f}  latent={latent:.3f}")

    # Agrégation
    n_refusals  = sum(1 for r in runs if r["refusal"])
    refusal_rate = round(n_refusals / n_runs, 4)
    valid        = [r for r in runs if not r["refusal"]]

    if valid:
        targets      = [r["target"]  for r in valid]
        latents      = [r["latent"]  for r in valid]
        target_mean  = round(statistics.mean(targets), 4)
        target_std   = round(statistics.stdev(targets), 4) if len(targets) > 1 else 0.0
        target_min   = round(min(targets), 4)
        target_max   = round(max(targets), 4)
        latent_mean  = round(statistics.mean(latents), 4)
        latent_std   = round(statistics.stdev(latents), 4) if len(latents) > 1 else 0.0
    else:
        target_mean = target_std = target_min = target_max = None
        latent_mean = latent_std = None

    verd = verdict(target_std, refusal_rate)

    print(f"  → mean={target_mean}  std={target_std}  "
          f"range=[{target_min}–{target_max}]  "
          f"refusal={refusal_rate}  {verd}")

    return {
        "key":           case["key"],
        "golden_id":     case["golden_id"],
        "prompt":        prompt,
        "emotion":       emotion,
        "alpha":         alpha,
        "note":          case["note"],
        "n_runs":        n_runs,
        "n_valid":       len(valid),
        "refusal_rate":  refusal_rate,
        "target_mean":   target_mean,
        "target_std":    target_std,
        "target_min":    target_min,
        "target_max":    target_max,
        "latent_mean":   latent_mean,
        "latent_std":    latent_std,
        "verdict":       verd,
        "runs":          runs,
    }


# ---------------------------------------------------------------------------
# Rapport final
# ---------------------------------------------------------------------------

def print_report(results: list[dict]) -> None:
    print("\n" + "=" * 90)
    print("GÉNÉRATION — STABILITÉ PAR CAS")
    print("=" * 90)

    header = (
        f"{'CAS':<20}  {'EMO':<6}  {'α':>4}  "
        f"{'MEAN':>6}  {'STD':>6}  {'MIN':>6}  {'MAX':>6}  "
        f"{'REFUS':>6}  {'LATENT':>7}  VERDICT"
    )
    print(header)
    print("─" * 90)

    for r in results:
        mean_s   = f"{r['target_mean']:.3f}" if r["target_mean"] is not None else "  —  "
        std_s    = f"{r['target_std']:.3f}"  if r["target_std"]  is not None else "  —  "
        min_s    = f"{r['target_min']:.3f}"  if r["target_min"]  is not None else "  —  "
        max_s    = f"{r['target_max']:.3f}"  if r["target_max"]  is not None else "  —  "
        lat_s    = f"{r['latent_mean']:.3f}" if r["latent_mean"] is not None else "  —  "

        print(
            f"{r['key']:<20}  {r['emotion']:<6}  {r['alpha']:>4.1f}  "
            f"{mean_s:>6}  {std_s:>6}  {min_s:>6}  {max_s:>6}  "
            f"{r['refusal_rate']:>6.2f}  {lat_s:>7}  {r['verdict']}"
        )

    print()
    counts = {}
    for r in results:
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
    for v, n in sorted(counts.items()):
        print(f"  {v:<10} : {n} cas")

    # Implication pour le demo
    print()
    stable_cases = [r for r in results if r["verdict"] == "STABLE"]
    if stable_cases:
        print("Cas fiables pour la démo :")
        for r in stable_cases:
            print(f"  • {r['key']}  ({r['emotion']} α={r['alpha']})"
                  f"  mean={r['target_mean']:.3f} ± {r['target_std']:.3f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(n_runs: int = 10, dry_run: bool = False) -> None:
    print(f"[stability] Mesure de stochasticité — {len(CASES)} cas × {n_runs} runs")
    print(f"[stability] Total générations : {len(CASES) * n_runs}")

    if dry_run:
        print("\n[dry-run] Cas qui seraient testés :")
        for case in CASES:
            print(f"  {case['key']:<20}  {case['emotion']:<6}  α={case['alpha']}  {case['note']}")
        return

    # Chargement
    wrapper    = ModelWrapper()
    classifier = load_classifier()

    vectors: dict[str, torch.Tensor] = {}
    for emotion in {"joy", "anger"}:
        path = VECTORS_DIR / f"{emotion}_vector.pt"
        if not path.exists():
            raise FileNotFoundError(f"Vecteur manquant : {path}")
        vectors[emotion] = torch.load(path, weights_only=True)

    # Mesure
    print(f"\n[stability] Début des mesures  (temperature={GENERATION_TEMPERATURE})\n")
    results = []
    for case in CASES:
        result = measure_case(case, wrapper, vectors, classifier, n_runs)
        results.append(result)

    # Rapport
    print_report(results)

    # Sauvegarde
    output = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "n_runs":        n_runs,
        "temperature":   GENERATION_TEMPERATURE,
        "layer_idx":     LAYER_IDX,
        "results":       results,
    }
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n[stability] Résultats sauvegardés → {OUTPUT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-runs",  type=int,  default=10,    help="Nombre de runs par cas (défaut : 10)")
    parser.add_argument("--dry-run", action="store_true",      help="Affiche les cas sans générer")
    args = parser.parse_args()
    main(n_runs=args.n_runs, dry_run=args.dry_run)
