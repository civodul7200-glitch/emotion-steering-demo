"""
Investigation ciblée des refus.

Pour un prompt donné, compare trois conditions sur N runs :
  1. Base (pas de steering) → taux de refus de référence
  2. Joy α=2.0              → refus de steering joy
  3. Anger α=1.5            → refus de steering anger

Et pour chaque refus, affiche le texte complet pour catégoriser :
  - Refus de sécurité       ("I cannot continue this story")
  - Disclaimer AI           ("As an AI language model...")
  - Déstabilisation         (contenu incohérent ou hors-sujet)

Également teste l'impact de l'alpha sur le taux de refus (sweep α).

Usage :
    python -m src.investigate_refusals
    python -m src.investigate_refusals --n-runs 10
"""
from __future__ import annotations

import argparse

from pathlib import Path

import torch

from src.model_loader import ModelWrapper
from src.steering import GENERATION_TEMPERATURE, generate_base, generate_steered

VECTORS_DIR = Path("vectors")
LAYER_IDX   = 20

_REFUSAL_PREFIXES = (
    "i apologize", "i'm not able", "i am not able",
    "i cannot", "i can't", "as an ai", "i'm an ai",
    "i am an ai", "i'm unable", "i am unable",
    "i'm sorry, but", "i'm sorry, i",
)

_AI_DISCLAIMER = ("as an ai", "i'm an ai", "i am an ai")
_SAFETY_REFUSAL = ("i cannot", "i can't", "i'm not able", "i am not able",
                   "i'm unable", "i am unable", "i apologize")
_SORRY         = ("i'm sorry, but", "i'm sorry, i")


def _is_refusal(text: str) -> bool:
    t = text.lower().strip()
    return any(t.startswith(p) for p in _REFUSAL_PREFIXES)


def _refusal_type(text: str) -> str:
    t = text.lower().strip()
    if any(t.startswith(p) for p in _AI_DISCLAIMER):
        return "DISCLAIMER-AI"
    if any(t.startswith(p) for p in _SAFETY_REFUSAL):
        return "SÉCURITÉ"
    if any(t.startswith(p) for p in _SORRY):
        return "SORRY"
    return "AUTRE"


def run_condition(
    label: str,
    gen_fn,
    n_runs: int,
) -> dict:
    print(f"\n  ── {label} ──")
    refusals = []
    valids   = []

    for i in range(n_runs):
        text     = gen_fn()
        is_ref   = _is_refusal(text)
        if is_ref:
            rtype = _refusal_type(text)
            refusals.append({"type": rtype, "text": text})
            print(f"    run {i+1:2d}  {rtype:<16}  {text[:70].replace(chr(10),' ')!r}")
        else:
            valids.append(text)
            print(f"    run {i+1:2d}  OK  {text[:70].replace(chr(10),' ')!r}")

    refusal_rate = len(refusals) / n_runs
    print(f"  → refusal_rate = {refusal_rate:.1%}  "
          f"({len(refusals)}/{n_runs})")
    if refusals:
        types = {}
        for r in refusals:
            types[r["type"]] = types.get(r["type"], 0) + 1
        print(f"  → types : {types}")

    return {"label": label, "refusal_rate": refusal_rate,
            "refusals": refusals, "valids": valids}


def run_alpha_sweep(
    wrapper: ModelWrapper,
    vector: torch.Tensor,
    prompt: str,
    emotion: str,
    alphas: list[float],
    n_runs: int,
) -> None:
    print(f"\n{'='*60}")
    print(f"SWEEP ALPHA — {emotion}  (N={n_runs} par alpha)")
    print(f"Prompt : {prompt[:70]}")
    print(f"{'='*60}")
    print(f"  {'α':>5}  {'refusal_rate':>13}  {'n_refus':>8}")
    print(f"  {'─'*5}  {'─'*13}  {'─'*8}")

    for alpha in alphas:
        n_ref = 0
        for _ in range(n_runs):
            text  = generate_steered(wrapper, prompt, vector,
                                     alpha, LAYER_IDX, max_new_tokens=120)
            if _is_refusal(text):
                n_ref += 1
        rate = n_ref / n_runs
        bar  = "█" * int(rate * 20)
        print(f"  {alpha:>5.1f}  {rate:>13.1%}  {n_ref:>8}/{n_runs}  {bar}")


def main(n_runs: int = 5) -> None:
    print(f"[investigate] N={n_runs} runs par condition")
    print(f"[investigate] temperature={GENERATION_TEMPERATURE}")

    wrapper = ModelWrapper()

    joy_vector   = torch.load(VECTORS_DIR / "joy_vector.pt",   weights_only=True)
    anger_vector = torch.load(VECTORS_DIR / "anger_vector.pt", weights_only=True)

    # ── Prompt 1 : continuation narrative (beaucoup de refus observés) ──
    prompt_narrative = (
        "Continue this story: "
        "She opened the envelope slowly and read the first line."
    )
    print(f"\n{'='*60}")
    print("PROMPT NARRATIF  (Continue this story...)")
    print(f"{'='*60}")

    run_condition(
        "BASE (pas de steering)",
        lambda: generate_base(wrapper, prompt_narrative, max_new_tokens=120),
        n_runs,
    )
    run_condition(
        "JOY α=2.0",
        lambda: generate_steered(wrapper, prompt_narrative,
                                 joy_vector, 2.0, LAYER_IDX, 120),
        n_runs,
    )
    run_condition(
        "ANGER α=1.5",
        lambda: generate_steered(wrapper, prompt_narrative,
                                 anger_vector, 1.5, LAYER_IDX, 120),
        n_runs,
    )

    # ── Prompt 2 : description (0 refus observés) ──
    prompt_descriptif = "Describe a walk through a park on a sunny afternoon."
    print(f"\n{'='*60}")
    print("PROMPT DESCRIPTIF  (Describe a walk...)")
    print(f"{'='*60}")

    run_condition(
        "BASE (pas de steering)",
        lambda: generate_base(wrapper, prompt_descriptif, max_new_tokens=120),
        n_runs,
    )
    run_condition(
        "JOY α=2.0",
        lambda: generate_steered(wrapper, prompt_descriptif,
                                 joy_vector, 2.0, LAYER_IDX, 120),
        n_runs,
    )
    run_condition(
        "ANGER α=2.0",
        lambda: generate_steered(wrapper, prompt_descriptif,
                                 anger_vector, 2.0, LAYER_IDX, 120),
        n_runs,
    )

    # ── Sweep alpha : joie sur prompt narratif ──
    run_alpha_sweep(
        wrapper, joy_vector, prompt_narrative, "joy",
        alphas=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
        n_runs=n_runs,
    )

    # ── Sweep alpha : anger sur prompt narratif ──
    run_alpha_sweep(
        wrapper, anger_vector, prompt_narrative, "anger",
        alphas=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
        n_runs=n_runs,
    )

    print("\n[investigate] Terminé.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-runs", type=int, default=5,
                        help="Runs par condition (défaut : 5)")
    args = parser.parse_args()
    main(n_runs=args.n_runs)
