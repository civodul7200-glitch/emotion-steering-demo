"""
Layer search — finds the optimal injection layer for joy steering.

Tests layers [16, 18, 20, 22, 24] on 2 prompts with alpha=1.5.
Vectors are extracted per layer and cached; re-running skips extraction.

Run from project root:
    python experiments/layer_search.py

Output:
    vectors/experiments/joy_vector_layer{N}.pt  (one per layer)
    experiments/results_layer_search.csv
"""
from __future__ import annotations

import sys
import csv
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F
from transformers import pipeline

from src.eval_latent import latent_score
from src.extract_vectors import encode_texts
from src.model_loader import ModelWrapper
from src.steering import generate_base, generate_steered

# ── Config ───────────────────────────────────────────────────────────────

LAYERS         = [16, 18, 20, 22, 24]
ALPHA          = 1.5
MAX_NEW_TOKENS = 120
PROMPTS        = [
    "Today was different from all the other days because",
    "After reviewing all the information carefully, I concluded that",
]

CORPUS_PATH      = Path("data/corpus.json")
VECTORS_EXPR_DIR = Path("vectors/experiments")
OUTPUT_CSV       = Path("experiments/results_layer_search.csv")

# ── Helpers ───────────────────────────────────────────────────────────────

def log(msg: str) -> None:
    print(f"[layer_search] {msg}", flush=True)


def hartmann_joy(classifier, text: str) -> float:
    results = classifier(text[:512], truncation=True)
    return round({r["label"]: r["score"] for r in results[0]}.get("joy", 0.0), 4)


def first_50_words(text: str) -> str:
    return " ".join(text.split()[:50])


# ── Vector extraction ─────────────────────────────────────────────────────

def load_or_extract(
    wrapper: ModelWrapper,
    joy_texts: list[str],
    neutral_texts: list[str],
    layer: int,
) -> torch.Tensor:
    path = VECTORS_EXPR_DIR / f"joy_vector_layer{layer}.pt"
    if path.exists():
        log(f"layer {layer} — vector cached, loading {path.name}")
        return torch.load(path, weights_only=True)

    log(f"layer {layer} — extracting vector...")

    log(f"  encoding neutral ({len(neutral_texts)} texts)...")
    neutral_mean = encode_texts(wrapper, neutral_texts, layer).mean(dim=0)

    log(f"  encoding joy ({len(joy_texts)} texts)...")
    joy_mean = encode_texts(wrapper, joy_texts, layer).mean(dim=0)

    vector = F.normalize(joy_mean - neutral_mean, dim=0)
    torch.save(vector, path)
    log(f"  saved {path.name}  norm={vector.norm():.4f}")
    return vector


# ── Main ──────────────────────────────────────────────────────────────────

def run() -> None:
    VECTORS_EXPR_DIR.mkdir(parents=True, exist_ok=True)

    # Load corpus
    log("Loading corpus...")
    with open(CORPUS_PATH) as f:
        corpus = json.load(f)
    by_label: dict[str, list[str]] = {}
    for item in corpus:
        by_label.setdefault(item["label"], []).append(item["text"])
    log(f"  {sum(len(v) for v in by_label.values())} examples — {sorted(by_label.keys())}")

    # Load model and classifier
    log("Loading model (this may take a minute)...")
    wrapper = ModelWrapper()

    log("Loading Hartmann classifier...")
    classifier = pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        top_k=None,
        device="cpu",
    )

    # Base generation — layer-independent, run once per prompt
    log("\nBase generation (run once per prompt, no steering)...")
    base: dict[int, dict] = {}
    for p_idx, prompt in enumerate(PROMPTS, 1):
        log(f"  [{p_idx}/2] generating base for prompt {p_idx}...")
        t0 = time.time()
        text     = generate_base(wrapper, prompt, max_new_tokens=MAX_NEW_TOKENS)
        base_joy = hartmann_joy(classifier, text)
        base[p_idx] = {"text": text, "joy": base_joy}
        log(f"  [{p_idx}/2] base joy={base_joy:.4f}  ({time.time() - t0:.1f}s)")

    # Layer × prompt grid
    rows     = []
    n_total  = len(LAYERS) * len(PROMPTS)
    run_idx  = 0

    for layer in LAYERS:
        log(f"\n{'─' * 52}")
        log(f"Layer {layer}")

        vector = load_or_extract(wrapper, by_label["joy"], by_label["neutral"], layer)

        for p_idx, prompt in enumerate(PROMPTS, 1):
            run_idx += 1
            log(f"  [{run_idx}/{n_total}] layer={layer}  prompt {p_idx}/2 — steered generation...")
            t0 = time.time()

            text        = generate_steered(wrapper, prompt, vector,
                                           alpha=ALPHA, layer_idx=layer,
                                           max_new_tokens=MAX_NEW_TOKENS)
            steered_joy = hartmann_joy(classifier, text)
            latent      = latent_score(wrapper, text, vector, layer_idx=layer)
            delta       = round(steered_joy - base[p_idx]["joy"], 4)

            log(
                f"  [{run_idx}/{n_total}] done — "
                f"base={base[p_idx]['joy']:.3f}  steered={steered_joy:.3f}  "
                f"delta={delta:+.3f}  latent={latent:.3f}  "
                f"({time.time() - t0:.1f}s)"
            )

            rows.append({
                "layer":        layer,
                "prompt_id":    p_idx,
                "base_joy":     base[p_idx]["joy"],
                "steered_joy":  steered_joy,
                "delta":        delta,
                "latent":       latent,
                "text_preview": first_50_words(text),
            })

    # Write CSV
    fieldnames = ["layer", "prompt_id", "base_joy", "steered_joy", "delta", "latent", "text_preview"]
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    log(f"\nResults saved → {OUTPUT_CSV}")

    # Summary table
    print(f"\n{'layer':>6}  {'p':>2}  {'base':>6}  {'steered':>7}  {'delta':>6}  {'latent':>7}")
    print("─" * 46)
    for r in rows:
        print(
            f"{r['layer']:>6}  {r['prompt_id']:>2}  "
            f"{r['base_joy']:>6.3f}  {r['steered_joy']:>7.3f}  "
            f"{r['delta']:>+6.3f}  {r['latent']:>7.3f}"
        )


if __name__ == "__main__":
    run()
