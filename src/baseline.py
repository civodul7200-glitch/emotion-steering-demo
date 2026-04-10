"""
Phase 8 — baseline de comparaison.

Trois conditions pour chaque prompt :
  BASE     : prompt neutre, pas d'instruction émotionnelle
  PROMPTED : instruction dans le system message ("respond in a joyful/angry tone")
  STEERED  : activation steering latent (vecteur couche 20, alpha=2)

Objectif : dire honnêtement si le steering apporte un effet distinct
ou si le prompting direct suffit.
"""
from __future__ import annotations

from pathlib import Path

import torch
from transformers import pipeline

from src.load_model import ModelWrapper
from src.steering import generate_base, generate_steered

VECTORS_DIR = Path("vectors")
LAYER_IDX   = 20
ALPHA       = 2.0

EVAL_PROMPTS = [
    "Continue this story: She opened the envelope slowly and read the first line.",
    "Continue this story: He walked into the office and everyone turned to look at him.",
    "Continue this story: The phone rang at 3am and she recognized the number.",
    "Continue this story: He found the old photograph at the bottom of the drawer.",
    "Write two sentences describing someone waiting for important news.",
]

SYSTEM_JOY   = "Respond in a joyful, warm, enthusiastic tone."
SYSTEM_ANGER = "Respond in an angry, frustrated, hostile tone."


def generate_prompted(
    wrapper: ModelWrapper,
    prompt: str,
    system: str,
    max_new_tokens: int = 120,
) -> str:
    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": prompt},
    ]
    text   = wrapper.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = wrapper.tokenizer(text, return_tensors="pt").to(wrapper.device)
    with torch.inference_mode():
        output_ids = wrapper.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            pad_token_id=wrapper.tokenizer.eos_token_id,
        )
    new_tokens = output_ids[0, inputs["input_ids"].shape[1]:]
    return wrapper.tokenizer.decode(new_tokens, skip_special_tokens=True)


def score_emotion(classifier, text: str) -> dict[str, float]:
    results = classifier(text[:512], truncation=True)
    return {r["label"]: r["score"] for r in results[0]}


def run_baseline() -> None:
    print("[baseline] Chargement classifieur...")
    classifier = pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        top_k=None,
        device="cpu",
    )

    wrapper      = ModelWrapper()
    joy_vector   = torch.load(VECTORS_DIR / "joy_vector.pt",   weights_only=True)
    anger_vector = torch.load(VECTORS_DIR / "anger_vector.pt", weights_only=True)

    results = []  # liste de dicts pour l'affichage

    for prompt in EVAL_PROMPTS:
        short = prompt[:45]
        print(f"\n→ {short}...")

        base          = generate_base(wrapper, prompt, max_new_tokens=120)
        prompted_joy  = generate_prompted(wrapper, prompt, SYSTEM_JOY)
        prompted_ang  = generate_prompted(wrapper, prompt, SYSTEM_ANGER)
        steered_joy   = generate_steered(wrapper, prompt, joy_vector,   ALPHA, LAYER_IDX, max_new_tokens=120)
        steered_anger = generate_steered(wrapper, prompt, anger_vector, ALPHA, LAYER_IDX, max_new_tokens=120)

        s_base  = score_emotion(classifier, base)
        s_pjoy  = score_emotion(classifier, prompted_joy)
        s_pang  = score_emotion(classifier, prompted_ang)
        s_sjoy  = score_emotion(classifier, steered_joy)
        s_sang  = score_emotion(classifier, steered_anger)

        results.append({
            "prompt":    short,
            "emotion":   "joy",
            "base":      round(s_base["joy"],  3),
            "prompted":  round(s_pjoy["joy"],  3),
            "steered":   round(s_sjoy["joy"],  3),
        })
        results.append({
            "prompt":    short,
            "emotion":   "anger",
            "base":      round(s_base["anger"], 3),
            "prompted":  round(s_pang["anger"], 3),
            "steered":   round(s_sang["anger"], 3),
        })

    # --- tableau ---
    print("\n\n" + "=" * 78)
    print(f"{'Prompt':45} {'emo':5}  {'base':>6}  {'prompt':>6}  {'steer':>6}  {'Δprompt':>8}  {'Δsteer':>7}")
    print("-" * 78)
    for r in results:
        dp = r["prompted"] - r["base"]
        ds = r["steered"]  - r["base"]
        winner = "P" if dp > ds else "S"   # P=prompt gagne, S=steer gagne
        print(
            f"{r['prompt']:45} {r['emotion']:5}  "
            f"{r['base']:6.3f}  {r['prompted']:6.3f}  {r['steered']:6.3f}  "
            f"{dp:+8.3f}  {ds:+7.3f}  [{winner}]"
        )

    # --- synthèse ---
    joy_rows   = [r for r in results if r["emotion"] == "joy"]
    anger_rows = [r for r in results if r["emotion"] == "anger"]

    for label, rows in [("joy", joy_rows), ("anger", anger_rows)]:
        mean_dp = sum(r["prompted"] - r["base"] for r in rows) / len(rows)
        mean_ds = sum(r["steered"]  - r["base"] for r in rows) / len(rows)
        print(f"\n  {label:5}  Δ moyen prompt={mean_dp:+.3f}  Δ moyen steer={mean_ds:+.3f}", end="")
        if mean_dp > mean_ds:
            print("  → prompting gagne en moyenne")
        elif mean_ds > mean_dp:
            print("  → steering gagne en moyenne")
        else:
            print("  → égalité")


if __name__ == "__main__":
    run_baseline()
