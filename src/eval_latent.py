"""
Évaluation triple : Hartmann (surface) + Latent (espace interne) + LLM judge (registre narratif).

Fonctions exportées :
    latent_score(wrapper, text, vector, layer_idx)  → float [-1, 1]
    llm_judge_score(wrapper, text, emotion)          → float [0, 1] | None
    score_triple(wrapper, text, emotion, vector, classifier, layer_idx)
        → dict avec les 3 scores + gaps

Principe de la mesure latente :
    On repasse le texte généré dans le modèle (forward pass sans hook de steering).
    On capture les hidden states à la couche 20 (même couche que le steering).
    On moyenne sur les tokens → représentation globale du texte.
    cosine(représentation, vecteur_émotion) = alignement interne.

    Si le steering a réellement modifié la géométrie interne, ce score sera élevé
    même quand le texte n'utilise pas de marqueurs lexicaux explicites
    (cas où Hartmann dit "neutral" mais le texte est narrativement émotionnel).

Principe du LLM judge :
    On demande au même modèle d'évaluer le registre émotionnel du texte.
    temperature=0.1 → quasi-déterministe.
    max_new_tokens=10 → force une réponse courte (un nombre).
    Le modèle comprend le registre narratif là où Hartmann lit seulement
    les marqueurs lexicaux de surface.

Interprétation des gaps :
    latent élevé, judge élevé, hartmann bas  → émotion encodée et lisible,
                                               mais sans marqueurs lexicaux explicites
    latent élevé, judge bas, hartmann bas    → émotion interne supprimée en surface
                                               (RLHF behavioral layer)
    latent bas,   judge élevé, hartmann élevé → prompt-driven, pas steering réel
"""
from __future__ import annotations

import re

import torch
import torch.nn.functional as F

from src.hooks import ActivationCapture
from src.model_loader import ModelWrapper

LAYER_IDX = 20

# ----------------------------------------------------------------------
# Prompt du LLM judge
# ----------------------------------------------------------------------

_JUDGE_PROMPT = """\
You are evaluating the emotional content of a generated text.
Rate the intensity of {emotion} in the text below.
Respond with only a decimal number between 0.0 and 1.0. Nothing else.

0.0 = no {emotion} at all
0.5 = moderate {emotion}
1.0 = very strong {emotion}

Text:
{text}

Score:"""


# ----------------------------------------------------------------------
# Mesure latente
# ----------------------------------------------------------------------

def latent_score(
    wrapper: ModelWrapper,
    text: str,
    vector: torch.Tensor,
    layer_idx: int = LAYER_IDX,
) -> float:
    """
    Cosine similarity entre la représentation interne du texte (couche layer_idx)
    et le vecteur émotionnel.

    Le texte est passé tel quel — pas de chat template, pas de steering.
    On mesure ce que le modèle "pense" du texte dans son espace interne.

    Retourne un float dans [-1, 1].
    Valeurs typiques : 0.0–0.3 pour du texte neutre, 0.5+ pour du texte aligné.
    """
    inputs = wrapper.tokenizer(text, return_tensors="pt").to(wrapper.device)

    with ActivationCapture(wrapper.model, layer_idx) as cap:
        with torch.inference_mode():
            wrapper.model(**inputs)

    # seq_mean() retourne [hidden_dim] en float16, sur CPU
    h = cap.seq_mean().float()                       # [hidden_dim]
    v = vector.cpu().float()                         # [hidden_dim]

    cos = F.cosine_similarity(h.unsqueeze(0), v.unsqueeze(0)).item()
    return round(cos, 4)


# ----------------------------------------------------------------------
# LLM judge
# ----------------------------------------------------------------------

def llm_judge_score(
    wrapper: ModelWrapper,
    text: str,
    emotion: str,
) -> float | None:
    """
    Demande au modèle d'évaluer l'intensité émotionnelle du texte.
    temperature=0.1, max_new_tokens=10 → réponse quasi-déterministe.

    Retourne un float dans [0, 1], ou None si le parsing échoue.
    """
    prompt = _JUDGE_PROMPT.format(emotion=emotion, text=text[:400])
    messages = [{"role": "user", "content": prompt}]
    formatted = wrapper.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = wrapper.tokenizer(formatted, return_tensors="pt").to(wrapper.device)

    with torch.inference_mode():
        output_ids = wrapper.model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False,
            pad_token_id=wrapper.tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0, inputs["input_ids"].shape[1]:]
    raw = wrapper.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    # Extrait le premier nombre flottant ou entier trouvé
    match = re.search(r"1\.0+|0\.\d+|[01](?!\d)", raw)
    if match:
        val = float(match.group())
        return round(min(max(val, 0.0), 1.0), 4)

    return None   # parse failure — raw loggé dans score_triple


# ----------------------------------------------------------------------
# Score triple
# ----------------------------------------------------------------------

def score_triple(
    wrapper: ModelWrapper,
    text: str,
    emotion: str,
    vector: torch.Tensor,
    classifier,
    layer_idx: int = LAYER_IDX,
) -> dict:
    """
    Calcule les 3 scores pour un texte généré.

    Retourne :
        hartmann  : surface lexicale (classifieur j-hartmann)
        latent    : alignement interne à la couche layer_idx
        llm_judge : score narratif donné par le modèle lui-même
        gap_latent_hartmann : latent − hartmann (positif = émotion interne non lexicalisée)
        gap_judge_hartmann  : llm_judge − hartmann
    """
    # Hartmann
    results = classifier(text[:512], truncation=True)
    hartmann_map = {r["label"]: r["score"] for r in results[0]}
    hartmann = round(hartmann_map.get(emotion, 0.0), 4)

    # Latent
    lat = latent_score(wrapper, text, vector, layer_idx)

    # LLM judge
    judge_raw = llm_judge_score(wrapper, text, emotion)
    judge = judge_raw  # peut être None

    return {
        "hartmann":             hartmann,
        "latent":               lat,
        "llm_judge":            judge,
        "gap_latent_hartmann":  round(lat - hartmann, 4),
        "gap_judge_hartmann":   round((judge or 0.0) - hartmann, 4),
    }


# ----------------------------------------------------------------------
# Démo standalone
# ----------------------------------------------------------------------

def _demo() -> None:
    from pathlib import Path
    from transformers import pipeline

    VECTORS_DIR = Path("vectors")

    print("[eval_latent] Chargement classifieur...")
    classifier = pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        top_k=None,
        device="cpu",
    )

    print("[eval_latent] Chargement modèle et vecteurs...")
    wrapper      = ModelWrapper()
    joy_vector   = torch.load(VECTORS_DIR / "joy_vector.pt",   weights_only=True)
    anger_vector = torch.load(VECTORS_DIR / "anger_vector.pt", weights_only=True)

    from src.steering import generate_base, generate_steered

    prompt = "Continue this story: She opened the envelope slowly and read the first line."
    cases = [
        ("base",        lambda: generate_base(wrapper, prompt),                            "joy",   joy_vector),
        ("joy  α=1.5",  lambda: generate_steered(wrapper, prompt, joy_vector,   1.5, 20), "joy",   joy_vector),
        ("joy  α=2.0",  lambda: generate_steered(wrapper, prompt, joy_vector,   2.0, 20), "joy",   joy_vector),
        ("anger α=1.5", lambda: generate_steered(wrapper, prompt, anger_vector, 1.5, 20), "anger", anger_vector),
    ]

    print(f"\nPrompt : {prompt}\n")
    print(f"{'Condition':<14}  {'Hartmann':>8}  {'Latent':>8}  {'Judge':>7}  {'ΔL−H':>7}  {'ΔJ−H':>7}")
    print("-" * 62)

    for label, gen_fn, emotion, vec in cases:
        text   = gen_fn()
        scores = score_triple(wrapper, text, emotion, vec, classifier)
        judge  = f"{scores['llm_judge']:.4f}" if scores["llm_judge"] is not None else "  None"
        print(
            f"{label:<14}  "
            f"{scores['hartmann']:>8.4f}  "
            f"{scores['latent']:>8.4f}  "
            f"{judge:>7}  "
            f"{scores['gap_latent_hartmann']:>+7.4f}  "
            f"{scores['gap_judge_hartmann']:>+7.4f}"
        )
        print(f"  → {text[:90].strip()!r}")
        print()


if __name__ == "__main__":
    _demo()
