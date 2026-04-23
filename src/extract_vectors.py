"""
Phase 4 — extraction des vecteurs contrastifs émotion − neutre.

Algorithme :
  1. Pour chaque texte du corpus : forward pass → hidden state du dernier token (couche 20)
  2. Moyenne par classe
  3. Pour chaque émotion non-neutre : vector = mean(émotion) − mean(neutral)
  4. Normalisation L2 + sauvegarde dans vectors/

Le script détecte automatiquement les labels présents dans corpus.json.
Toute nouvelle émotion ajoutée au corpus sera extraite sans modifier ce fichier.

Note : on utilise le texte brut (sans chat template) pour que le modèle
encode le contenu émotionnel, pas la structure instruction/réponse.

Couche 20 choisie en Phase 6 : meilleur cosine(joy, anger) parmi {8, 14, 20}.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F

from src.model_loader import ModelWrapper
from src.hooks import ActivationCapture

CORPUS_PATH = Path("data/corpus.json")
VECTORS_DIR = Path("vectors")
LAYER_IDX   = 22


def encode_texts(
    wrapper: ModelWrapper,
    texts: list[str],
    layer_idx: int,
) -> torch.Tensor:
    """
    Retourne un tenseur [N, hidden_dim] — hidden state du dernier token.

    last_token() capture le mot final de la phrase, qui dans un corpus émotionnel
    est souvent le terme le plus chargé ("lifting", "grin", "hurled", "content").
    Cela donne un signal lexical discriminatif entre émotions.
    seq_mean() lisse ce signal et rapproche toutes les représentations vers
    la moyenne globale "texte narratif court" — les cosines montent vers 0.99.
    """
    hiddens = []
    for i, text in enumerate(texts):
        inputs = wrapper.tokenizer(text, return_tensors="pt").to(wrapper.device)
        with ActivationCapture(wrapper.model, layer_idx) as cap:
            with torch.inference_mode():
                wrapper.model(**inputs)
        hiddens.append(cap.last_token())   # [hidden_dim]
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(texts)} textes encodés")
    return torch.stack(hiddens)            # [N, hidden_dim]


def extract_and_save(layer_idx: int = LAYER_IDX) -> None:
    with open(CORPUS_PATH) as f:
        corpus = json.load(f)

    # Regroupe par label — détecte automatiquement toutes les émotions présentes
    by_label: dict[str, list[str]] = {}
    for item in corpus:
        by_label.setdefault(item["label"], []).append(item["text"])

    emotions = [lb for lb in by_label if lb != "neutral"]

    print(f"\n[extract] Corpus chargé : {sum(len(v) for v in by_label.values())} exemples")
    for label in sorted(by_label):
        print(f"  {label:8} : {len(by_label[label])} exemples")
    print(f"\n[extract] Émotions à extraire : {emotions}")

    if "neutral" not in by_label:
        raise ValueError("Le corpus doit contenir une classe 'neutral'.")

    wrapper = ModelWrapper()
    print(f"\n[extract] Encodage sur couche {layer_idx} ...\n")

    means: dict[str, torch.Tensor] = {}
    for label in sorted(by_label):
        print(f"→ {label}")
        hiddens = encode_texts(wrapper, by_label[label], layer_idx)   # [N, hidden_dim]
        means[label] = hiddens.mean(dim=0)                             # [hidden_dim]
        print(f"  mean norm : {means[label].norm():.4f}\n")

    neutral_mean = means["neutral"]

    # Vecteurs contrastifs : émotion − neutre, normalisés.
    # Le neutre agit comme référence stable : phrases sans contenu émotionnel.
    VECTORS_DIR.mkdir(exist_ok=True)
    vectors: dict[str, torch.Tensor] = {}
    for emotion in emotions:
        v = means[emotion] - neutral_mean
        v = F.normalize(v, dim=0)
        vectors[emotion] = v
        torch.save(v, VECTORS_DIR / f"{emotion}_vector.pt")
        print(f"[extract] {emotion}_vector.pt  norm={v.norm():.4f}")

    # Matrice de cosine entre toutes les paires
    print(f"\n[extract] Cosine similarity entre directions :")
    emotion_list = sorted(emotions)
    for i, a in enumerate(emotion_list):
        for b in emotion_list[i + 1:]:
            cos = F.cosine_similarity(
                vectors[a].unsqueeze(0),
                vectors[b].unsqueeze(0),
            ).item()
            flag = "  ← overlap élevé" if abs(cos) > 0.6 else ""
            print(f"  cosine({a:8}, {b:8}) = {cos:+.4f}{flag}")

    print(f"\n[extract] {len(emotions)} vecteurs sauvegardés dans {VECTORS_DIR}/")


if __name__ == "__main__":
    extract_and_save(layer_idx=LAYER_IDX)
