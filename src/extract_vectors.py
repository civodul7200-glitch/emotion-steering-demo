"""
Phase 4 — extraction des vecteurs contrastifs émotion − neutre.

Algorithme :
  1. Pour chaque texte du corpus : forward pass → hidden state du dernier token (couche 20)
  2. Moyenne par classe (joy, anger, neutral)
  3. joy_vector   = mean(joy_hiddens)   − mean(neutral_hiddens)
     anger_vector = mean(anger_hiddens) − mean(neutral_hiddens)
  4. Sauvegarde dans vectors/

Note : on utilise le texte brut (sans chat template) pour que le modèle
encode le contenu émotionnel, pas la structure instruction/réponse.

Couche 20 choisie en Phase 6 : meilleur cosine(joy, anger) parmi {8, 14, 20}.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from src.model_loader import ModelWrapper
from src.hooks import ActivationCapture

CORPUS_PATH = Path("data/corpus.json")
VECTORS_DIR = Path("vectors")
LAYER_IDX = 20


def encode_texts(
    wrapper: ModelWrapper,
    texts: list[str],
    layer_idx: int,
) -> torch.Tensor:
    """
    Retourne un tenseur [N, hidden_dim] avec le hidden state du dernier
    token utile pour chacun des N textes.
    """
    hiddens = []
    for i, text in enumerate(texts):
        inputs = wrapper.tokenizer(text, return_tensors="pt").to(wrapper.device)
        with ActivationCapture(wrapper.model, layer_idx) as cap:
            with torch.inference_mode():
                wrapper.model(**inputs)
        hiddens.append(cap.last_token())  # [hidden_dim]
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(texts)} textes encodés")
    return torch.stack(hiddens)  # [N, hidden_dim]


def extract_and_save(layer_idx: int = LAYER_IDX) -> None:
    with open(CORPUS_PATH) as f:
        corpus = json.load(f)

    by_label: dict[str, list[str]] = {"joy": [], "anger": [], "neutral": []}
    for item in corpus:
        by_label[item["label"]].append(item["text"])

    print(f"\n[extract] Corpus chargé : {sum(len(v) for v in by_label.values())} exemples")
    for label, texts in by_label.items():
        print(f"  {label:7} : {len(texts)} exemples")

    wrapper = ModelWrapper()
    print(f"\n[extract] Encodage sur couche {layer_idx} ...\n")

    means: dict[str, torch.Tensor] = {}
    for label, texts in by_label.items():
        print(f"→ {label}")
        hiddens = encode_texts(wrapper, texts, layer_idx)  # [N, 1536]
        means[label] = hiddens.mean(dim=0)                 # [1536]
        print(f"  mean norm : {means[label].norm():.2f}\n")

    # Vecteurs contrastifs : émotion − neutre
    joy_vector   = means["joy"]   - means["neutral"]
    anger_vector = means["anger"] - means["neutral"]

    print(f"[extract] joy_vector   norm : {joy_vector.norm():.4f}")
    print(f"[extract] anger_vector norm : {anger_vector.norm():.4f}")

    # Cosine similarity entre les deux directions (idéalement faible)
    cos = torch.nn.functional.cosine_similarity(
        joy_vector.unsqueeze(0), anger_vector.unsqueeze(0)
    ).item()
    print(f"[extract] cosine(joy, anger) : {cos:.4f}  (proche de 0 = directions distinctes)")

    VECTORS_DIR.mkdir(exist_ok=True)
    torch.save(joy_vector,   VECTORS_DIR / "joy_vector.pt")
    torch.save(anger_vector, VECTORS_DIR / "anger_vector.pt")
    print(f"\n[extract] Vecteurs sauvegardés dans {VECTORS_DIR}/")
    print("  joy_vector.pt")
    print("  anger_vector.pt")


if __name__ == "__main__":
    extract_and_save(layer_idx=LAYER_IDX)
