"""
Mesure de la stabilité du corpus d'extraction.

Deux analyses complémentaires :

1. Subsampling (N=20, subsample 35/44 sans remise)
   Pour chaque itération, on tire 35 exemples par classe sans remise et on
   ré-extrait le vecteur. On mesure cosine(vecteur_subsample, vecteur_complet).
   → Répond à : "les résultats dépendent-ils de quelques phrases spécifiques ?"
   Note : il s'agit de sous-échantillonnage sans remise, pas de bootstrap au sens
   statistique (tirage avec remise). Cette procédure mesure la sensibilité du
   vecteur à la composition du corpus, pas l'intervalle de confiance de la moyenne.

2. Leave-one-out
   On retire une phrase à la fois et on mesure l'impact sur le vecteur.
   → Répond à : "quelles phrases tirent le plus fort le vecteur ?"

Les deux analyses partagent les mêmes forward passes (encodage unique au démarrage).
Temps estimé : ~8 min sur MPS (132 encodages + calculs tensoriels légers).

Usage :
    python -m src.measure_corpus_stability
    python -m src.measure_corpus_stability --quick   # N=5 itérations, pour tester
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from src.hooks import ActivationCapture
from src.model_loader import ModelWrapper

CORPUS_PATH = Path("data/corpus.json")
VECTORS_DIR = Path("vectors")
LAYER_IDX   = 20
N_SUBSAMPLING = 20
SUBSAMPLE_N = 35   # sur 44 — ~80 %


# ---------------------------------------------------------------------------
# Encodage
# ---------------------------------------------------------------------------

def encode_all(
    wrapper: ModelWrapper,
    by_label: dict[str, list[str]],
    layer_idx: int,
) -> dict[str, torch.Tensor]:
    """
    Encode toutes les phrases du corpus, une seule fois.
    Retourne dict[label → Tensor[N, hidden_dim]] sur CPU en float32.
    """
    hiddens: dict[str, torch.Tensor] = {}
    for label, texts in sorted(by_label.items()):
        print(f"\n[stability] Encodage '{label}' ({len(texts)} phrases)...")
        label_hiddens = []
        for i, text in enumerate(texts):
            inputs = wrapper.tokenizer(text, return_tensors="pt").to(wrapper.device)
            with ActivationCapture(wrapper.model, layer_idx) as cap:
                with torch.inference_mode():
                    wrapper.model(**inputs)
            label_hiddens.append(cap.last_token().cpu().float())
            if (i + 1) % 10 == 0:
                print(f"  {i + 1}/{len(texts)}")
        hiddens[label] = torch.stack(label_hiddens)   # [N, hidden_dim]
    return hiddens


# ---------------------------------------------------------------------------
# Vecteur contrastif
# ---------------------------------------------------------------------------

def contrastive_vector(
    emotion_hiddens: torch.Tensor,   # [N, hidden_dim]
    neutral_hiddens: torch.Tensor,   # [M, hidden_dim]
) -> torch.Tensor:
    """mean(émotion) − mean(neutre), normalisé L2."""
    v = emotion_hiddens.mean(dim=0) - neutral_hiddens.mean(dim=0)
    return F.normalize(v, dim=0)


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

def run_subsampling(
    all_hiddens: dict[str, torch.Tensor],
    full_vectors: dict[str, torch.Tensor],
    emotions: list[str],
    n_iterations: int = N_SUBSAMPLING,
    subsample_n: int = SUBSAMPLE_N,
) -> dict[str, list[float]]:
    """
    Sous-échantillonnage sans remise : tire subsample_n/N exemples par classe,
    ré-extrait le vecteur, mesure cosine avec le vecteur complet. Répété n_iterations fois.

    Les indices émotion et neutre sont tirés indépendamment à chaque itération
    pour capturer la variabilité des deux côtés de la soustraction contrastive.

    Note : ceci est du sous-échantillonnage (sans remise), pas du bootstrap
    statistique (avec remise). Il mesure la sensibilité aux exemples spécifiques.
    """
    results: dict[str, list[float]] = {e: [] for e in emotions}
    n_neutral = len(all_hiddens["neutral"])

    print(f"\n[subsampling] N={n_iterations}, subsample={subsample_n}/{len(all_hiddens[emotions[0]])}")

    for i in range(n_iterations):
        line_parts = []
        for emotion in emotions:
            n_emotion = len(all_hiddens[emotion])

            idx_e = torch.randperm(n_emotion)[:subsample_n]
            idx_n = torch.randperm(n_neutral)[:subsample_n]

            v_boot = contrastive_vector(
                all_hiddens[emotion][idx_e],
                all_hiddens["neutral"][idx_n],
            )
            cos_val = cosine(v_boot, full_vectors[emotion])
            results[emotion].append(cos_val)
            line_parts.append(f"{emotion}: {cos_val:.4f}")

        print(f"  iter {i + 1:2d}/{n_iterations}  " + "  ".join(line_parts))

    return results


def report_subsampling(results: dict[str, list[float]]) -> None:
    thresholds = {
        "STABLE":   0.95,
        "MODÉRÉ":   0.90,
        "FRAGILE":  0.0,
    }

    print("\n" + "=" * 60)
    print("SUBSAMPLING — STABILITÉ DU CORPUS")
    print("=" * 60)

    for emotion, cosines in results.items():
        t = torch.tensor(cosines)
        mean_val = t.mean().item()
        std_val  = t.std().item()
        min_val  = t.min().item()
        max_val  = t.max().item()

        if mean_val >= thresholds["STABLE"]:
            verdict = "STABLE   ✓"
        elif mean_val >= thresholds["MODÉRÉ"]:
            verdict = "MODÉRÉ   ~"
        else:
            verdict = "FRAGILE  ✗"

        print(f"\n{emotion.upper()}")
        print(f"  mean  : {mean_val:.4f}")
        print(f"  std   : {std_val:.4f}")
        print(f"  range : [{min_val:.4f} – {max_val:.4f}]")
        print(f"  verdict : {verdict}")

        if std_val > 0.02:
            print("  → std élevé : le vecteur est sensible à la composition du sous-échantillon.")
        if min_val < 0.90:
            print("  → min < 0.90 : certains sous-échantillons produisent un vecteur significativement différent.")


# ---------------------------------------------------------------------------
# Leave-one-out
# ---------------------------------------------------------------------------

def run_leave_one_out(
    all_hiddens: dict[str, torch.Tensor],
    full_vectors: dict[str, torch.Tensor],
    emotions: list[str],
    by_label: dict[str, list[str]],
) -> dict[str, list[tuple[float, str]]]:
    """
    Pour chaque phrase de chaque classe émotion :
      - recalcule le vecteur sans cette phrase (neutre = moyenne complète)
      - mesure cosine(vecteur_sans_i, vecteur_complet)
      - pull_i = 1 − cosine  (plus c'est élevé, plus la phrase tire le vecteur)

    Retourne dict[emotion → list[(pull, texte)]] trié par pull décroissant.
    """
    results: dict[str, list[tuple[float, str]]] = {}
    neutral_mean = all_hiddens["neutral"].mean(dim=0)

    print("\n[leave-one-out] Calcul des influences par phrase...")

    for emotion in emotions:
        hiddens = all_hiddens[emotion]   # [N, hidden_dim]
        texts   = by_label[emotion]
        n       = len(hiddens)
        full_v  = full_vectors[emotion]

        influences = []
        for i in range(n):
            mask = torch.ones(n, dtype=torch.bool)
            mask[i] = False
            mean_without = hiddens[mask].mean(dim=0)
            v_without    = F.normalize(mean_without - neutral_mean, dim=0)
            cos_val      = cosine(v_without, full_v)
            pull         = round(1.0 - cos_val, 6)
            influences.append((pull, texts[i]))

        influences.sort(key=lambda x: -x[0])
        results[emotion] = influences

    return results


def report_leave_one_out(
    results: dict[str, list[tuple[float, str]]],
    top_n: int = 5,
) -> None:
    print("\n" + "=" * 60)
    print(f"LEAVE-ONE-OUT — PHRASES LES PLUS INFLUENTES (top {top_n})")
    print("=" * 60)

    for emotion, influences in results.items():
        print(f"\n{emotion.upper()}")
        print(f"  {'pull':>8}  texte")
        print(f"  {'─'*8}  {'─'*50}")
        for pull, text in influences[:top_n]:
            short = text[:70] + ("..." if len(text) > 70 else "")
            print(f"  {pull:>8.4f}  {short!r}")

        # Signalement si pull max > seuil
        max_pull = influences[0][0]
        if max_pull > 0.005:
            print(f"\n  ⚠ pull max = {max_pull:.4f} — cette phrase tire significativement le vecteur.")
        else:
            print(f"\n  pull max = {max_pull:.4f} — aucun outlier notable.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(quick: bool = False) -> None:
    n_iter = 5 if quick else N_SUBSAMPLING

    # Chargement corpus
    with open(CORPUS_PATH) as f:
        corpus = json.load(f)

    by_label: dict[str, list[str]] = {}
    for item in corpus:
        by_label.setdefault(item["label"], []).append(item["text"])

    emotions = sorted(lb for lb in by_label if lb != "neutral")

    print(f"[stability] Corpus : {sum(len(v) for v in by_label.values())} phrases")
    for label in sorted(by_label):
        print(f"  {label:8} : {len(by_label[label])} exemples")

    # Vecteurs complets (référence)
    full_vectors: dict[str, torch.Tensor] = {}
    for emotion in emotions:
        path = VECTORS_DIR / f"{emotion}_vector.pt"
        if not path.exists():
            raise FileNotFoundError(f"Vecteur manquant : {path}. Lancez d'abord extract_vectors.")
        v = torch.load(path, weights_only=True).cpu().float()
        full_vectors[emotion] = v

    # Encodage unique (toutes les phrases, une seule fois)
    wrapper    = ModelWrapper()
    all_hiddens = encode_all(wrapper, by_label, LAYER_IDX)

    # Subsampling
    subsampling_results = run_subsampling(all_hiddens, full_vectors, emotions, n_iterations=n_iter)
    report_subsampling(subsampling_results)

    # Leave-one-out
    loo_results = run_leave_one_out(all_hiddens, full_vectors, emotions, by_label)
    report_leave_one_out(loo_results)

    print("\n[stability] Analyse terminée.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="N=5 itérations (test rapide)")
    args = parser.parse_args()
    main(quick=args.quick)
