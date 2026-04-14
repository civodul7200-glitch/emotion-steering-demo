"""
Phase 5 — steering brut.

Principe : pendant la génération, à chaque forward pass sur la couche cible,
on ajoute alpha * vector aux hidden states → le modèle "voit" un espace
latent décalé dans la direction émotionnelle.

Le hook retourne le tenseur modifié (PyTorch remplace alors la sortie
du module), ce qui propage le décalage dans toutes les couches suivantes.
"""
from __future__ import annotations

from pathlib import Path

import torch

from src.model_loader import ModelWrapper

VECTORS_DIR = Path("vectors")

# Source de vérité unique pour les paramètres de génération.
# generate_base() et generate_steered() doivent toujours utiliser ces valeurs
# pour que les comparaisons base / steered restent valides.
GENERATION_TEMPERATURE: float = 0.7
GENERATION_DO_SAMPLE:   bool  = True


# ----------------------------------------------------------------------
# Steering hook (modifie les activations, ne fait pas que les lire)
# ----------------------------------------------------------------------

class SteeringHook:
    """
    Context manager qui injecte un vecteur dans les hidden states
    d'une couche pendant toute la durée du bloc `with`.

    Différence avec ActivationCapture : le hook retourne une valeur,
    ce qui remplace la sortie du module pour les couches suivantes.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        layer_idx: int,
        vector: torch.Tensor,   # [hidden_dim]
        alpha: float,
    ) -> None:
        self.vector = vector
        self.alpha = alpha
        target = model.model.layers[layer_idx]
        self._handle = target.register_forward_hook(self._hook)

    def _hook(self, module, input, output):
        # transformers 5.x : output est un Tensor
        # transformers 4.x : output est un tuple, output[0] = hidden states
        is_tuple = isinstance(output, tuple)
        h = output[0] if is_tuple else output

        v = self.vector.to(device=h.device, dtype=h.dtype)
        h_steered = h + self.alpha * v          # broadcast sur [batch, seq_len, hidden_dim]

        return (h_steered,) + output[1:] if is_tuple else h_steered

    def remove(self) -> None:
        self._handle.remove()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.remove()


# ----------------------------------------------------------------------
# Fonctions de génération
# ----------------------------------------------------------------------

def generate_base(wrapper: ModelWrapper, prompt: str, max_new_tokens: int = 150) -> str:
    return wrapper.generate(
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=GENERATION_TEMPERATURE,
    )


def generate_steered(
    wrapper: ModelWrapper,
    prompt: str,
    vector: torch.Tensor,
    alpha: float,
    layer_idx: int = 20,
    max_new_tokens: int = 150,
) -> str:
    """Génère avec le vecteur injecté à chaque pas de décodage."""
    inputs = wrapper.tokenizer(prompt, return_tensors="pt").to(wrapper.device)

    with SteeringHook(wrapper.model, layer_idx, vector, alpha):
        with torch.inference_mode():
            output_ids = wrapper.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=GENERATION_TEMPERATURE,
                do_sample=GENERATION_DO_SAMPLE,
                pad_token_id=wrapper.tokenizer.eos_token_id,
            )

    new_tokens = output_ids[0, inputs["input_ids"].shape[1]:]
    return wrapper.tokenizer.decode(new_tokens, skip_special_tokens=True)


# ----------------------------------------------------------------------
# Test Phase 5
# ----------------------------------------------------------------------

def test_steering() -> None:
    wrapper = ModelWrapper()

    joy_vector   = torch.load(VECTORS_DIR / "joy_vector.pt",   weights_only=True)
    anger_vector = torch.load(VECTORS_DIR / "anger_vector.pt", weights_only=True)

    # Prompts de complétion créative — évitent la réponse boilerplate "As an AI..."
    prompts = [
        "Continue this story: She opened the envelope slowly and read the first line.",
        "Continue this story: He walked into the office and everyone turned to look at him.",
        "Continue this story: The phone rang at 3am and she recognized the number.",
    ]

    alphas = [1, 2]

    for prompt in prompts:
        print("=" * 70)
        print(f"PROMPT : {prompt}\n")

        base = generate_base(wrapper, prompt)
        print(f"[BASE]\n{base}\n")

        for alpha in alphas:
            steered_joy   = generate_steered(wrapper, prompt, joy_vector,   alpha)
            steered_anger = generate_steered(wrapper, prompt, anger_vector, alpha)

            print(f"[JOY   α={alpha}]\n{steered_joy}\n")
            print(f"[ANGER α={alpha}]\n{steered_anger}\n")

        print()


def test_isolation() -> None:
    """
    Validation Phase 9 : deux générations successives ne se polluent pas.

    Vérifie qu'après chaque génération steered, count_active_hooks() == 0
    et que la génération base suivante produit un texte cohérent.
    """
    from src.hooks import count_active_hooks

    wrapper    = ModelWrapper()
    joy_vector = torch.load(VECTORS_DIR / "joy_vector.pt", weights_only=True)
    prompt     = "Continue this story: She opened the door and stepped inside."

    # Avant toute génération : 0 hooks
    n = count_active_hooks(wrapper.model)
    assert n == 0, f"Hooks présents avant démarrage : {n}"
    print(f"[isolation] avant génération       : {n} hook(s) ✓")

    # Génération steered — le hook doit se supprimer en sortie de `with`
    _ = generate_steered(wrapper, prompt, joy_vector, alpha=2.0, layer_idx=20)
    n = count_active_hooks(wrapper.model)
    assert n == 0, f"Hook leak après steered (run 1) : {n}"
    print(f"[isolation] après steered (run 1)  : {n} hook(s) ✓")

    # Génération base — ne doit pas être affectée par le steered précédent
    base = generate_base(wrapper, prompt)
    assert len(base) > 20, "Base trop courte — possible pollution"
    n = count_active_hooks(wrapper.model)
    assert n == 0, f"Hook leak après base : {n}"
    print(f"[isolation] après base             : {n} hook(s) ✓")

    # Second steered — vérification que les runs s'enchaînent proprement
    _ = generate_steered(wrapper, prompt, joy_vector, alpha=2.0, layer_idx=20)
    n = count_active_hooks(wrapper.model)
    assert n == 0, f"Hook leak après steered (run 2) : {n}"
    print(f"[isolation] après steered (run 2)  : {n} hook(s) ✓")

    print("\n[isolation] PASS — aucune pollution entre générations.")


if __name__ == "__main__":
    test_steering()
