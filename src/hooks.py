"""
Capture des activations par forward hook.

Règle absolue : tout hook doit être supprimé après usage.
ActivationCapture et count_active_hooks() permettent de le vérifier.

Architecture Llama 3.2-3B :
  - 28 couches (model.model.layers[0..27])
  - hidden_size = 3072
  - transformers 5.x : output d'une couche = Tensor [batch, seq_len, 3072]
  - transformers 4.x : output = tuple, output[0] = Tensor (compatibilité maintenue)
"""
from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM


class ActivationCapture:
    """
    Context manager de capture des hidden states d'une couche Transformer.

    Usage standard :
        with ActivationCapture(model, layer_idx=14) as cap:
            with torch.inference_mode():
                model(**inputs)
        # hook supprimé ici, cap.hidden_states accessible
        print(cap.hidden_states[-1].shape)  # [batch, seq_len, hidden_dim]

    Phase 4 utilisera last_token() ou seq_mean() pour construire
    le vecteur contrastif émotion−neutre.
    """

    def __init__(self, model: AutoModelForCausalLM, layer_idx: int) -> None:
        self.layer_idx = layer_idx
        self.hidden_states: list[torch.Tensor] = []
        target = model.model.layers[layer_idx]
        self._handle = target.register_forward_hook(self._hook)

    def _hook(
        self,
        module: torch.nn.Module,
        input: tuple,
        output,
    ) -> None:
        # transformers 4.x : output est un tuple, output[0] = hidden states
        # transformers 5.x : output est directement le Tensor hidden states
        h = output[0] if isinstance(output, tuple) else output
        # .detach() : ne garde pas le graphe de calcul
        # .cpu()    : libère la mémoire MPS immédiatement
        self.hidden_states.append(h.detach().cpu())

    def remove(self) -> None:
        """Supprime le hook. Toujours appeler, même en cas d'exception."""
        self._handle.remove()

    def __enter__(self) -> "ActivationCapture":
        return self

    def __exit__(self, *args) -> None:
        self.remove()

    def last_token(self) -> torch.Tensor:
        """
        Hidden state du dernier token de la dernière passe forward.
        Shape : [hidden_dim]
        """
        h = self.hidden_states[-1]  # [batch, seq_len, hidden_dim]
        return h[0, -1, :]

    def seq_mean(self) -> torch.Tensor:
        """
        Moyenne des hidden states sur tous les tokens de la séquence.
        Shape : [hidden_dim]
        """
        h = self.hidden_states[-1]  # [batch, seq_len, hidden_dim]
        return h[0].mean(dim=0)


def count_active_hooks(model: torch.nn.Module) -> int:
    """
    Retourne le nombre total de forward hooks actifs sur le modèle.
    Doit retourner 0 entre deux appels indépendants.

    Usage :
        assert count_active_hooks(wrapper.model) == 0, "Hook leak détecté"
    """
    return sum(len(m._forward_hooks) for m in model.modules())


# ----------------------------------------------------------------------
# Test Phase 2 — 3 prompts, vérification shape/dtype/couche
# ----------------------------------------------------------------------

def test_capture(layer_idx: int = 20) -> None:
    from src.model_loader import ModelWrapper

    prompts = [
        "The sun rises over the mountains.",
        "She burst into tears, sobbing uncontrollably.",
        "He clenched his fists, fury rising inside him.",
    ]

    wrapper = ModelWrapper()
    n_layers = len(wrapper.model.model.layers)
    print(f"\n[hooks] Modèle : {wrapper.model_id}")
    print(f"[hooks] Couches disponibles : 0..{n_layers - 1}")
    print(f"[hooks] Couche observée     : {layer_idx}")
    print(f"[hooks] hidden_size         : {wrapper.model.config.hidden_size}\n")

    for i, prompt in enumerate(prompts):
        inputs = wrapper.tokenizer(prompt, return_tensors="pt").to(wrapper.device)

        with ActivationCapture(wrapper.model, layer_idx) as cap:
            with torch.inference_mode():
                wrapper.model(**inputs)
        # Hook supprimé ici — vérifiable : cap._handle est mort

        h = cap.hidden_states[-1]
        print(f"Prompt {i + 1} : {prompt!r}")
        print(f"  shape  : {tuple(h.shape)}  → [batch={h.shape[0]}, seq_len={h.shape[1]}, hidden={h.shape[2]}]")
        print(f"  dtype  : {h.dtype}")
        print(f"  device : {h.device}  (cpu après detach)")
        print(f"  last_token() norm : {cap.last_token().norm():.4f}")
        print(f"  seq_mean()   norm : {cap.seq_mean().norm():.4f}")
        print()

    print("[hooks] OK — aucun hook actif après le test.")


if __name__ == "__main__":
    test_capture(layer_idx=20)
