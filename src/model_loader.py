"""
Chargement du modèle — module principal.

Structuré en classe ModelWrapper pour que les hooks (Phase 2),
l'extraction de vecteurs (Phase 4) et le steering (Phase 5)
puissent tous partager la même instance sans recharger le modèle.

Avertissement premier lancement : ~3 GB téléchargés dans ~/.cache/huggingface/
"""
from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    print("[WARNING] MPS non disponible, fallback CPU.")
    return torch.device("cpu")


class ModelWrapper:
    """
    Charge le modèle et expose generate().

    Instancier une seule fois par processus — le modèle reste en mémoire
    GPU/MPS pour toutes les générations successives.
    """

    MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

    def __init__(
        self,
        model_id: str = MODEL_ID,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float16,
    ) -> None:
        self.model_id = model_id
        self.device = device or get_device()
        self.dtype = dtype

        print(f"[model_loader] Using device: {self.device}")
        print("[model_loader] NOTE: premier lancement = ~3 GB de téléchargement.")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            dtype=self.dtype,
            device_map={"": self.device},  # explicit, pas "auto"
            trust_remote_code=True,
        )
        self.model.eval()
        print("[model_loader] Modèle prêt.")

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 200,
        temperature: float = 0.7,
    ) -> str:
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        new_tokens = output_ids[0, inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)


def main() -> None:
    wrapper = ModelWrapper()
    prompt = "Describe a walk through a park on a sunny afternoon. Use vivid sensory details."
    print(f"\n[Prompt]\n{prompt}\n")
    print(f"[Response]\n{wrapper.generate(prompt)}\n")


if __name__ == "__main__":
    main()
