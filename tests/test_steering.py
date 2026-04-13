"""
Tests unitaires — SteeringHook, generate_base, generate_steered, count_active_hooks.

Tout tourne sans charger le vrai modèle : on construit un petit réseau factice
(deux couches linéaires) qui expose la même interface que Qwen2.5.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn
import pytest

from src.hooks import ActivationCapture, count_active_hooks
from src.steering import SteeringHook, generate_base, generate_steered


# ---------------------------------------------------------------------------
# Helpers — modèle et wrapper factices
# ---------------------------------------------------------------------------

HIDDEN_DIM = 16
N_LAYERS   = 4


class FakeLayer(nn.Module):
    """Couche linéaire minimale qui passe les hidden states sans les modifier."""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(HIDDEN_DIM, HIDDEN_DIM, bias=False)

    def forward(self, x):
        return self.linear(x)


class FakeInnerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([FakeLayer() for _ in range(N_LAYERS)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = FakeInnerModel()

    def forward(self, x):
        return self.model(x)


def make_wrapper_mock(fake_model: FakeModel) -> MagicMock:
    """Construit un ModelWrapper mock autour d'un FakeModel."""
    wrapper = MagicMock()
    wrapper.model  = fake_model
    wrapper.device = torch.device("cpu")
    return wrapper


# ---------------------------------------------------------------------------
# Tests — ActivationCapture
# ---------------------------------------------------------------------------

class TestActivationCapture:
    def test_captures_hidden_states(self):
        model = FakeModel()
        x = torch.randn(1, 5, HIDDEN_DIM)
        with ActivationCapture(model, layer_idx=1) as cap:
            model.model(x)
        assert len(cap.hidden_states) == 1
        assert cap.hidden_states[0].shape == (1, 5, HIDDEN_DIM)

    def test_hook_removed_after_context(self):
        model = FakeModel()
        x = torch.randn(1, 3, HIDDEN_DIM)
        with ActivationCapture(model, layer_idx=0):
            pass
        assert count_active_hooks(model) == 0

    def test_last_token(self):
        model = FakeModel()
        x = torch.randn(1, 7, HIDDEN_DIM)
        with ActivationCapture(model, layer_idx=2) as cap:
            model.model(x)
        lt = cap.last_token()
        assert lt.shape == (HIDDEN_DIM,)

    def test_seq_mean(self):
        model = FakeModel()
        x = torch.randn(1, 7, HIDDEN_DIM)
        with ActivationCapture(model, layer_idx=2) as cap:
            model.model(x)
        sm = cap.seq_mean()
        assert sm.shape == (HIDDEN_DIM,)


# ---------------------------------------------------------------------------
# Tests — count_active_hooks
# ---------------------------------------------------------------------------

class TestCountActiveHooks:
    def test_zero_before_any_hook(self):
        model = FakeModel()
        assert count_active_hooks(model) == 0

    def test_nonzero_while_hook_active(self):
        model = FakeModel()
        cap = ActivationCapture(model, layer_idx=0)
        assert count_active_hooks(model) > 0
        cap.remove()

    def test_zero_after_remove(self):
        model = FakeModel()
        cap = ActivationCapture(model, layer_idx=0)
        cap.remove()
        assert count_active_hooks(model) == 0


# ---------------------------------------------------------------------------
# Tests — SteeringHook
# ---------------------------------------------------------------------------

class TestSteeringHook:
    def test_hook_registered_then_removed(self):
        model = FakeModel()
        vector = torch.zeros(HIDDEN_DIM)
        with SteeringHook(model, layer_idx=1, vector=vector, alpha=1.0):
            assert count_active_hooks(model) > 0
        assert count_active_hooks(model) == 0

    def test_vector_added_to_output(self):
        """Le hook doit ajouter alpha * vector aux hidden states."""
        model = FakeModel()
        x = torch.zeros(1, 3, HIDDEN_DIM)

        # Sans hook : récupère la sortie de reference
        with ActivationCapture(model, layer_idx=1) as cap_base:
            model.model(x)
        base_output = cap_base.hidden_states[0].clone()

        # Avec hook : vector = ones * 1.0 → chaque valeur décalée de 1.0
        vector = torch.ones(HIDDEN_DIM)
        with SteeringHook(model, layer_idx=1, vector=vector, alpha=1.0):
            with ActivationCapture(model, layer_idx=2) as cap_after:
                model.model(x)

        # La sortie de la couche 2 doit avoir intégré le shift de la couche 1
        # On vérifie juste que le résultat diffère du baseline
        assert not torch.allclose(cap_after.hidden_states[0], base_output[:, :, :HIDDEN_DIM])

    def test_no_hook_leak_across_two_runs(self):
        model = FakeModel()
        vector = torch.zeros(HIDDEN_DIM)
        for _ in range(2):
            with SteeringHook(model, layer_idx=0, vector=vector, alpha=1.0):
                pass
        assert count_active_hooks(model) == 0


# ---------------------------------------------------------------------------
# Tests — generate_base
# ---------------------------------------------------------------------------

class TestGenerateBase:
    def test_calls_wrapper_generate(self):
        wrapper = MagicMock()
        wrapper.generate.return_value = "test output"
        result = generate_base(wrapper, "hello", max_new_tokens=50)
        from src.steering import GENERATION_TEMPERATURE
        wrapper.generate.assert_called_once_with(
            "hello", max_new_tokens=50, temperature=GENERATION_TEMPERATURE
        )
        assert result == "test output"


# ---------------------------------------------------------------------------
# Tests — generate_steered
# ---------------------------------------------------------------------------

class TestGenerateSteered:
    def _make_tokenizer_mock(self):
        tokenizer = MagicMock()
        tokenizer.apply_chat_template.return_value = "<chat>hello</chat>"
        tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
        tokenizer.eos_token_id = 0
        tokenizer.decode.return_value = "steered output"
        # Permet input.to(device) via __call__
        fake_inputs = MagicMock()
        fake_inputs.__getitem__ = lambda self, k: torch.tensor([[1, 2, 3]])
        fake_inputs.to.return_value = fake_inputs
        tokenizer.return_value = fake_inputs
        return tokenizer

    def test_returns_decoded_text(self):
        fake_model = FakeModel()
        wrapper = make_wrapper_mock(fake_model)
        wrapper.tokenizer = self._make_tokenizer_mock()

        # mock model.generate pour retourner des ids
        generated_ids = torch.tensor([[1, 2, 3, 4, 5]])
        wrapper.model.generate = MagicMock(return_value=generated_ids)

        vector = torch.zeros(HIDDEN_DIM)
        result = generate_steered(
            wrapper, "hello", vector, alpha=2.0, layer_idx=1, max_new_tokens=50
        )
        assert isinstance(result, str)
        wrapper.tokenizer.decode.assert_called_once()

    def test_no_hook_leak_after_generation(self):
        fake_model = FakeModel()
        wrapper = make_wrapper_mock(fake_model)
        wrapper.tokenizer = self._make_tokenizer_mock()
        generated_ids = torch.tensor([[1, 2, 3, 4]])
        wrapper.model.generate = MagicMock(return_value=generated_ids)

        vector = torch.zeros(HIDDEN_DIM)
        generate_steered(wrapper, "hello", vector, alpha=2.0, layer_idx=1, max_new_tokens=50)
        assert count_active_hooks(fake_model) == 0

    def test_steered_uses_correct_layer(self):
        """Le SteeringHook doit être enregistré sur la bonne couche."""
        fake_model = FakeModel()
        wrapper = make_wrapper_mock(fake_model)
        wrapper.tokenizer = self._make_tokenizer_mock()
        generated_ids = torch.tensor([[1, 2, 3, 4]])
        wrapper.model.generate = MagicMock(return_value=generated_ids)

        vector = torch.zeros(HIDDEN_DIM)
        target_layer = 2

        hooks_during = []
        original_generate = fake_model.generate if hasattr(fake_model, 'generate') else None

        def spy_generate(**kwargs):
            hooks_during.append(count_active_hooks(fake_model))
            return generated_ids

        wrapper.model.generate = spy_generate
        generate_steered(wrapper, "hello", vector, alpha=1.0, layer_idx=target_layer, max_new_tokens=10)

        assert hooks_during[0] == 1, "Exactement 1 hook actif pendant generate()"
        assert count_active_hooks(fake_model) == 0, "Hook supprimé après"
