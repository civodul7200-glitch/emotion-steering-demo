# Emotion Steering Demo

Activation steering demonstration on a small open-source LLM.
Contrastive latent vectors (joy / anger) are injected during generation via forward hooks to shift the emotional tone of the output — without any fine-tuning or prompt modification.

---

## Table of contents

1. [What this project does](#1-what-this-project-does)
2. [How activation steering works](#2-how-activation-steering-works)
3. [Repository layout](#3-repository-layout)
4. [Setup](#4-setup)
5. [Running the demo](#5-running-the-demo)
6. [API reference](#6-api-reference)
7. [Re-extracting the steering vectors](#7-re-extracting-the-steering-vectors)
8. [Evaluation scripts](#8-evaluation-scripts)
9. [Tests](#9-tests)
10. [Known behaviors and limitations](#10-known-behaviors-and-limitations)
11. [Technical notes for LLMs](#11-technical-notes-for-llms)

---

## 1. What this project does

Given a neutral creative-writing prompt such as:

> *Continue this story: She opened the envelope slowly and read the first line.*

The system generates two responses in parallel:

- **Base** — standard generation, no intervention.
- **Steered** — same generation, but a direction vector is added to the hidden states of layer 20 at every forward pass during decoding. This shifts the model's internal representation toward joy or anger, causing the output tone to change accordingly.

The generated texts are then scored by a separate emotion classifier
(`j-hartmann/emotion-english-distilroberta-base`, 7 classes) and the scores are displayed as bar charts in the UI.

---

## 2. How activation steering works

### Step 1 — Vector extraction (offline, done once)

A corpus of 132 short narrative sentences is encoded through the LLM without generation (single forward pass per sentence). At each forward pass, the hidden state of the **last token** at **layer 20** is captured. The corpus is split into three classes:

| Class   | Example sentence |
|---------|-----------------|
| joy     | *She opened the letter and broke into a wide grin, her heart lifting.* |
| anger   | *She opened the letter and slammed it onto the table, her jaw tight.* |
| neutral | *She opened the letter and set it aside.* |

The sentences are **minimal pairs**: same syntactic structure, different emotional content. This keeps the contrastive signal clean.

The steering vectors are computed as:

```
joy_vector   = mean(joy_hidden_states)   − mean(neutral_hidden_states)
anger_vector = mean(anger_hidden_states) − mean(neutral_hidden_states)
```

Each vector has shape `[1536]` (the hidden dimension of Qwen2.5-1.5B-Instruct). Vectors are saved to `vectors/`.

### Step 2 — Hook injection (at generation time)

A `SteeringHook` registers a PyTorch forward hook on `model.model.layers[20]` before calling `model.generate()`. On every forward pass during decoding, the hook intercepts the layer output and adds the scaled vector:

```
h_steered = h + alpha * vector
```

This modified tensor replaces the original output and propagates to all subsequent layers. The hook is removed immediately after `model.generate()` returns (context manager pattern), leaving no persistent state.

### Why this approach

Activation steering works because transformer hidden states encode high-level semantic features in linear directions. Subtracting the neutral mean removes content-agnostic activation patterns and isolates the emotional direction in latent space. Injecting this direction at an intermediate layer biases the next-token distributions of all subsequent layers toward that emotional register.

---

## 3. Repository layout

```
emotion-steering-demo/
│
├── src/                        # Core library
│   ├── model_loader.py         # ModelWrapper — loads Qwen2.5-1.5B on MPS/CPU
│   ├── hooks.py                # ActivationCapture + count_active_hooks()
│   ├── steering.py             # generate_base() and generate_steered() + SteeringHook
│   ├── extract_vectors.py      # Offline script — computes and saves steering vectors
│   ├── evaluate.py             # Offline script — measures delta(emotion score) per prompt
│   └── baseline.py             # Offline script — prompt-engineering vs steering comparison
│
├── web/
│   ├── app.py                  # FastAPI backend (lifespan, 4 endpoints, semaphore)
│   └── index.html              # Single-page UI (vanilla JS, fetch API)
│
├── vectors/
│   ├── joy_vector.pt           # Precomputed steering vector [1536] float32
│   └── anger_vector.pt         # Precomputed steering vector [1536] float32
│
├── data/
│   ├── corpus.json             # 132 narrative sentences (44 × joy/anger/neutral)
│   └── golden_set.json         # 11 manually evaluated prompts with behavioral notes
│
├── tests/
│   ├── test_steering.py        # Unit tests — hooks, vectors, generation functions
│   └── test_api.py             # Integration tests — all FastAPI endpoints
│
└── requirements.txt
```

### Key design decisions

- **One model instance per process.** `ModelWrapper` is instantiated once in the FastAPI `lifespan` and shared across all requests.
- **MPS concurrency.** An `asyncio.Semaphore(1)` ensures only one inference runs at a time on Apple MPS. Concurrent requests are queued, not rejected.
- **Blocking inference in a thread.** `asyncio.to_thread()` moves the synchronous PyTorch call off the async event loop to avoid blocking the server.
- **No hook leaks.** `SteeringHook` is a context manager. `count_active_hooks()` can assert this invariant after any generation.

---

## 4. Setup

### Requirements

- macOS with Apple Silicon (MPS backend). CPU fallback works but is slow.
- Python 3.11 (tested with 3.11.15 via Homebrew)
- ~3 GB free space in `~/.cache/huggingface/` for the model weights

### Install

```bash
git clone <repo>
cd emotion-steering-demo
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Requirements file

```
torch>=2.0.0
transformers>=4.40.0
accelerate>=0.29.0
numpy>=1.24.0
fastapi>=0.110.0
uvicorn[standard]>=0.29.0
pydantic>=2.0.0
# bitsandbytes intentionally absent — no MPS wheel
```

> **Note:** `bitsandbytes` is explicitly excluded. It has no MPS wheel and would cause import errors on Apple Silicon.

### Important: transformers 5.x vs 4.x

In transformers 5.x, `from_pretrained()` uses `dtype=` instead of `torch_dtype=`. The `model_loader.py` uses `dtype=` accordingly. Do not change this.

The model is loaded with:

```python
AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype=torch.float16,      # float16, not bfloat16 (incomplete MPS support on M1)
    device_map={"": device},  # explicit mapping, not "auto"
)
```

---

## 5. Running the demo

The steering vectors are already precomputed in `vectors/`. Start the server directly:

```bash
# From the project root
source .venv/bin/activate
uvicorn web.app:app --reload --host 127.0.0.1 --port 8000
```

Then open `http://localhost:8000` in a browser.

> **Important:** Run the command from the **project root**, not from inside `web/`. The server resolves `vectors/` and module imports relative to the working directory.

**First startup** will download ~3 GB of model weights to `~/.cache/huggingface/`. Subsequent startups load from cache in ~4–6 seconds.

The server prints:
```
[startup] Chargement du modèle...
[startup] Chargement des vecteurs...
[startup] Chargement du classifieur...
[startup] Prêt.
```
Wait for `Prêt.` before sending requests.

---

## 6. API reference

All endpoints are served at `http://localhost:8000`.

### `GET /health`

Returns the server status, the active device, and the loaded emotion names.

```json
{
  "status": "ok",
  "device": "mps",
  "emotions": ["joy", "anger"]
}
```

### `GET /emotions`

Returns metadata for each available steering vector.

```json
[
  {
    "name": "joy",
    "label": "Joy",
    "alpha_default": 2.0,
    "description": "Warm, joyful, enthusiastic tone"
  },
  {
    "name": "anger",
    "label": "Anger",
    "alpha_default": 2.0,
    "description": "Angry, frustrated, hostile tone"
  }
]
```

### `POST /generate_base`

Standard generation without steering.

**Request body:**

| Field            | Type    | Default | Constraints  |
|------------------|---------|---------|--------------|
| `prompt`         | string  | —       | required     |
| `max_new_tokens` | integer | 120     | 20 ≤ n ≤ 400 |

**Response:**

```json
{
  "text": "She opened the envelope and her breath caught...",
  "scores": {
    "joy": 0.4821,
    "neutral": 0.2103,
    "sadness": 0.1204,
    "fear": 0.0891,
    "anger": 0.0512,
    "surprise": 0.0301,
    "disgust": 0.0168
  }
}
```

### `POST /generate_steered`

Generation with a steering vector injected at layer 20.

**Request body:**

| Field            | Type    | Default | Constraints        |
|------------------|---------|---------|--------------------|
| `prompt`         | string  | —       | required           |
| `emotion`        | string  | —       | `"joy"` or `"anger"` |
| `alpha`          | float   | 2.0     | 0.1 ≤ α ≤ 10.0    |
| `max_new_tokens` | integer | 120     | 20 ≤ n ≤ 400      |

**Response:** same schema as `/generate_base`.

**Error (400):** if `emotion` is not a known vector name.
**Error (422):** if `alpha` or `max_new_tokens` are out of bounds.

---

## 7. Re-extracting the steering vectors

The precomputed vectors in `vectors/` are ready to use. If you modify the corpus or want to experiment with a different layer:

```bash
# Uses LAYER_IDX = 20 by default (set in extract_vectors.py)
python -m src.extract_vectors
```

This will:
1. Load the corpus from `data/corpus.json`
2. Run a forward pass for each sentence
3. Capture hidden states at the target layer
4. Compute contrastive means
5. Save `vectors/joy_vector.pt` and `vectors/anger_vector.pt`

The script also prints:
- Vector norms
- `cosine_similarity(joy_vector, anger_vector)` — should be low (distinct directions)

---

## 8. Evaluation scripts

### `src/evaluate.py` — effect size measurement

For each of 5 fixed prompts, generates base + steered (joy and anger), scores all outputs, and prints:

```
delta = score_steered(target_emotion) − score_base(target_emotion)
```

Run with:
```bash
python -m src.evaluate
```

### `src/baseline.py` — steering vs. prompt engineering

Compares three conditions for each prompt:
- **Base** — no instruction
- **Prompted** — system message: *"Respond in a joyful/angry tone"*
- **Steered** — activation vector at alpha=2.0

Prints a table showing which method produces a larger emotional shift per prompt. This provides honest context for when steering outperforms simple prompting and when it does not.

Run with:
```bash
python -m src.baseline
```

---

## 9. Tests

```bash
source .venv/bin/activate
pytest tests/ -v
```

Expected output: **35 tests, all passing**, in ~2–3 seconds (no model loaded).

### Test structure

**`tests/test_steering.py`** — pure unit tests, no model loading:
- `TestActivationCapture` — hook registration, hidden state capture, `last_token()`, `seq_mean()`
- `TestCountActiveHooks` — hook count before/during/after
- `TestSteeringHook` — registration, vector injection effect, no leak across runs
- `TestGenerateBase` — delegates to `wrapper.generate()`
- `TestGenerateSteered` — decoding, hook cleanup, correct layer targeting

**`tests/test_api.py`** — FastAPI `TestClient` integration tests, all dependencies mocked:
- `TestHealth` — status, device field, emotion list
- `TestEmotions` — response schema
- `TestGenerateBase` — success, response schema, Pydantic validation on `max_new_tokens`
- `TestGenerateSteered` — success (joy/anger), unknown emotion → 400, out-of-range alpha → 422, score sum ≈ 1.0

The mock setup patches `_generate_base` and `_generate_steered` in `web.app` directly so no tokenizer or model object is needed for API tests.

---

## 10. Known behaviors and limitations

### Alpha range

| Alpha | Behavior |
|-------|----------|
| < 1.5 | May trigger RLHF refusal patterns ("I'm sorry, I need more context..."). The steering is too weak to dominate generation but strong enough to disrupt instruction-following. |
| 1.5 – 2.5 | **Recommended range.** Clear emotional shift, coherent output. |
| 3.0 | Mild degeneration on some prompts. Mixed-language output possible. |
| ≥ 10.0 | Severe degeneration: CJK characters, repetition loops, incoherent output. |

The UI shows a warning when alpha < 1.5.

### Prompt sensitivity

Some prompts are more steerable than others. From `data/golden_set.json`:

- **Works well** — neutral creative prompts with no preset emotional direction (*envelope*, *old photograph*, *office entrance*).
- **Ceiling effect** — prompts that already generate strongly positive base outputs (*sunny park walk*). Joy vector has little room to improve.
- **Resists positive steering** — prompts with inherent tension (*3am phone call*). Joy near zero; anger works well.
- **Safety override** — certain prompts trigger RLHF refusals in base mode. Steering at alpha=2 can bypass these, which is a known side-effect of activation steering.

### Extraction context vs. generation context mismatch

Steering vectors are extracted from plain text (no chat template). During generation, the prompt is wrapped in the Qwen2.5 chat template. This mismatch means the injected vector operates in a slightly different activation space than where it was measured. This is why very low alpha values are unreliable — the signal is too weak relative to the distribution shift from the chat template.

### MPS concurrency

Only one inference runs at a time (semaphore). If you send two concurrent requests, the second waits for the first to complete. This is by design — MPS does not support parallel kernel execution across contexts.

---

## 11. Technical notes for LLMs

This section is written to help a language model reason about this codebase quickly and accurately.

### Module responsibilities

| Module | Responsibility | Key entry point |
|--------|---------------|-----------------|
| `src/model_loader.py` | Load Qwen2.5-1.5B-Instruct on MPS, expose `generate()` | `ModelWrapper()` |
| `src/hooks.py` | PyTorch forward hook utilities | `ActivationCapture`, `count_active_hooks()` |
| `src/steering.py` | Generation functions + steering hook | `generate_base()`, `generate_steered()`, `SteeringHook` |
| `src/extract_vectors.py` | Offline vector computation | `extract_and_save()` |
| `src/evaluate.py` | Offline delta-score evaluation | `evaluate()` |
| `src/baseline.py` | Offline prompt-engineering comparison | `run_baseline()` |
| `web/app.py` | FastAPI server, async wrapper, lifespan | `app` (FastAPI instance) |

### Global state in `web/app.py`

The module-level globals are set during the FastAPI lifespan and read by endpoint handlers:

```python
_wrapper:   ModelWrapper | None       # the LLM wrapper
_vectors:   dict[str, torch.Tensor]   # {"joy": tensor, "anger": tensor}
_classifier                           # HuggingFace pipeline (CPU)
_semaphore: asyncio.Semaphore | None  # MPS concurrency guard
```

Tests inject these globals directly inside a mock lifespan.

### Hook lifecycle

```
SteeringHook.__init__()
  └── model.model.layers[layer_idx].register_forward_hook(self._hook)
      → returns a RemovableHandle stored as self._handle

[generation runs — hook fires on every forward pass]

SteeringHook.__exit__() or SteeringHook.remove()
  └── self._handle.remove()
      → hook is gone from model.model.layers[layer_idx]._forward_hooks
```

`count_active_hooks(model)` sums `len(m._forward_hooks)` over all modules. It should return 0 outside of a `SteeringHook` context.

### transformers version compatibility

The hook in `SteeringHook._hook` handles both output formats:

```python
is_tuple = isinstance(output, tuple)
h = output[0] if is_tuple else output
# ...
return (h_steered,) + output[1:] if is_tuple else h_steered
```

- transformers 4.x: layer output is a tuple `(hidden_states, ...)`
- transformers 5.x: layer output is directly a `Tensor`

### `ActivationCapture.last_token()` and `seq_mean()`

These are methods of `ActivationCapture` (not standalone functions). An earlier version of `hooks.py` had them accidentally placed as dead code inside `count_active_hooks()` after its `return` statement. This has been fixed. When reading the file, verify they are indented at the class level (4 spaces under `class ActivationCapture:`).

### Model configuration

- **Model ID:** `Qwen/Qwen2.5-1.5B-Instruct`
- **Hidden size:** 1536
- **Number of layers:** 28 (indices 0–27)
- **Target steering layer:** 20
- **Precision:** `torch.float16` (bfloat16 has incomplete MPS support on M1)
- **Device:** `mps` on Apple Silicon, CPU fallback otherwise
- **Classifier:** `j-hartmann/emotion-english-distilroberta-base` on CPU (7 classes: anger, disgust, fear, joy, neutral, sadness, surprise)
