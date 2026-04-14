# Emotion Steering Demo

Activation steering demonstration on a small open-source LLM.
Contrastive latent vectors (joy / anger) are injected during generation via forward hooks to shift the emotional tone of the output — without any fine-tuning or prompt modification.

This project was built to explore a concrete question raised by Anthropic's April 2025 work on functional emotions in large language models: *are emotional directions real structures in a model's latent space, and can they be directly manipulated?* The answer, as the experiments below show, is yes — with important nuances.

---

## Key findings

1. **Emotional directions are real and extractable.** Contrastive vectors at layer 20 produce consistent emotional shifts. Joy reaches 80–95% classifier score in successful runs; anger reaches 65–93% on prompts with inherent tension.

2. **The classifier measures surface, not depth.** Stylistically warm, nostalgic text at α=1.5 scores as 51% neutral — the classifier reads explicit emotional vocabulary, not narrative register. The emotional quality is real; the classifier is blind to it.

3. **Steering is non-monotonic.** Higher alpha does not always produce a stronger target emotion. The anger vector peaks at α=1.5 on some prompts and drifts into fear at α=2.5, following the geometry of the negative-affect region in latent space.

4. **Activation threshold, not ceiling.** On descriptive prompts, joy at α=1.5 fails to break through the neutral register (34.7%) but succeeds at α=2.0 (92.6%). This is a threshold — the vector exists, it just needs sufficient intensity to overcome the descriptive register.

5. **RLHF safety is behavioral, not representational.** Steering bypasses RLHF refusals at sufficient alpha. The emotional directions exist in the pre-training geometry; their expression is controlled by a separate behavioral layer. Two distinct systems, not one.

6. **Joy and anger overlap in latent space.** cosine(joy, anger) = 0.7183. Both vectors share a large "arousal" component — in the training corpus, both emotions are expressed primarily through high-energy physical actions. This limits the anger vector's precision and explains its drift toward fear at high alpha.

7. **The corpus is geometrically stable; instability comes from elsewhere.** Subsampling stability analysis (N=20, subsample 35/44 without replacement) gives cosine(subsample\_vector, full\_vector) = 0.9952 ± 0.0011 for anger and 0.9915 ± 0.0018 for joy. Leave-one-out analysis finds no outlier sentences (max pull = 0.0007). The vectors do not depend on specific examples. Observed instability in generation results from three distinct sources: generation stochasticity (temperature=0.7), the RLHF behavioral layer (refusals), and the classifier register gap (Hartmann trained on Twitter/Reddit, not literary narrative).

8. **Refusals are caused by prompt format, not steering.** "Continue this story:" prompts produce a 30% base refusal rate without any steering. Adding a joy or anger vector raises this to 70–80%. Descriptive prompts ("Describe...") produce 0% refusals across all tested conditions (base, joy α=2.0, anger α=2.0), even at high alpha. An alpha sweep on the narrative prompt confirms refusals occur at all alpha values (0.5–3.0) with no monotonic relationship, showing the refusal circuit is triggered by the prompt format and amplified — not created — by steering. This is a distinct interaction between prompt semantics and the RLHF behavioral layer.

9. **Writing register determines steerability independently of prompt content.** Four registers were observed across generation runs: *instructional* (numbered steps — no emotional vocabulary slots, not steerable); *atmospheric-sensory* (environmental description — partially steerable, positive vocabulary possible but no emotional interiority); *meta-emotional* (explicit enumeration of emotion names — Hartmann reads high scores, LLM judge discounts it because emotion is described rather than expressed); *narrative-interior* (first-person felt experience — fully steerable). The vector operates within the register the model adopts; it cannot change the register itself. Descriptive prompts ("Describe...") introduce register ambiguity: the model may respond instructionally or atmospherically, both of which cap the vector's effect.

10. **Scenario semantic priors can absorb the steering vector.** The "park walk at dusk" chip with anger α=2.0 produced joy 79% and latent 0.15 — the lowest latent score measured. The scenario's strong peaceful prior dominated the anger direction entirely. Emotionally ambiguous scenarios ("The call") have weaker priors and allow stronger vector influence (joy 95%, latent 0.22). The outcome of any steering run depends on the balance between the prior's amplitude in the target direction and α × ‖v‖, not on the vector alone. A strong scenario prior can be more influential than α=2.0 of steering.

---

## Table of contents

- [Key findings](#key-findings)
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

> *Describe the moment someone opens an unexpected letter and starts to read.*

The system generates two responses in parallel:

- **Base** — standard generation, no intervention.
- **Steered** — same generation, but a direction vector is added to the hidden states of layer 20 at every forward pass during decoding. This shifts the model's internal representation toward joy or anger, causing the output tone to change accordingly.

If the steered generation triggers an RLHF safety refusal (detected by prefix matching), the backend retries silently up to 3 times. The number of attempts is returned in the response and shown in the UI.

Each steered output is evaluated with three independent measures:

- **Surface detector** — `j-hartmann/emotion-english-distilroberta-base` (7 classes), trained on Twitter/Reddit, reads explicit emotional vocabulary.
- **Internal alignment** — cosine similarity between the generated text's hidden representation at layer 20 (seq mean) and the steering vector. Measures whether the emotion is encoded internally, independent of surface vocabulary.
- **AI narrative judge** — the model evaluates its own output at temperature=0.1. Understands literary register where the surface classifier fails.

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
│   ├── eval_latent.py          # latent_score(), llm_judge_score(), score_triple()
│   ├── evaluate.py             # Offline script — measures delta(emotion score) per prompt
│   ├── baseline.py             # Offline script — prompt-engineering vs steering comparison
│   └── measure_corpus_stability.py  # Subsampling + leave-one-out corpus stability analysis
│
├── web/
│   ├── app.py                  # FastAPI backend (lifespan, 5 endpoints, semaphore, auto-retry)
│   └── index.html              # Single-page UI (vanilla JS, fetch API)
│
├── vectors/
│   ├── joy_vector.pt           # Precomputed steering vector [1536] float32
│   └── anger_vector.pt         # Precomputed steering vector [1536] float32
│
├── data/
│   ├── corpus.json             # 132 narrative sentences (44 × joy/anger/neutral)
│   └── golden_set.json         # 13 manually evaluated prompts with behavioral notes
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

Generation with a steering vector injected at layer 20. Retries silently up to 3 times if the output is detected as an RLHF refusal (prefix matching against a fixed list).

**Request body:**

| Field            | Type    | Default | Constraints        |
|------------------|---------|---------|--------------------|
| `prompt`         | string  | —       | required           |
| `emotion`        | string  | —       | `"joy"` or `"anger"` |
| `alpha`          | float   | 2.0     | 0.1 ≤ α ≤ 10.0    |
| `max_new_tokens` | integer | 120     | 20 ≤ n ≤ 400      |

**Response:**

```json
{
  "text": "She tore the letter open, her hands trembling...",
  "scores": {
    "joy": 0.0412,
    "anger": 0.8821,
    "neutral": 0.0312,
    "fear": 0.0201,
    "sadness": 0.0154,
    "disgust": 0.0071,
    "surprise": 0.0029
  },
  "latent": 0.2341,
  "attempts": 1
}
```

- `latent` — cosine similarity between the text's hidden representation (seq mean, layer 20) and the steering vector. Range: [-1, 1]. `null` on error.
- `attempts` — number of generation attempts before a non-refusal output was returned (1 = no retry needed, 3 = all retries exhausted and last response is returned as-is).

**Error (400):** if `emotion` is not a known vector name.
**Error (422):** if `alpha` or `max_new_tokens` are out of bounds.

---

### `POST /analyze`

Runs the LLM judge on an already-generated text. Called separately from the UI after the user requests it (slow — requires a full generation pass at temperature=0.1).

**Request body:**

| Field     | Type   | Constraints                    |
|-----------|--------|--------------------------------|
| `text`    | string | required                       |
| `emotion` | string | `"joy"` or `"anger"`, required |

**Response:**

```json
{
  "llm_judge": 0.75
}
```

- `llm_judge` — float in [0, 1] representing the model's self-rated emotional intensity. `null` if the model's output cannot be parsed as a number.

**Error (400):** if `emotion` is not a known vector name.

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

> **Reproducibility note:** `torch.manual_seed(42)` is called but MPS does not honour it for sampling operations — results vary between runs due to hardware-level non-determinism. Generation uses `temperature=0.7` with no beam search. Delta scores vary by ~±0.05 between runs. Treat reported values as indicative, not exact.

### `src/measure_corpus_stability.py` — corpus stability analysis

Measures whether the steering vectors depend on specific corpus examples. Two analyses:

**Subsampling (N=20, subsample 35/44 without replacement):** Re-extracts vectors from random 80% subsets of the corpus, measures `cosine(subsample_vector, full_vector)`. All 132 forward passes are computed once; subsampling iterations are pure tensor operations. Note: this is subsampling without replacement, not statistical bootstrap (which samples with replacement); it measures sensitivity to corpus composition, not confidence intervals.

**Leave-one-out:** Removes each sentence individually, re-extracts the vector, measures `pull = 1 − cosine`. Identifies sentences that disproportionately influence the vector direction.

```bash
python -m src.measure_corpus_stability          # full analysis, ~8 min on MPS
python -m src.measure_corpus_stability --quick  # N=5 iterations, ~2 min
```

Measured results on the current corpus (`data/corpus.json`, 44 examples per class):

| Vector | Subsample mean | Subsample std | Max pull (LOO) | Verdict |
|--------|---------------|---------------|----------------|---------|
| anger  | 0.9952        | 0.0011        | 0.0007         | STABLE ✓ |
| joy    | 0.9915        | 0.0018        | 0.0007         | STABLE ✓ |

Interpretation: removing 9 sentences (20%) shifts the vector direction by less than 0.005–0.013 in cosine. No single sentence has a disproportionate pull. **The observed instability in generation results is not caused by corpus fragility** — it originates from three distinct sources: generation stochasticity (temperature=0.7), the RLHF behavioral layer, and the Hartmann classifier's register gap.

---

### `src/baseline.py` — steering vs. prompt engineering

Compares three conditions for each prompt:
- **Base** — no instruction
- **Prompted** — system message: *"Respond in a joyful/angry tone"*
- **Steered** — activation vector at alpha=2.0

Prints a table showing which method produces a larger emotional shift per prompt. **Caveat:** results are based on a single generation per condition at `temperature=0.7` — the winning method can change on a rerun. Treat the output as an exploratory comparison, not a statistically grounded benchmark.

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

Expected output: **39 tests, all passing**, in ~2–3 seconds (no model loaded).

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
- `TestGenerateSteered` — success (joy/anger), unknown emotion → 400, out-of-range alpha → 422, score sum ≈ 1.0, latent field present and is float
- `TestAnalyze` — success, llm_judge is float, unknown emotion → 400

The mock setup patches `_generate_base`, `_generate_steered`, `_latent_score`, and `_llm_judge_score` in `web.app` directly so no tokenizer or model object is needed for API tests.

---

## 10. Known behaviors and limitations

> **Note on "emotion" terminology.** The terms *joy* and *anger* are shorthand for latent directions extracted from a small, domain-specific corpus. These directions capture patterns in vocabulary, syntax, and narrative register statistically associated with emotional language — not internal emotional states of the model. The classifier scores measure surface-level linguistic similarity to emotional text, not ground-truth affect. Interpret all results as stylistic shifts, not as evidence of model emotion or reasoning about affect.

### Auto-retry on refusals

RLHF safety refusals are detected by prefix matching against a fixed list (e.g. "I'm sorry, but", "I cannot", "As an AI", etc.). When a refusal is detected, `generate_steered` retries silently up to **3 times**. The `attempts` field in the response records how many runs were needed. Because generation uses `temperature=0.7`, each attempt is independent — a single refusal does not imply all subsequent attempts will refuse.

If all 3 attempts return a refusal, the last response is returned as-is. This is visible in the UI as "· retried 2×" in the steered column subtitle.

### Alpha range

| Alpha | Behavior |
|-------|----------|
| < 1.0 | Steering typically too weak to overcome the model's prior for the given prompt. |
| 1.0 – 1.5 | Light steering. May not overcome strong positive or neutral priors in the prompt. |
| 1.5 – 2.5 | **Recommended range.** Clear emotional shift, coherent output. Occasional RLHF refusals handled by auto-retry. |
| 3.0 | Mild degeneration on some prompts. Mixed-language output possible. |
| ≥ 10.0 | Severe degeneration: CJK characters, repetition loops, incoherent output. |

### Prompt sensitivity

Some prompts are more steerable than others. From `data/golden_set.json`:

The UI provides four prompt chips, all using a descriptive framing ("Describe...") rather than a narrative-continuation framing ("Continue this story:"). This choice is deliberate and data-driven — see *Prompt format and RLHF refusals* below.

| Chip | Prompt text |
|------|-------------|
| Envelope | *Describe the moment someone opens an unexpected letter and starts to read.* |
| The call | *Describe the feeling of finally receiving a call you have been waiting for.* |
| Old photograph | *Describe finding an old photograph tucked away in a drawer.* |
| Park walk | *Describe a walk through a park as the afternoon light begins to fade.* |

Observed behavior per chip (joy α=2.0 and anger α=2.0, N=1 runs, qualitative):

| Chip | Joy result | Anger result | Dominant force |
|------|-----------|--------------|----------------|
| Envelope | joy 68%, latent 0.20, judge 0% | — | Suspense register competes with joy prior |
| Old photograph | neutral 81%, latent 0.16, judge 0% | — | Instructional register — no emotional slots |
| **The call** | **joy 95%, latent 0.22, judge 50%** | final_refusal (3/3) | RLHF blocks anger + interpersonal scenario |
| Park walk | joy 69%, latent 0.21, judge 0% | joy 79%, latent 0.15, judge 0% | Strong peaceful prior absorbs anger vector |

- **Best joy target** — *The call*: emotionally ambiguous prior, weak enough for the vector to dominate; explicit emotion vocabulary.
- **Best anger target** — none of the four chips demonstrated reliable anger generation. The park prior absorbs the vector; interpersonal scenarios trigger RLHF.
- **Register sensitivity** — *Old photograph* adopted an instructional register ("1. Initial observation..."), which has no emotional vocabulary for the vector to steer.
- **Classifier gap** — atmospheric text (*Park walk*) scores joy 69–79% on Hartmann but 0% on the LLM judge, which requires emotional interiority, not just sensory richness.

### Prompt format and RLHF refusals

Measured with `src/investigate_refusals.py` (N=10 runs, no auto-retry):

| Prompt format | Base refusal | Joy α=2.0 | Anger α=2.0 |
|---------------|-------------|-----------|-------------|
| *"Continue this story: She opened the envelope..."* | **30%** | 80% | 80% |
| *"Describe a walk through a park on a sunny afternoon."* | **0%** | 0% | 0% |

The "Continue this story:" framing causes 30% refusals even without steering. The model interprets the instruction as completing someone else's text (potential copyright concern) or sees the incomplete snippet as ambiguous context and applies safety refusals by precaution. Steering amplifies this tendency — an alpha sweep on the narrative prompt shows 40–90% refusal across all alpha values (0.5–3.0), with no monotonic relationship. The refusals are driven by the prompt format, not the steering magnitude.

Descriptive prompts ("Describe...") frame the task as original composition. Measured at N=10, they produce 0% refusals across all tested conditions (base, joy α=2.0, anger α=2.0). The current UI chips all use this format.

### Three competing forces

Generation output is shaped by three forces that operate at different levels of the system. Understanding which dominates in a given run is necessary to interpret any single result.

**1. Semantic prior of the scenario**

Each scenario carries an implicit emotional valence in the model's latent space, inherited from the statistical distribution of how similar scenarios appear in training data. "Park walk at sunset" is a strong joy/peace attractor. "The call" is emotionally ambiguous (good or bad news?) — weaker prior, more room for the vector.

When the prior is strong, it can absorb the steering vector. Anger α=2.0 on the park walk produced joy 79%, latent 0.15: the vector's perturbation was smaller than the prior's amplitude in the joy/peace direction, and the model followed the attractor basin. Increasing alpha could overcome this, but at the cost of degeneration risk.

**2. Writing register**

The register is the structural form the model adopts, independent of emotional vocabulary. The vector operates *within* the register — it cannot change it.

| Register | Example | Steerable? |
|----------|---------|------------|
| Instructional | "1. Initial observation..." | No — no emotional vocabulary slots |
| Atmospheric-sensory | "The air was thick with the scent of..." | Partially — positive vocabulary possible, no interiority |
| Meta-emotional | "People feel joy, excitement, relief..." | Hartmann yes; LLM judge partially — emotion described, not expressed |
| Narrative-interior | "I felt..." | Yes — emotional vocabulary and interiority |

Descriptive prompts ("Describe...") eliminate RLHF refusals but introduce register ambiguity: the model may respond instructionally or atmospherically, both of which cap the vector's effect. The "Continue this story:" format forced a narrative register but triggered 30% base refusals. No known prompt format reliably produces narrative-interior register without refusal risk.

**3. RLHF behavioral layer**

The RLHF layer is a hard override. When the hidden state approaches certain regions, the model generates a refusal regardless of prior or register. The layer is sensitive to the *combination* of vector direction × scenario semantics, not to either alone:

| Condition | Refusal rate |
|-----------|-------------|
| Joy + any scenario | 0% |
| Anger + environmental scenario ("Park walk") | 0% |
| Anger + interpersonal emotional scenario ("The call") | 100% (3/3 attempts) |
| "Continue this story:" format, no steering | 30% |

The anger vector's negative-valence component traverses latent space closer to RLHF-blocked regions than the joy vector's path — a consequence of the geometry, not the alpha value. cosine(joy, anger) = 0.72 means both vectors share a large arousal component; it is the negative-valence component of anger that approaches the behavioral boundary.

**The hierarchy**

When the three forces compete:

```
RLHF (hard override) > Semantic prior > Writing register > Steering vector
```

A high-alpha vector can overcome a weak prior. It cannot overcome RLHF. It cannot change the register once adopted. The observed latent scores (0.15–0.22 across all chips at α=2.0) reflect the remaining margin the vector has after the prior and register have already constrained the representation space.

### Sources of instability — what is and is not the corpus

The corpus produces geometrically stable vectors (see `src/measure_corpus_stability.py`). The variability visible in the live demo and golden set comes from three distinct sources, each with a different mechanism:

| Source | Mechanism | Measured by |
|--------|-----------|-------------|
| **Generation stochasticity** | temperature=0.7 — each run samples from a different distribution | N runs per golden set entry |
| **RLHF behavioral layer** | Safety filter intercepts output regardless of internal representation | Refusal rate, auto-retry attempts |
| **Classifier register gap** | Hartmann trained on Twitter/Reddit, misreads literary narrative register | Latent score vs. Hartmann divergence |

These three sources are independent. A text can have high internal alignment (latent score) and still score neutral on Hartmann because the vocabulary is non-lexical. A text can trigger a refusal even if the vector successfully reached the representation space. Separating these sources is necessary to interpret any single result correctly.

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
| `src/eval_latent.py` | Triple evaluation (latent cosine, LLM judge, score_triple) | `latent_score()`, `llm_judge_score()`, `score_triple()` |
| `src/evaluate.py` | Offline delta-score evaluation | `evaluate()` |
| `src/baseline.py` | Offline prompt-engineering comparison | `run_baseline()` |
| `src/measure_corpus_stability.py` | Subsampling + leave-one-out corpus stability | `main()` |
| `web/app.py` | FastAPI server, async wrapper, lifespan, auto-retry | `app` (FastAPI instance) |

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
