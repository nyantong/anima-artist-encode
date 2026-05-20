A drop-in CLIP text encoder for ComfyUI that mixes multiple artist tags from a single prompt. Built for the Anima model family.

## What it does

ComfyUI's native `CLIPTextEncode` applies token-weight extrapolation per token, but with multiple artist tags in one prompt those tokens interact during a single encoding pass and collapse into a virtual artist — most of the secondary artist signal is lost.

This node isolates each artist's contribution by encoding the prompt N additional times (once per artist, with all others removed), then sums the resulting directions on top of the native pass. Native is preserved bit-for-bit when slider strength is zero.

## Features

- Single-text, single-node interface — write `@name`, `@(name:weight)`, or `(@name:weight)` inline.
- Four modes: `native`, `slider_boost`, `slider_dare`, `slider_ties`.
- Concept slider (independent positive/negative text axis).
- Negative weights supported.
- No silent clamping, no automatic token reordering.
- Configurable attention-mask extension.

## Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/nyantong/anima-artist-encode
```

Restart ComfyUI. The node appears under `conditioning/anima`.

## Quick Start

1. Connect `CheckpointLoader → CLIP` to the node's `clip` input.
2. Write a prompt with artist tags:
    
    ```
    masterpiece, 1girl, @(arist_name:1.0), @(arist_name:0.5), long hair, white sweater
    ```
    
3. Connect `conditioning` to `KSampler.positive`.
4. Default mode is `slider_boost` with `boost_strength=1.0` — mixing is active out of the box.

## How it works

### Artist tag syntax

Four equivalent forms:

- `@name` (weight = 1.0)
- `@(name)` (weight = 1.0)
- `@(name:0.5)` (weight = 0.5)
- `(@name:0.5)` (weight = 0.5, native syntax)

Weights may be negative or zero. Tag position is preserved as written.

### Native pass

The node converts every `@(name:w)` to `(@name:w)` and calls `clip.tokenize` + `encode_from_tokens_scheduled` exactly once. Mathematically identical to ComfyUI's `CLIPTextEncode`:

$$
z_{out}[i][j] = (z[i][j] - z_{empty}[j]) \cdot w_i + z_{empty}[j]
$$

where $z_{empty}$ is the encoding of the empty prompt.

### Slider direction

For each artist $i$ the node builds two additional texts:

- $\text{bare}$: the prompt with all artist tags removed.
- $\text{single}_i$: the prompt with only artist $i$ kept (weight = 1.0).

Then:

$$
d_i = \text{encode}(\text{single}_i) - \text{encode}(\text{bare})
$$

This is the Concept Sliders textual variant (arXiv 2311.12092) applied per artist token. Equivalent to ADV_CLIP_emb's `from_masked` mode.

### Combining

$$
\text{out} = \text{native} + s \cdot \sum_i w_i \cdot d_i + s_c \cdot d_{\text{concept}}
$$

where $s$ is `boost_strength`, $w_i$ is each artist's prompt weight, and $d_{\text{concept}}$ is the optional concept slider direction.

When `boost_strength = 0` and concept is empty, output equals native bit-for-bit.

## Modes

| Mode | Direction transform | Use case |
| --- | --- | --- |
| `native` | None — single pass only | Match `CLIPTextEncode` exactly |
| `slider_boost` | Sum directions as-is | Strongest mix; may produce artifacts when artists clash |
| `slider_dare` | Random drop + rescale | Reduce cross-artist interference (varies per seed) |
| `slider_ties` | Trim + sign election | Cleaner output with many artists |

### slider_dare

DARE (arXiv 2311.03099):

$$
m_i \sim \text{Bernoulli}(1-p), \quad \tilde{d}_i = (d_i \odot m_i) / (1-p)
$$

Drops a fraction $p$ of dimensions per direction, rescales survivors to preserve expected magnitude.

### slider_ties

TIES (arXiv 2306.01708), applied to directions only — per-artist weights are preserved:

1. **Trim**: keep top $\rho$-fraction of dimensions per direction by magnitude.
2. **Elect Sign**: dominant sign per dimension is $\text{sign}\left(\sum_i |d_i| \cdot \text{sign}(d_i)\right)$.
3. **Disjoint Mask**: keep only dimensions whose sign matches the dominant sign.

$$
\tilde{d}_i = d_i \odot \mathbb{1}[\text{sign}(d_i) = \text{sign}_{\text{dom}}] \odot \mathbb{1}[\text{sign}_{\text{dom}} \neq 0]
$$

## Parameters

| Name | Type | Default | Range | Active in |
| --- | --- | --- | --- | --- |
| clip | CLIP | — | — | always |
| text | STRING | "" | — | always |
| mode | ENUM | slider_boost | — | always |
| boost_strength | FLOAT | 1.0 | −10~10 | slider_* |
| dare_drop | FLOAT | 0.5 | 0~0.99 | slider_dare |
| ties_density | FLOAT | 0.2 | 0.01~1 | slider_ties |
| concept_positive | STRING | "" | — | slider_* |
| concept_negative | STRING | "" | — | slider_* |
| concept_strength | FLOAT | 1.0 | −10~10 | slider_* |
| extend_attention | BOOLEAN | False | — | slider_* |

## Concept Slider

Independent of artist mixing. Provide free-text `concept_positive` and/or `concept_negative`:

$$
d_{\text{concept}} = \text{encode}(\text{positive}) - \text{encode}(\text{negative})
$$

Added to output with `concept_strength`. Either side may be empty (empty prompt is used as anchor). Auto-disabled when both are empty.

Example:

```
concept_positive: "chibi, cute, simplified"
concept_negative: "realistic, photographic"
concept_strength: 0.8
```

## Difference from native CLIPTextEncode

`CLIPTextEncode` applies token-weight extrapolation per artist token, but all artist tokens share a single encoding pass. CLIP's causal self-attention causes adjacent artist tokens to interact during that pass, producing a single mixed style rather than a weighted combination. Empirically (Anima model card, community reports): two artist tags in one prompt collapse to the dominant artist with most of the second artist's signal lost regardless of weight.

This node keeps the native pass intact and adds a slider correction that isolates each artist's pure contribution. The native pass alone matches `CLIPTextEncode` bit-for-bit; the slider is purely additive on top.

## Output

- `conditioning` — connect to `KSampler.positive`.
- `debug_info` (STRING) — detected artists with weights, active mode, truncated native text preview.

## Performance

`slider_*` modes run N+1 encoding passes (N artists + bare), plus 2 more when the concept slider is active. Roughly 3~5× slower than native. For animation or large batches, cache the conditioning once and reuse.

## References

- **Concept Sliders** — Gandikota et al., *arXiv 2311.12092*. Latent direction extraction from text pairs.
- **DARE** — Yu et al., *arXiv 2311.03099*. Drop And REscale for parameter merging.
- **TIES-Merging** — Yadav et al., *arXiv 2306.01708*. Trim, Elect Sign, Disjoint Merge.
- **ComfyUI ADV_CLIP_emb** — BlenderNeko. `from_masked` reference implementation for slider direction extraction.
- **Anima Mod Guidance** — Anzhc. Companion sampler-level guidance node for the Anima family.

## Limitations

- `slider_ties` has no effect with a single artist (sign election needs ≥2 directions).
- Long concept texts blur the concept direction; keep concept fields focused on the axis you want to control.
- `extend_attention=True` shifts the attention distribution and may push the model out of distribution; leave off unless slider effect is too weak.
- Pooled output skips TIES (sign election is ill-defined for 1D tensors).
