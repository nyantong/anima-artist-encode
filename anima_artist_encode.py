import re

import torch


# Parsing
ARTIST_PATTERNS = [
	re.compile(r"@\(((?:[^()\\]|\\\(|\\\))+?):(-?\d+(?:\.\d+)?)\)"),   # @(name:weight)
	re.compile(r"@\(((?:[^()\\]|\\\(|\\\))+?)\)"),                       # @(name)
	re.compile(r"\(@((?:[^()\\]|\\\(|\\\))+?):(-?\d+(?:\.\d+)?)\)"),    # (@name:weight)
	re.compile(r"@([\w\u4e00-\u9fff\uac00-\ud7af\-]+)"),                  # @name
]


def parse_artists(text):
	spans = []
	consumed = [False] * len(text)
	for pattern in ARTIST_PATTERNS:
		for m in pattern.finditer(text):
			if any(consumed[m.start():m.end()]):
				continue
			groups = m.groups()
			name = groups[0].strip()
			weight = float(groups[1]) if len(groups) > 1 and groups[1] is not None else 1.0
			spans.append((m.start(), m.end(), name, weight))
			for i in range(m.start(), m.end()):
				consumed[i] = True
	spans.sort(key=lambda x: x[0])
	return spans


def cleanup_text(text):
	text = re.sub(r"(?:\s*,\s*){2,}", ", ", text)
	text = re.sub(r"^\s*,\s*", "", text)
	text = re.sub(r"\s*,\s*$", "", text)
	text = re.sub(r"[ \t]+", " ", text)
	return text.strip()


def escape_for_native(name):
	out = []
	i = 0
	while i < len(name):
		ch = name[i]
		if ch == "\\" and i + 1 < len(name) and name[i + 1] in "()":
			out.append(name[i:i + 2])
			i += 2
			continue
		if ch in "()":
			out.append("\\" + ch)
		else:
			out.append(ch)
		i += 1
	return "".join(out)


def artist_token_str(name, weight):
	safe = escape_for_native(name)
	core = f"@{safe}"
	if weight == 1.0:
		return core
	return f"({core}:{weight:g})"


def to_native_text(text, spans):
	if not spans:
		return text
	parts = []
	last = 0
	for s, e, name, w in spans:
		parts.append(text[last:s])
		parts.append(artist_token_str(name, w))
		last = e
	parts.append(text[last:])
	return "".join(parts)


def remove_artists(text, spans):
	parts = []
	last = 0
	for s, e, _, _ in spans:
		parts.append(text[last:s])
		last = e
	parts.append(text[last:])
	return cleanup_text("".join(parts))


def single_artist_text(text, spans, keep_idx):
	parts = []
	last = 0
	for idx, (s, e, name, _) in enumerate(spans):
		parts.append(text[last:s])
		if idx == keep_idx:
			parts.append(artist_token_str(name, 1.0))
		last = e
	parts.append(text[last:])
	return cleanup_text("".join(parts))


# === Encoding helpers ===
def native_encode(clip, text):
	tokens = clip.tokenize(text)
	return clip.encode_from_tokens_scheduled(tokens)


def pad_seq(tensor, target_len):
	cur = tensor.shape[1]
	if cur == target_len:
		return tensor
	if cur > target_len:
		return tensor[:, :target_len]
	pad_shape = list(tensor.shape)
	pad_shape[1] = target_len - cur
	pad = torch.zeros(*pad_shape, dtype=tensor.dtype, device=tensor.device)
	return torch.cat([tensor, pad], dim=1)


def dare_drop_rescale(delta, p):
	if p <= 0.0:
		return delta
	if p >= 1.0:
		return torch.zeros_like(delta)
	mask = (torch.rand_like(delta) > p).to(delta.dtype)
	return (delta * mask) / (1.0 - p)


def ties_align(directions, density):
	"""TIES alignment: trim + sign election. Mask-only — preserves per-artist weights."""
	if not directions or density >= 1.0:
		return list(directions)

	# Step 1: Trim
	trimmed = []
	for d in directions:
		flat_abs = d.abs().flatten()
		n = flat_abs.numel()
		k = max(1, int(n * density))
		if k >= n:
			trimmed.append(d)
			continue
		threshold = torch.kthvalue(flat_abs, n - k + 1).values
		mask = (d.abs() >= threshold).to(d.dtype)
		trimmed.append(d * mask)

	# Step 2: Elect dominant sign
	stack = torch.stack(trimmed, dim=0)
	sign_sum = (stack.abs() * stack.sign()).sum(dim=0)
	dominant_sign = sign_sum.sign()
	nonzero_dom = (dominant_sign != 0).to(stack.dtype)

	# Step 3: Apply align mask per direction
	aligned = []
	for d in trimmed:
		align_mask = (d.sign() == dominant_sign).to(d.dtype)
		aligned.append(d * align_mask * nonzero_dom)
	return aligned


# Main node
class AnimaArtistEncode:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"clip": ("CLIP",),
				"text": ("STRING", {
					"multiline": True,
					"tooltip": "Prompt. Artists: @name, @(name), @(name:weight), or (@name:weight). Weight may be negative or zero. Artist position is preserved as-is — no auto-reordering.",
				}),
				"mode": (["native", "slider_boost", "slider_dare", "slider_ties"], {
					"default": "slider_boost",
					"tooltip": "native: ComfyUI default token-weight extrapolation, single pass. slider_boost: extract each artist's direction independently and sum — strongest effect, preserves each artist's signature, but artifact-prone when artists clash. slider_dare: randomly drop 50% of each direction and rescale survivors — effect varies per seed. slider_ties: keep top-magnitude dims per direction, then only dims where artists agree on sign — cleaner with many artists.",
				}),
				"boost_strength": ("FLOAT", {
					"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.05,
					"tooltip": "Slider direction magnitude. 0 = slider off (native only). 0.5 = half strength, subtle blend. 1.0 = full direction (default). 2.0+ = exaggerated, artifacts likely. Negative flips every artist's direction.",
				}),
				"dare_drop": ("FLOAT", {
					"default": 0.5, "min": 0.0, "max": 0.99, "step": 0.01,
					"tooltip": "Fraction of direction dims randomly zeroed; survivors rescaled by 1/(1-p) so the expected magnitude stays constant. 0 = no drop (= slider_boost). 0.5 = drop half, survivors x2 (default). 0.9 = drop 90%, survivors x10, sparse emphasis.",
				}),
				"ties_density": ("FLOAT", {
					"default": 0.2, "min": 0.01, "max": 1.0, "step": 0.01,
					"tooltip": "Top-magnitude dim fraction kept per direction (Trim step). 0.2 = keep top 20% (paper default). 1.0 = no trimming. After trim, only dims where the majority of artists share the same sign survive.",
				}),
				"concept_positive": ("STRING", {
					"multiline": True, "default": "",
					"tooltip": "Optional concept slider positive text. Example: 'chibi, cute, simplified'. Leave empty to skip.",
				}),
				"concept_negative": ("STRING", {
					"multiline": True, "default": "",
					"tooltip": "Optional concept slider negative text. Example: 'realistic, photographic'. Leave empty to use empty prompt as anchor.",
				}),
				"concept_strength": ("FLOAT", {
					"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.05,
					"tooltip": "Concept slider magnitude. 0 = off. Positive pulls toward positive text, negative toward negative text. Auto-off when both concept texts are empty.",
				}),
				"extend_attention": ("BOOLEAN", {
					"default": False,
					"tooltip": "OFF: slider's extended sequence positions are masked out (model ignores them; slider effect may weaken). ON: full effect at the cost of a shifted attention distribution.",
				}),
			}
		}

	RETURN_TYPES = ("CONDITIONING", "STRING")
	RETURN_NAMES = ("conditioning", "debug_info")
	FUNCTION = "process"
	CATEGORY = "conditioning/anima"

	def process(self, clip, text, mode, boost_strength, dare_drop, ties_density,
			concept_positive, concept_negative, concept_strength, extend_attention):
		spans = parse_artists(text)

		# Pass A: native encoding
		native_text = to_native_text(text, spans)
		native_cond_list = native_encode(clip, native_text)

		artist_slider_active = (mode != "native" and spans and boost_strength != 0.0)
		concept_slider_active = (mode != "native" and concept_strength != 0.0 and
			(concept_positive.strip() or concept_negative.strip()))

		if not artist_slider_active and not concept_slider_active:
			return (native_cond_list, self._debug(spans, mode, boost_strength, dare_drop,
				ties_density, concept_strength, native_text, applied=False))

		# Pass B: slider directions
		base_entry = native_cond_list[0]
		base_cond = base_entry[0]
		base_ctx = dict(base_entry[1])
		base_pooled = base_ctx.get("pooled_output", None)
		base_mask = base_ctx.get("attention_mask", None)

		directions = []
		pooled_directions = []
		weights = []

		if artist_slider_active:
			bare_text = remove_artists(text, spans)
			bare_list = native_encode(clip, bare_text)
			bare_cond = bare_list[0][0]
			bare_pooled = bare_list[0][1].get("pooled_output", None)

			for i in range(len(spans)):
				w = spans[i][3]
				if w == 0.0:
					continue
				single_text = single_artist_text(text, spans, i)
				single_list = native_encode(clip, single_text)
				single_cond = single_list[0][0]
				single_pooled = single_list[0][1].get("pooled_output", None)

				tgt = max(single_cond.shape[1], bare_cond.shape[1])
				directions.append(pad_seq(single_cond, tgt) - pad_seq(bare_cond, tgt))
				weights.append(w)

				if single_pooled is not None and bare_pooled is not None:
					pooled_directions.append(single_pooled - bare_pooled)
				else:
					pooled_directions.append(None)

			if directions:
				dir_max = max(d.shape[1] for d in directions)
				directions = [pad_seq(d, dir_max) for d in directions]

				if mode == "slider_dare":
					directions = [dare_drop_rescale(d, dare_drop) for d in directions]
					pooled_directions = [
						dare_drop_rescale(pd, dare_drop) if pd is not None else None
						for pd in pooled_directions
					]
				elif mode == "slider_ties":
					directions = ties_align(directions, ties_density)
					# Skip TIES on pooled (sign election ill-defined for 1D)

		# Concept slider direction
		concept_direction = None
		concept_pooled_dir = None
		if concept_slider_active:
			pos_text = concept_positive.strip() or ""
			neg_text = concept_negative.strip() or ""
			pos_list = native_encode(clip, pos_text)
			neg_list = native_encode(clip, neg_text)
			pos_cond = pos_list[0][0]
			neg_cond = neg_list[0][0]
			tgt = max(pos_cond.shape[1], neg_cond.shape[1])
			concept_direction = pad_seq(pos_cond, tgt) - pad_seq(neg_cond, tgt)
			pos_pooled = pos_list[0][1].get("pooled_output", None)
			neg_pooled = neg_list[0][1].get("pooled_output", None)
			if pos_pooled is not None and neg_pooled is not None:
				concept_pooled_dir = pos_pooled - neg_pooled

		# Combine
		max_len = base_cond.shape[1]
		if directions:
			max_len = max(max_len, *(d.shape[1] for d in directions))
		if concept_direction is not None:
			max_len = max(max_len, concept_direction.shape[1])

		out_cond = pad_seq(base_cond, max_len).clone()
		for d, w in zip(directions, weights):
			out_cond = out_cond + boost_strength * w * pad_seq(d, max_len)
		if concept_direction is not None:
			out_cond = out_cond + concept_strength * pad_seq(concept_direction, max_len)

		if base_pooled is not None:
			out_pooled = base_pooled.clone()
			for pd, w in zip(pooled_directions, weights):
				if pd is not None:
					out_pooled = out_pooled + boost_strength * w * pd
			if concept_pooled_dir is not None:
				out_pooled = out_pooled + concept_strength * concept_pooled_dir
			base_ctx["pooled_output"] = out_pooled

		if extend_attention and base_mask is not None and max_len > base_cond.shape[1]:
			if base_mask.dim() == 2:
				ext_len = max_len - base_cond.shape[1]
				ext = torch.ones(base_mask.shape[0], ext_len,
					dtype=base_mask.dtype, device=base_mask.device)
				base_ctx["attention_mask"] = torch.cat([base_mask, ext], dim=1)

		out_entry = [out_cond, base_ctx]
		debug = self._debug(spans, mode, boost_strength, dare_drop,
			ties_density, concept_strength, native_text, applied=True)
		return ([out_entry], debug)

	@staticmethod
	def _debug(spans, mode, boost, drop, density, concept_s, native_text, applied):
		artists = " + ".join(f"{n}({w:g})" for _, _, n, w in spans) if spans else "(none)"
		lines = [f"mode={mode}", f"artists={artists}"]
		if applied:
			lines.append(f"boost={boost:g}")
			if mode == "slider_dare":
				lines.append(f"dare_drop={drop:g}")
			elif mode == "slider_ties":
				lines.append(f"ties_density={density:g}")
			if concept_s != 0.0:
				lines.append(f"concept_strength={concept_s:g}")
		else:
			lines.append("slider=inactive")
		preview = native_text if len(native_text) <= 140 else native_text[:140] + "..."
		lines.append(f"native_text={preview}")
		return " | ".join(lines)