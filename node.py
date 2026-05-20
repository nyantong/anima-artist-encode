import torch

from .helpers.parse import (
	parse_artists, to_native_text, remove_artists, single_artist_text, relocate_artists,
)
from .helpers.encode import native_encode, pad_seq
from .helpers.merge import dare_drop_rescale, ties_align


class AnimaArtistEncode:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"clip": ("CLIP",),
				"text": ("STRING", {
					"multiline": True,
					"tooltip": "Prompt. Artists: @name, @(name), @(name:weight), or (@name:weight). Weight may be negative or zero. Position depends on `artist_placement`.",
				}),
				"mode": (["native", "slider_boost", "slider_dare", "slider_ties"], {
					"default": "slider_boost",
					"tooltip": "native: ComfyUI default token-weight extrapolation, single pass. slider_boost: extract each artist's direction independently and sum. slider_dare: randomly drop 50% of each direction. slider_ties: keep top-magnitude dims per direction, then only dims where artists agree on sign.",
				}),
				"artist_placement": (["force_back", "force_front", "as_written"], {
					"default": "force_back",
					"tooltip": "Where to anchor @ artists before all encodings. force_back (default): move every artist to the end of the prompt — stabilizes bare/single/base alignment, may clash with natural-language suffixes. force_front: move every artist to the very front (before quality tags) — causal attention propagates artist style across the entire prompt. as_written: keep user-written positions; alignment depends on prompt structure.",
				}),
				"boost_strength": ("FLOAT", {
					"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.05,
					"tooltip": "Slider direction magnitude. 0 = slider off (native only). 1.0 = full direction (default). Negative flips every artist's direction.",
				}),
				"dare_drop": ("FLOAT", {
					"default": 0.5, "min": 0.0, "max": 0.99, "step": 0.01,
					"tooltip": "Fraction of direction dims randomly zeroed; survivors rescaled by 1/(1-p). slider_dare only.",
				}),
				"ties_density": ("FLOAT", {
					"default": 0.2, "min": 0.01, "max": 1.0, "step": 0.01,
					"tooltip": "Top-magnitude dim fraction kept per direction (Trim step). slider_ties only.",
				}),
				"concept_positive": ("STRING", {
					"multiline": True, "default": "",
					"tooltip": "Optional concept slider positive text.",
				}),
				"concept_negative": ("STRING", {
					"multiline": True, "default": "",
					"tooltip": "Optional concept slider negative text. Empty = empty prompt anchor.",
				}),
				"concept_strength": ("FLOAT", {
					"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.05,
					"tooltip": "Concept slider magnitude. Auto-off when both concept texts are empty.",
				}),
				"extend_attention": ("BOOLEAN", {
					"default": False,
					"tooltip": "OFF: slider's extended sequence positions are masked out. ON: full effect at the cost of shifted attention.",
				}),
			}
		}

	RETURN_TYPES = ("CONDITIONING", "STRING")
	RETURN_NAMES = ("conditioning", "debug_info")
	FUNCTION = "process"
	CATEGORY = "conditioning/anima"

	def process(self, clip, text, mode, artist_placement, boost_strength, dare_drop, ties_density,
			concept_positive, concept_negative, concept_strength, extend_attention):
		spans = parse_artists(text)
		# Apply placement first — every subsequent encoding uses the relocated text + spans.
		if artist_placement != "as_written" and spans:
			text, spans = relocate_artists(text, spans, artist_placement)

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

		directions, pooled_directions, weights = [], [], []

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

		# Concept slider direction
		concept_direction = None
		concept_pooled_dir = None
		if concept_slider_active:
			pos_list = native_encode(clip, concept_positive.strip())
			neg_list = native_encode(clip, concept_negative.strip())
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