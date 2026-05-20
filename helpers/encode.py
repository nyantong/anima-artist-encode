import torch


def native_encode(clip, text):
	"""Single-pass ComfyUI text encoding. Returns CONDITIONING list."""
	tokens = clip.tokenize(text)
	return clip.encode_from_tokens_scheduled(tokens)


def pad_seq(tensor, target_len):
	"""Pad/truncate a (B, S, D) cond tensor along the sequence axis."""
	cur = tensor.shape[1]
	if cur == target_len:
		return tensor
	if cur > target_len:
		return tensor[:, :target_len]
	pad_shape = list(tensor.shape)
	pad_shape[1] = target_len - cur
	pad = torch.zeros(*pad_shape, dtype=tensor.dtype, device=tensor.device)
	return torch.cat([tensor, pad], dim=1)