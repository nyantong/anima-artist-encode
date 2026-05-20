import torch


def dare_drop_rescale(delta, p):
	"""DARE: Bernoulli drop with rate p, survivors rescaled by 1/(1-p). Expectation preserved."""
	if p <= 0.0:
		return delta
	if p >= 1.0:
		return torch.zeros_like(delta)
	mask = (torch.rand_like(delta) > p).to(delta.dtype)
	return (delta * mask) / (1.0 - p)


def ties_align(directions, density):
	"""TIES alignment: trim + sign election + disjoint mask. Mask-only — per-artist weights preserved upstream."""
	if not directions or density >= 1.0:
		return list(directions)
	# Step 1: Trim — keep top-(density) magnitude dims per direction
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
	# Step 2: Elect dominant sign per dim
	stack = torch.stack(trimmed, dim=0)
	sign_sum = (stack.abs() * stack.sign()).sum(dim=0)
	dominant_sign = sign_sum.sign()
	nonzero_dom = (dominant_sign != 0).to(stack.dtype)
	# Step 3: Disjoint mask — each direction keeps only dims agreeing with dominant
	aligned = []
	for d in trimmed:
		align_mask = (d.sign() == dominant_sign).to(d.dtype)
		aligned.append(d * align_mask * nonzero_dom)
	return aligned