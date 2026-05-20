import re


ARTIST_PATTERNS = [
	re.compile(r"@\(((?:[^()\\]|\\\(|\\\))+?):(-?\d+(?:\.\d+)?)\)"),   # @(name:weight)
	re.compile(r"@\(((?:[^()\\]|\\\(|\\\))+?)\)"),                       # @(name)
	re.compile(r"\(@((?:[^()\\]|\\\(|\\\))+?):(-?\d+(?:\.\d+)?)\)"),    # (@name:weight)
	re.compile(r"@([\w\u4e00-\u9fff\uac00-\ud7af\-]+)"),                  # @name
]


def parse_artists(text):
	"""Return list of (start, end, name, weight) spans, ordered by position."""
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
	"""Escape unescaped parens inside an artist name so ComfyUI parser treats them literally."""
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
	"""Render an artist as ComfyUI-native `(@name:weight)` or bare `@name` when weight==1."""
	safe = escape_for_native(name)
	core = f"@{safe}"
	if weight == 1.0:
		return core
	return f"({core}:{weight:g})"


def to_native_text(text, spans):
	"""Replace each artist span with its ComfyUI-native form. Preserves character positions for non-artist content."""
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
	"""Strip all artist spans, returning cleaned content."""
	parts = []
	last = 0
	for s, e, _, _ in spans:
		parts.append(text[last:s])
		last = e
	parts.append(text[last:])
	return cleanup_text("".join(parts))


def single_artist_text(text, spans, keep_idx):
	"""Keep only one artist (rendered at weight 1.0); strip the rest."""
	parts = []
	last = 0
	for idx, (s, e, name, _) in enumerate(spans):
		parts.append(text[last:s])
		if idx == keep_idx:
			parts.append(artist_token_str(name, 1.0))
		last = e
	parts.append(text[last:])
	return cleanup_text("".join(parts))


def relocate_artists(text, spans, placement):
	"""Move all @ artists to the front or back of the text.

	placement: 'as_written' | 'force_back' | 'force_front'.
	Returns (new_text, new_spans). Artists are emitted in ComfyUI-native form
	so bare/single/native text all share identical artist tokenization positions.
	"""
	if placement == "as_written" or not spans:
		return text, spans
	artist_strs = [artist_token_str(name, w) for _, _, name, w in spans]
	parts = []
	last = 0
	for s, e, _, _ in spans:
		parts.append(text[last:s])
		last = e
	parts.append(text[last:])
	content = cleanup_text("".join(parts))
	artist_block = ", ".join(artist_strs)
	if placement == "force_back":
		new_text = f"{content}, {artist_block}" if content else artist_block
	else:  # force_front
		new_text = f"{artist_block}, {content}" if content else artist_block
	new_spans = parse_artists(new_text)
	return new_text, new_spans