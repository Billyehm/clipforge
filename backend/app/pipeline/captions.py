"""Turn word-level timestamps into styled ASS subtitles.

The karaoke preset burns one dialogue event per caption chunk. Each
word carries a \\kf tag so the highlight sweeps across as it is spoken:
upcoming words stay yellow, spoken words turn white. Chunks are capped
at about 64 characters and split into at most two balanced lines, and
events never overlap.
"""

from .models import Word

PLAY_RES_X = 1080
PLAY_RES_Y = 1920
MAX_CHUNK_CHARS = 64
FONT_SIZE = 64
MARGIN_V = 220


def _ass_time(seconds: float) -> str:
    seconds = max(0.0, seconds)
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours}:{minutes:02d}:{secs:05.2f}"


def _chunk_words(words: list[Word]) -> list[list[Word]]:
    chunks: list[list[Word]] = []
    current: list[Word] = []
    size = 0
    for word in words:
        text = word.text.strip()
        if not text:
            continue
        if current and size + 1 + len(text) > MAX_CHUNK_CHARS:
            chunks.append(current)
            current = []
            size = 0
        current.append(word)
        size += len(text) + (1 if len(current) > 1 else 0)
    if current:
        chunks.append(current)
    return chunks


def _karaoke_line(chunk: list[Word]) -> str:
    parts = []
    for word in chunk:
        centiseconds = max(1, int(round((word.end - word.start) * 100)))
        parts.append(f"{{\\kf{centiseconds}}}{word.text.strip()}")
    return " ".join(parts)


def _split_lines(chunk: list[Word]) -> list[list[Word]]:
    """Split a chunk into at most two balanced lines."""
    texts = [w.text.strip() for w in chunk]
    total = sum(len(t) for t in texts)
    best_index, best_diff = 1, float("inf")
    running = 0
    for i, text in enumerate(texts, start=1):
        running += len(text) + 1
        diff = abs(running - total / 2)
        if diff < best_diff:
            best_diff, best_index = diff, i
    if best_index >= len(texts):
        return [chunk]
    return [chunk[:best_index], chunk[best_index:]]


def words_to_ass(words: list[Word], style: str = "karaoke") -> str:
    """Build ASS subtitle content from word-level timestamps."""
    if style != "karaoke":
        raise ValueError(f"Unknown style {style!r}. Only 'karaoke' is supported.")

    chunks = _chunk_words(words)
    events: list[tuple[float, float, str]] = []
    for chunk in chunks:
        line = "\\N".join(_karaoke_line(part) for part in _split_lines(chunk))
        events.append((chunk[0].start, chunk[-1].end, line))

    # Guard against overlapping events after chunking.
    for i in range(len(events) - 1):
        start, end, line = events[i]
        next_start = events[i + 1][0]
        if end > next_start:
            end = max(start + 0.01, next_start - 0.01)
            events[i] = (start, end, line)

    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {PLAY_RES_X}
PlayResY: {PLAY_RES_Y}
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Karaoke,Arial,{FONT_SIZE},&H00FFFFFF,&H0000FFFF,&H80000000,&H80000000,-1,0,0,0,100,100,0,0,1,3,1,2,40,40,{MARGIN_V},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    lines = [header.rstrip()]
    for start, end, line in events:
        lines.append(
            f"Dialogue: 0,{_ass_time(start)},{_ass_time(end)},Karaoke,,0,0,0,,{line}"
        )
    return "\n".join(lines) + "\n"
