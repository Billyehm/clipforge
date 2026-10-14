"""Turn word-level timestamps into styled ASS subtitles.

The karaoke preset burns one dialogue event per word. Each event shows
the word's caption chunk: words already spoken render white, the active
word carries a \\kf tag so the highlight sweeps across as it is spoken,
and upcoming words stay yellow. Chunks hold at most 8 words (or about
64 characters) and are split into at most two balanced lines, and events
never overlap in time.

Every event also gets an entrance effect (pop by default): a quick scale
punch, a rise-and-settle, a spin, or a plain fade that plays as the word
lands. Effect timings are clamped to fit short words, and words shorter
than 120 ms fall back to a plain fade so the tags never look broken.
"""

from .models import Word

PLAY_RES_X = 1080
PLAY_RES_Y = 1920
MAX_CHUNK_CHARS = 64
FONT_SIZE = 64
MARGIN_V = 220

EFFECTS = ("none", "pop", "bounce", "spin", "fade")

# Caption anchor for the entrance effects: centered, a little above the
# bottom safe area on a 1080x1920 canvas.
CAPTION_X = 540
CAPTION_Y = 1500

# Words shorter than this fall back to a plain fade instead of their
# effect, so transform timings never outrun the event.
MIN_EFFECT_MS = 120


def _ass_time(seconds: float) -> str:
    seconds = max(0.0, seconds)
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours}:{minutes:02d}:{secs:05.2f}"


def _chunk_words(words: list[Word], max_words: int = 8) -> list[list[Word]]:
    """Group words into chunks of at most max_words words / 64 chars."""
    chunks: list[list[Word]] = []
    current: list[Word] = []
    size = 0
    for word in words:
        text = word.text.strip()
        if not text:
            continue
        if current and (
            len(current) >= max_words or size + 1 + len(text) > MAX_CHUNK_CHARS
        ):
            chunks.append(current)
            current = []
            size = 0
        current.append(word)
        size += len(text) + (1 if len(current) > 1 else 0)
    if current:
        chunks.append(current)
    return chunks


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


def _chunk_text_for_word(
    chunk: list[Word], active_idx: int, active_cs: int
) -> str:
    """Render a chunk with one word highlighted.

    Words before the active one render plain (white, already spoken).
    The active word gets a \\kf sweep for its own duration. Upcoming
    words get a far-future \\kf so they sit in the secondary color
    (yellow) until their turn comes, without the active sweep bleeding
    into them.
    """
    texts = [w.text.strip() for w in chunk]
    rendered: list[str] = []
    for i, text in enumerate(texts):
        if i < active_idx:
            rendered.append(text)
        elif i == active_idx:
            rendered.append(f"{{\\kf{active_cs}}}{text}")
        else:
            if i == active_idx + 1:
                rendered.append("{\\kf100000}")
            rendered.append(text)
    groups = _split_lines(chunk)
    lines: list[str] = []
    pos = 0
    for group in groups:
        lines.append(" ".join(rendered[pos : pos + len(group)]))
        pos += len(group)
    return "\\N".join(lines)


def _effect_tags(effect: str, dur_ms: float) -> str:
    """ASS override tags for an entrance effect, clamped to the event."""
    x, y = CAPTION_X, CAPTION_Y
    if effect == "none":
        return ""
    if effect == "fade" or dur_ms < MIN_EFFECT_MS:
        f = max(1, int(min(140, dur_ms)))
        return f"{{\\fad({f},{f})}}"
    if effect == "pop":
        total = int(min(260, dur_ms))
        peak = int(round(total * 140 / 260))
        fade_out = int(min(90, dur_ms))
        return (
            f"{{\\an5\\pos({x},{y})\\fad(0,{fade_out})"
            f"\\t(0,{peak},\\fscx118\\fscy118)"
            f"\\t({peak},{total},\\fscx100\\fscy100)}}"
        )
    if effect == "bounce":
        move_ms = int(min(180, dur_ms))
        fade_out = int(min(90, dur_ms))
        return (
            f"{{\\an5\\pos({x},{y})\\fad(0,{fade_out})"
            f"\\move({x},{y + 26},{x},{y},0,{move_ms})}}"
        )
    if effect == "spin":
        spin_ms = int(min(260, dur_ms))
        fade_out = int(min(90, dur_ms))
        return (
            f"{{\\an5\\pos({x},{y})\\fad(0,{fade_out})"
            f"\\frz360\\t(0,{spin_ms},\\frz0)}}"
        )
    raise ValueError(
        f"Unknown effect {effect!r}. Choose from {', '.join(EFFECTS)}."
    )


ACCENTS: dict[str, str] = {
    "yellow": "&H0000FFFF",
    "cyan": "&H00FFFF00",
    "gold": "&H0037AFD4",
    "white": "&H00FFFFFF",
}


def words_to_ass(
    words: list[Word],
    style: str = "karaoke",
    effect: str = "pop",
    max_words: int = 8,
    accent: str = "yellow",
) -> str:
    """Build ASS subtitle content from word-level timestamps.

    One Dialogue event is emitted per word, showing the word's chunk
    with the active word highlighted by a karaoke sweep. The entrance
    effect is one of "none", "pop", "bounce", "spin", or "fade"
    ("pop" by default); words shorter than 120 ms fall back to "fade"
    for that event. The accent color names the karaoke highlight:
    "yellow", "cyan", "gold", or "white". Unknown styles, effects,
    or accents raise ValueError.
    """
    if style != "karaoke":
        raise ValueError(f"Unknown style {style!r}. Only 'karaoke' is supported.")
    if effect not in EFFECTS:
        raise ValueError(
            f"Unknown effect {effect!r}. Choose from {', '.join(EFFECTS)}."
        )
    if accent not in ACCENTS:
        raise ValueError(
            f"Unknown accent {accent!r}. Choose from {', '.join(ACCENTS)}."
        )
    if max_words < 1:
        raise ValueError("max_words must be at least 1.")

    events: list[tuple[float, float, str]] = []
    for chunk in _chunk_words(words, max_words=max_words):
        for idx, word in enumerate(chunk):
            start, end = word.start, word.end
            if end <= start:
                continue
            sweep_cs = max(1, int(round((end - start) * 100)))
            text = _chunk_text_for_word(chunk, idx, sweep_cs)
            events.append((start, end, text))

    # Guard against overlapping events after chunking.
    for i in range(len(events) - 1):
        start, end, text = events[i]
        next_start = events[i + 1][0]
        if end > next_start:
            end = max(start + 0.01, next_start - 0.01)
            events[i] = (start, end, text)

    # Entrance effects, computed from each event's final length so the
    # transform timings always fit inside the event.
    tagged: list[tuple[float, float, str]] = []
    for start, end, text in events:
        dur_ms = (end - start) * 1000.0
        tagged.append((start, end, _effect_tags(effect, dur_ms) + text))

    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {PLAY_RES_X}
PlayResY: {PLAY_RES_Y}
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Karaoke,Arial,{FONT_SIZE},&H00FFFFFF,{ACCENTS[accent]},&H80000000,&H80000000,-1,0,0,0,100,100,0,0,1,3,1,2,40,40,{MARGIN_V},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    lines = [header.rstrip()]
    for start, end, text in tagged:
        lines.append(
            f"Dialogue: 0,{_ass_time(start)},{_ass_time(end)},Karaoke,,0,0,0,,{text}"
        )
    return "\n".join(lines) + "\n"


def _srt_time(seconds: float) -> str:
    ms = max(0, int(round(seconds * 1000)))
    h, rem = divmod(ms, 3600000)
    m, rem = divmod(rem, 60000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def words_to_srt(words: list[Word]) -> str:
    """Build an SRT subtitle file from word-level timestamps.

    Words are grouped into cues of up to 42 characters or 3 seconds,
    whichever comes first. Plain text, no styling.
    """
    cues: list[list[Word]] = []
    current: list[Word] = []
    for w in words:
        if not w.text.strip():
            continue
        if current:
            text_len = sum(len(x.text) + 1 for x in current) + len(w.text)
            gap = w.start - current[-1].end
            dur = w.end - current[0].start
            if text_len > 42 or gap > 0.8 or dur > 3.0:
                cues.append(current)
                current = []
        current.append(w)
    if current:
        cues.append(current)

    lines: list[str] = []
    for i, cue in enumerate(cues, 1):
        start = max(0.0, cue[0].start)
        end = max(start + 0.2, cue[-1].end)
        text = " ".join(w.text for w in cue)
        lines.append(f"{i}\n{_srt_time(start)} --> {_srt_time(end)}\n{text}\n")
    return "\n".join(lines)
