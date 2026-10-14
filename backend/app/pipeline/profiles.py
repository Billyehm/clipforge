"""Genre-aware editing profiles.

Each profile distills a professional editing brief for one content genre
into data the pipeline can act on: how to pick moments, how captions
should look, and how the camera should behave. Anything a simple
FFmpeg/OpenCV pipeline cannot honestly automate (B-roll sourcing, music
selection, motion graphics) is left out rather than faked.
"""

GENRE_PROFILES: dict[str, dict] = {
    "podcast": {
        "label": "Podcast / interview",
        "selection": (
            "Favor complete thoughts: a strong opening statement, question, "
            "or surprising revelation, then the argument or story that "
            "supports it, ending on a memorable line or cliffhanger. Keep "
            "the speaker's exact meaning; never stitch unrelated sentences "
            "into fake controversy. Prefer medium close-up framing and "
            "restraint over effects."
        ),
        "caption_effect": "pop",
        "accent": "gold",
        "zoom": 0.06,
        "allow_split": True,
        "punch_in": True,
    },
    "tech": {
        "label": "AI and technology",
        "selection": (
            "Open on a surprising capability, problem, discovery, or demo. "
            "Keep enough context for the clip to stand alone and preserve "
            "technical accuracy: product names, model names, numbers, and "
            "acronyms must stay exact. Never exaggerate a capability or "
            "present speculation as fact. End on the most useful insight."
        ),
        "caption_effect": "pop",
        "accent": "cyan",
        "zoom": 0.05,
        "allow_split": True,
        "punch_in": False,
    },
    "business": {
        "label": "Business and entrepreneurship",
        "selection": (
            "Open on a concrete lesson, counterintuitive opinion, mistake, "
            "or compelling claim, then keep the evidence or story that backs "
            "it. Preserve numbers, examples, and qualifications; never drop "
            "context in a way that changes the argument. End on a practical "
            "takeaway. Never present unsupported figures as fact."
        ),
        "caption_effect": "pop",
        "accent": "gold",
        "zoom": 0.06,
        "allow_split": False,
        "punch_in": True,
    },
    "gaming": {
        "label": "Gaming and streamer",
        "selection": (
            "Pick one complete event: a clutch play, funny fail, big "
            "reaction, or surprising outcome. Start close to the setup but "
            "never cut information the viewer needs to understand what "
            "happened. Let the payoff play at its natural speed and end "
            "right after it. Never invent kills, reactions, or outcomes."
        ),
        "caption_effect": "bounce",
        "accent": "yellow",
        "zoom": 0.09,
        "allow_split": True,
        "punch_in": True,
    },
    "education": {
        "label": "Educational and explainer",
        "selection": (
            "Open on a surprising fact, question, or misconception, then the "
            "explanation in logical order, ending on a clear takeaway that "
            "answers the opening. Keep the evidence and reasoning; never drop "
            "qualifications needed for accuracy. Favor clarity over flash."
        ),
        "caption_effect": "fade",
        "accent": "cyan",
        "zoom": 0.04,
        "allow_split": False,
        "punch_in": False,
    },
    "anime": {
        "label": "Anime, film and analysis",
        "selection": (
            "Find the emotional or analytical core: a compelling line, "
            "observation, or meaningful visual, building to the strongest "
            "supported moment. Preserve the context a viewer needs. Never "
            "misrepresent dialogue or combine scenes to imply events that "
            "never happened. Prefer clean cuts and restraint."
        ),
        "caption_effect": "fade",
        "accent": "white",
        "zoom": 0.04,
        "allow_split": False,
        "punch_in": False,
    },
    "motivational": {
        "label": "Motivational",
        "selection": (
            "Start on a powerful statement or challenging question, build "
            "from struggle toward insight or action, and end on a memorable "
            "line or takeaway. Preserve the speaker's real meaning; never "
            "use generic inspirational lines they did not say. Give strong "
            "statements room to land."
        ),
        "caption_effect": "pop",
        "accent": "gold",
        "zoom": 0.07,
        "allow_split": False,
        "punch_in": True,
    },
    "news": {
        "label": "News, debate and commentary",
        "selection": (
            "Pick one complete argument, explanation, or exchange, opening "
            "on the most informative supported moment. Keep the context and "
            "qualifications the claim needs; never edit words to reverse "
            "meaning. Distinguish fact from opinion. End on the conclusion, "
            "not a manufactured cliffhanger."
        ),
        "caption_effect": "fade",
        "accent": "white",
        "zoom": 0.03,
        "allow_split": True,
        "punch_in": False,
    },
}

# Keywords used for cheap genre detection on the transcript.
_GENRE_KEYWORDS: dict[str, list[str]] = {
    "podcast": ["interview", "host", "guest", "episode", "welcome back", "podcast", "conversation"],
    "tech": [" ai ", "model", "algorithm", "code", "software", "api", "robot", "neural", "data center", "chip"],
    "business": ["money", "revenue", "startup", "founder", "marketing", "sales", "profit", "business", "investor", "customers"],
    "gaming": ["game", "gameplay", "level", "boss", "stream", "clutch", "ranked", "controller"],
    "education": ["lesson", "learn", "explain", "science", "equation", "study", "experiment", "history"],
    "anime": ["anime", "manga", "episode", "character", "film", "movie", "scene", "director", "studio"],
    "motivational": ["mindset", "discipline", "dream", "believe", "fear", "success", "goal", "grind", "purpose"],
    "news": ["breaking", "report", "election", "government", "policy", "minister", "according to", "officials"],
}


def detect_genre(transcript) -> str:
    """Pick the best genre for a transcript by keyword scoring.

    Returns a genre key from GENRE_PROFILES, defaulting to "podcast"
    when nothing matches clearly.
    """
    text = " ".join(s.text for s in transcript.segments).lower()
    if not text.strip():
        return "podcast"
    words = len(text.split())
    best, best_score = "podcast", 0.0
    for genre, keywords in _GENRE_KEYWORDS.items():
        hits = sum(text.count(k.strip()) for k in keywords)
        score = hits / max(1, words / 100)
        if score > best_score:
            best, best_score = genre, score
    return best if best_score > 0.5 else "podcast"


def get_profile(genre: str) -> dict:
    """Return the editing profile for a genre key, defaulting to podcast."""
    return GENRE_PROFILES.get(genre, GENRE_PROFILES["podcast"])
