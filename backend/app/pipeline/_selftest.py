"""Tiny self-test for BIT 1: builds a fake transcript and exercises
score, select, captions, and the stage cache. Run with:
    .venv/bin/python backend/app/pipeline/_selftest.py
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.pipeline import (  # noqa: E402
    Segment,
    StageCache,
    Transcript,
    Word,
    pick_top,
    score_transcript,
    words_to_ass,
)

SENTENCES = [
    (0, "Welcome back to the show, today we talk about building in public."),
    (9, "Nobody talks about this mistake, but it killed my first startup."),
    (18, "I lost fifty thousand dollars in six months. Here is why it happened."),
    (28, "The truth about fundraising is simple: nobody cares about your idea."),
    (38, "They care about traction. Show me the numbers, show me the growth!"),
    (48, "We went from zero to ten thousand users with one crazy landing page."),
    (59, "Imagine waking up to a thousand signups overnight. It was insane!"),
    (70, "But then everything broke. The servers melted at 3am on a Tuesday."),
    (81, "I learned the hard way: always load test before you launch."),
    (91, "So here is what I would do differently. First, start with the customer."),
    (102, "Stop building features nobody asked for. Start having conversations."),
    (113, "Second, avoid vanity metrics. Revenue is the only metric that matters."),
    (124, "What happened next surprised everyone on the team, including me."),
    (135, "A competitor copied our entire product in just two weeks. Shocking!"),
    (146, "But here is the secret: execution beats ideas every single time."),
    (157, "We shipped faster, we listened better, and we won the market."),
    (168, "If you take one thing from this episode, make it this one lesson."),
    (179, "Build something people want, then tell the world. That is the whole game."),
    (190, "Thanks for listening. Share this with a founder who needs to hear it."),
]


def make_transcript() -> Transcript:
    segments = []
    for i, (start, text) in enumerate(SENTENCES):
        end = SENTENCES[i + 1][0] - 0.5 if i + 1 < len(SENTENCES) else start + 9.0
        tokens = text.split()
        span = max(0.1, end - start)
        words = [
            Word(
                text=tok,
                start=round(start + span * j / len(tokens), 2),
                end=round(start + span * (j + 1) / len(tokens), 2),
            )
            for j, tok in enumerate(tokens)
        ]
        segments.append(
            Segment(
                id=i,
                start=float(start),
                end=float(end),
                text=text,
                speaker="SPEAKER_0" if i % 3 else "SPEAKER_1",
                words=words,
            )
        )
    return Transcript(video_id="selftest", duration=200.0, segments=segments)


def main() -> None:
    transcript = make_transcript()
    candidates = score_transcript(transcript)
    print(f"scored {len(candidates)} raw candidates from {len(transcript.segments)} segments")
    top = pick_top(candidates, n=10)

    print("\nTop 3 candidates:")
    for c in top[:3]:
        print(f"  [{c.id}] score={c.score}  {c.start:.1f}s -> {c.end:.1f}s  speaker={c.speaker}")
        print(f"       title: {c.title}")
        print(f"       hook: {c.hook_line}")
        print(f"       reason: {c.reason}")
        print(f"       hashtags: {' '.join(c.hashtags)}")

    # captions from the words inside the top candidate's window
    best = top[0]
    words = [
        w
        for s in transcript.segments
        for w in s.words
        if w.end > best.start and w.start < best.end
    ]
    ass = words_to_ass(words)
    print(f"\nASS output: {len(ass.splitlines())} lines, {ass.count('Dialogue:')} events")
    print("first event:")
    for line in ass.splitlines():
        if line.startswith("Dialogue:"):
            print("  " + line[:120] + "...")
            break

    # cache round trip
    with tempfile.TemporaryDirectory() as tmp:
        cache = StageCache("selftest", root=tmp)
        cache.save("candidates", [c.model_dump() for c in top])
        loaded = cache.load("candidates")
        assert loaded is not None and len(loaded) == len(top)
        assert cache.exists("candidates") and not cache.exists("transcript")
        print(f"\ncache round trip ok: {len(loaded)} candidates saved and loaded")

    # reframe must fail cleanly without heavy deps
    from app.pipeline import reframe

    try:
        reframe.plan_reframe("nope.mp4", 0, 10)
        print("reframe: unexpectedly ran (deps present)")
    except RuntimeError as e:
        print(f"reframe import guard ok: {e}")

    print("\nself-test passed")


if __name__ == "__main__":
    main()
