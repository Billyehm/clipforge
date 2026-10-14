# Clipforge

Turn long-form videos into vertical shorts. Upload a video, get back
ranked clip suggestions with titles, captions, and hashtags, then
rendered 9:16 clips ready for TikTok, Reels, and YouTube Shorts.

Built for a weak GPU: heavy AI work (transcription, moment scoring)
runs through free cloud APIs, while the server only handles light
FFmpeg rendering.

## Repo layout

- `backend/app/pipeline/` - the core pipeline library (pure Python,
  no API keys, no web framework)
- `backend/requirements.txt` - Python dependencies
- `work/` - per-video stage caches (`work/<video_id>/*.json`), gitignored

## Pipeline modules

- `models.py` - shared Pydantic v2 data contracts. `Word`, `Segment`,
  `Transcript` describe the transcript; `Candidate` is one scored
  moment; `ClipRecord` tracks a clip through rendering. Every stage
  reads and writes these, so keep them stable.
- `cache.py` - `StageCache`, a per-video JSON cache at
  `work/<video_id>/`. Stages: `ingest`, `transcript`, `candidates`,
  `clips`. Writes are atomic (temp file plus `os.replace`), so a
  crash never leaves a half-written stage.
- `score.py` - heuristic viral-moment scorer (no LLM yet). Slides
  windows of 20/30/45/60 seconds over the transcript and scores each
  on five signals: hook strength, emotional energy, information
  density, pacing, and curiosity gap. Returns ranked candidates with
  a hook line, a plain-English reason, a title, and hashtags.
- `select.py` - collapses overlapping candidates (over 50% overlap
  keeps the higher score) and picks the top N within a length range,
  with fresh sequential ids.
- `captions.py` - turns word-level timestamps into styled ASS
  subtitles. The karaoke preset highlights each word as it is spoken,
  wraps at about 32 characters, and never shows more than two lines.
- `reframe.py` - plans the 9:16 crop for a clip. Samples frames,
  detects faces with MediaPipe, follows the largest most-central
  face with a smoothed crop path, and falls back to a static center
  crop when no face is found.

## Data flow

```
upload -> ingest -> transcribe -> score -> select -> render -> clips.json
              work/<video_id>/ingest.json
              work/<video_id>/transcript.json
              work/<video_id>/candidates.json
              work/<video_id>/clips.json
```

Each stage caches its output, so re-running selection after tuning
never repeats transcription.

## Quick check

```bash
python3 -m venv .venv && .venv/bin/pip install -r backend/requirements.txt
.venv/bin/python backend/app/pipeline/_selftest.py
```

## Quick start (local dev)

You need: Python 3.11, Node 20, FFmpeg, and a free Groq API key
(console.groq.com).

```bash
# Backend
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then put your GROQ_API_KEY in .env
uvicorn app.main:app --reload --port 8000

# Frontend (new terminal)
cd frontend
npm install
npm run dev
```

Open http://localhost:5173, upload a video, and follow the progress
screen.

## Docker

```bash
cp backend/.env.example backend/.env   # add your GROQ_API_KEY
docker compose up --build
```

API on :8000, web UI on :5173.

## API overview

- `POST /api/videos` - upload a video file, returns a video id. Files over
  `MAX_UPLOAD_MB` (default 500) are not rejected: only the first `TRIM_MB`
  (default 499) megabytes are kept, and the response flags `trimmed: true`
  with a note shown in the dashboard.
- `POST /api/videos/{id}/process?n=8` - run the pipeline in the background
- `GET /api/videos/{id}/progress` - poll for stage progress
- `GET /api/videos/{id}/clips` - ranked clips with scores, titles, hashtags
- `POST /api/videos/{id}/clips/{clip_id}/rerender` - adjust start/end/title
  and re-render one clip
- `GET /api/videos/{id}/download/{clip_id}` - download the finished mp4

## Genre editing profiles

Clipforge detects the video's genre from the transcript (podcast,
tech, business, gaming, education, anime, motivational, news) and
edits each clip to that genre's brief:

- The moment picker follows genre-specific selection rules and also
  writes a description and a thumbnail concept per clip.
- Captions use the genre's accent color (gold, cyan, yellow, white)
  and entrance effect; override with `CAPTION_EFFECT` in `backend/.env`
  (`auto` follows the genre).
- Camera behavior follows the genre: zoom intensity and whether the
  two-speaker split-screen is used.

Every finished video also gets `work/<video_id>/edl.json`, an
edit decision list per clip (timestamps, genre, render settings,
caption files, thumbnail concept), and each clip gets a downloadable
`.srt` subtitle file next to the burned-in captions.

## Honest limits

The scorer ranks engagement signals; it cannot predict what will go
viral. Speaker labels and emotion scores degrade on noisy audio.
Face tracking assumes talking-head style footage. Groq's free tier
has rate limits, so this is built for personal and portfolio use,
not bulk processing. Only clip and repost videos you have the rights
to.
