"""Plan 9:16 crop paths for a clip, with multi-speaker support.

Frames are sampled a few times per second and run through MediaPipe
face detection. Every detection becomes a point (cx, cy, s): normalized
center x, center y, and size. Detections are clustered across samples
into speaker tracks with greedy nearest-neighbor matching, each track
is smoothed with a moving average so crops glide instead of jumping,
and stable tracks pick the render mode:

- 2 or more stable tracks -> "split" (two speakers stacked vertically)
- 1 stable track          -> "track" (single face-tracked crop)
- 0 stable tracks         -> "center" (static center crop)

Import safe: cv2 and mediapipe are imported lazily inside the function,
so importing this module never fails on a machine without them.
"""

SMOOTH_WINDOW = 15
# A detection only joins an existing track if it is this close (normalized).
TRACK_MAX_JUMP = 0.25
# A track counts as stable when actually detected in this fraction of samples.
STABLE_FRACTION = 0.40


def _smooth(values: list[float], window: int = SMOOTH_WINDOW) -> list[float]:
    half = window // 2
    smoothed = []
    for i in range(len(values)):
        lo = max(0, i - half)
        hi = min(len(values), i + half + 1)
        smoothed.append(sum(values[lo:hi]) / (hi - lo))
    return smoothed


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def _interp_track(
    track: list[tuple[float, float, float, float]], t: float
) -> tuple[float, float, float]:
    """Linearly interpolate (cx, cy, s) of a track at time t.

    Track samples are (t, cx, cy, s). Times outside the sampled range
    clamp to the nearest end.
    """
    if not track:
        return (0.5, 0.5, 0.3)
    if t <= track[0][0]:
        _, cx, cy, s = track[0]
        return (cx, cy, s)
    if t >= track[-1][0]:
        _, cx, cy, s = track[-1]
        return (cx, cy, s)
    for i in range(len(track) - 1):
        t0, cx0, cy0, s0 = track[i]
        t1, cx1, cy1, s1 = track[i + 1]
        if t0 <= t <= t1:
            span = t1 - t0
            f = 0.0 if span <= 0 else (t - t0) / span
            return (
                cx0 + f * (cx1 - cx0),
                cy0 + f * (cy1 - cy0),
                s0 + f * (s1 - s0),
            )
    _, cx, cy, s = track[-1]
    return (cx, cy, s)


def plan_reframe(
    video_path: str, start: float, end: float, fps_sample: int = 2
) -> dict:
    """Return a multi-track reframe plan for a clip.

    The returned dict has:
      mode: "track" | "split" | "center"
      tracks: list of tracks, each a list of (t, cx, cy, s) with t in
              seconds CLIP-RELATIVE (0-based from start), cx/cy/s
              normalized 0..1. Tracks are ordered by presence, most
              detected first.
      src_w, src_h: source frame size in pixels
      width, height: output size (1080 x 1920)
      path: backward-compatible [(t, cx), ...] of the primary track
    """
    try:
        import cv2
    except ImportError:
        raise RuntimeError(
            "plan_reframe needs the 'opencv-python-headless' package "
            "(import cv2 failed)."
        )
    try:
        import mediapipe as mp
    except ImportError:
        raise RuntimeError(
            "plan_reframe needs the 'mediapipe' package (import failed)."
        )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    times: list[float] = []
    t = start
    while t < end:
        times.append(t)
        t += 1.0 / fps_sample
    if not times:
        cap.release()
        raise RuntimeError("Empty time range for reframe planning.")

    detector = mp.solutions.face_detection.FaceDetection(
        model_selection=0, min_detection_confidence=0.5
    )

    # Each track: {"samples": [(t, cx, cy, s, detected)], "last": (cx, cy, s)}
    tracks: list[dict] = []
    for sample_t in times:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(sample_t * fps))
        ok, frame = cap.read()
        detections: list[tuple[float, float, float]] = []
        if ok:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = detector.process(rgb)
            if results.detections:
                h, w = frame.shape[:2]
                for detection in results.detections:
                    box = detection.location_data.relative_bounding_box
                    cx = min(1.0, max(0.0, box.xmin + box.width / 2.0))
                    cy = min(1.0, max(0.0, box.ymin + box.height / 2.0))
                    s = (box.width * box.height) ** 0.5
                    detections.append((cx, cy, s))

        # Greedy nearest-neighbor matching of detections to tracks.
        pairs: list[tuple[float, int, int]] = []
        for di, det in enumerate(detections):
            for ti, track in enumerate(tracks):
                pairs.append((_dist(det[:2], track["last"][:2]), di, ti))
        pairs.sort(key=lambda p: p[0])
        det_used = [False] * len(detections)
        # Indices of tracks that got a real detection this sample
        # (matched or newly created). Everything else coasts.
        live: set[int] = set()
        for dist, di, ti in pairs:
            if dist > TRACK_MAX_JUMP or det_used[di] or ti in live:
                continue
            det_used[di] = True
            live.add(ti)
            cx, cy, s = detections[di]
            tracks[ti]["samples"].append((sample_t, cx, cy, s, True))
            tracks[ti]["last"] = (cx, cy, s)

        # Unmatched detections start new tracks; unmatched tracks coast.
        for di, det in enumerate(detections):
            if not det_used[di]:
                cx, cy, s = det
                tracks.append(
                    {
                        "samples": [(sample_t, cx, cy, s, True)],
                        "last": (cx, cy, s),
                    }
                )
                live.add(len(tracks) - 1)
        for ti, track in enumerate(tracks):
            if ti not in live:
                cx, cy, s = track["last"]
                track["samples"].append((sample_t, cx, cy, s, False))

    cap.release()
    detector.close()

    # Smooth each track and measure stability.
    n = len(times)
    stable: list[tuple[int, list[tuple[float, float, float, float]]]] = []
    for track in tracks:
        samples = track["samples"]
        detected = sum(1 for s in samples if s[4])
        if detected / n <= STABLE_FRACTION:
            continue
        cxs = _smooth([s[1] for s in samples])
        cys = _smooth([s[2] for s in samples])
        sss = _smooth([s[3] for s in samples])
        smoothed = [
            (round(s[0] - start, 2), round(cx, 4), round(cy, 4), round(ss, 4))
            for s, cx, cy, ss in zip(samples, cxs, cys, sss)
        ]
        stable.append((detected, smoothed))

    # Most present track first.
    stable.sort(key=lambda item: item[0], reverse=True)

    if len(stable) >= 2:
        mode = "split"
        kept = [stable[0][1], stable[1][1]]
    elif len(stable) == 1:
        mode = "track"
        kept = [stable[0][1]]
    else:
        mode = "center"
        kept = []

    path = [(s_t, cx) for s_t, cx, _cy, _s in kept[0]] if kept else [
        (round(sample_t - start, 2), 0.5) for sample_t in times
    ]

    return {
        "mode": mode,
        "tracks": kept,
        "src_w": src_w,
        "src_h": src_h,
        "width": 1080,
        "height": 1920,
        "path": path,
    }
