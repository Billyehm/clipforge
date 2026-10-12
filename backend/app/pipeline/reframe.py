"""Plan a 9:16 crop path for a clip.

Frames are sampled a few times per second and run through MediaPipe
face detection. The largest, most central face becomes the subject and
its horizontal center is smoothed with a moving average so the crop
glides instead of jumping. When faces are missing from most samples,
the plan falls back to a static center crop.

Import safe: cv2 and mediapipe are imported lazily inside the function,
so importing this module never fails on a machine without them.
"""

SMOOTH_WINDOW = 15
FACE_MISS_THRESHOLD = 0.7


def _smooth(values: list[float], window: int = SMOOTH_WINDOW) -> list[float]:
    half = window // 2
    smoothed = []
    for i in range(len(values)):
        lo = max(0, i - half)
        hi = min(len(values), i + half + 1)
        smoothed.append(sum(values[lo:hi]) / (hi - lo))
    return smoothed


def plan_reframe(
    video_path: str, start: float, end: float, fps_sample: int = 2
) -> dict:
    """Return a crop plan: mode, per-sample (t, x_center), and output size.

    x_center is normalized 0 to 1 across the frame width.
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
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    times = []
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
    centers: list[float] = []
    misses = 0
    for sample_t in times:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(sample_t * fps))
        ok, frame = cap.read()
        if not ok:
            misses += 1
            centers.append(0.5)
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = detector.process(rgb)
        best = None
        if results.detections:
            for detection in results.detections:
                box = detection.location_data.relative_bounding_box
                cx = box.xmin + box.width / 2.0
                area = box.width * box.height
                score = area - 0.5 * abs(cx - 0.5)
                if best is None or score > best[0]:
                    best = (score, cx)
        if best is None:
            misses += 1
            centers.append(0.5)
        else:
            centers.append(max(0.0, min(1.0, best[1])))

    cap.release()
    detector.close()

    if misses / len(times) > FACE_MISS_THRESHOLD:
        mode = "center"
        path = [(round(sample_t, 2), 0.5) for sample_t in times]
    else:
        mode = "track"
        smoothed = _smooth(centers)
        path = [
            (round(sample_t, 2), round(x, 4))
            for sample_t, x in zip(times, smoothed)
        ]

    return {"mode": mode, "path": path, "width": 1080, "height": 1920}
