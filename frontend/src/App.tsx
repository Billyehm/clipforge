import { useCallback, useEffect, useRef, useState } from 'react'
import {
  captionsUrl,
  downloadUrl,
  getClips,
  getProgress,
  rerenderClip,
  startProcessing,
  uploadVideo,
} from './api'
import type { ClipRecord, ProgressResponse } from './types'

type View = 'upload' | 'processing' | 'clips'

const STAGE_LABELS: Record<string, string> = {
  ingest: 'Reading your video',
  upload: 'Reading your video',
  transcribe: 'Transcribing audio',
  analyze: 'Finding viral moments',
  score: 'Finding viral moments',
  select: 'Picking the best clips',
  render: 'Rendering clips',
  done: 'Done',
}

function stageLabel(stage: string): string {
  return STAGE_LABELS[stage] ?? stage.charAt(0).toUpperCase() + stage.slice(1)
}

function formatTime(sec: number): string {
  const s = Math.max(0, Math.floor(sec))
  const m = Math.floor(s / 60)
  return `${m}:${String(s % 60).padStart(2, '0')}`
}

function scoreClass(score: number): string {
  if (score >= 80) return 'score high'
  if (score >= 60) return 'score good'
  if (score >= 40) return 'score mid'
  return 'score low'
}

export default function App() {
  const [view, setView] = useState<View>('upload')
  const [videoId, setVideoId] = useState<string | null>(null)
  const [notice, setNotice] = useState<string | null>(null)

  return (
    <div className="app">
      <header className="topbar">
        <h1>
          Clipforge <span className="tag">beta</span>
        </h1>
        {view !== 'upload' && (
          <button
            className="btn ghost"
            onClick={() => {
              setVideoId(null)
              setNotice(null)
              setView('upload')
            }}
          >
            New video
          </button>
        )}
      </header>

      <main>
        {view === 'upload' && (
          <UploadView
            onUploaded={(id, note) => {
              setVideoId(id)
              setNotice(note)
              setView('processing')
            }}
          />
        )}
        {view === 'processing' && videoId && (
          <>
            {notice && <p className="notice">{notice}</p>}
            <ProcessingView
              videoId={videoId}
              onDone={() => setView('clips')}
              onError={() => setView('upload')}
            />
          </>
        )}
        {view === 'clips' && videoId && <ClipsView videoId={videoId} />}
      </main>
    </div>
  )
}

function UploadView({
  onUploaded,
}: {
  onUploaded: (id: string, note: string | null) => void
}) {
  const [dragging, setDragging] = useState(false)
  const [progress, setProgress] = useState<number | null>(null)
  const [error, setError] = useState<string | null>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  const send = useCallback(
    async (file: File) => {
      setError(null)
      setProgress(0)
      try {
        const res = await uploadVideo(file, setProgress)
        onUploaded(res.video_id, res.note ?? null)
      } catch (e) {
        setError(e instanceof Error ? e.message : 'Upload failed')
        setProgress(null)
      }
    },
    [onUploaded],
  )

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setDragging(false)
    const file = e.dataTransfer.files?.[0]
    if (file) void send(file)
  }

  return (
    <section className="panel">
      <h2>Turn a long video into short clips</h2>
      <p className="muted">
        Upload a video file. We will transcribe it, find the best moments, and
        cut vertical clips ready for TikTok, Reels, and Shorts.
      </p>

      <div
        className={`dropzone${dragging ? ' dragging' : ''}`}
        onDragOver={(e) => {
          e.preventDefault()
          setDragging(true)
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        onClick={() => inputRef.current?.click()}
        role="button"
        tabIndex={0}
        onKeyDown={(e) => {
          if (e.key === 'Enter') inputRef.current?.click()
        }}
      >
        <div className="dropzone-icon">+</div>
        <p>
          <strong>Drop your video here</strong> or click to browse
        </p>
        <p className="muted small">MP4, MOV, MKV, WebM up to a few GB</p>
        <input
          ref={inputRef}
          type="file"
          accept="video/*"
          hidden
          onChange={(e) => {
            const file = e.target.files?.[0]
            if (file) void send(file)
          }}
        />
      </div>

      {progress !== null && (
        <div className="progress-wrap">
          <div className="progress-bar">
            <div className="progress-fill" style={{ width: `${progress}%` }} />
          </div>
          <p className="muted small">Uploading... {progress}%</p>
        </div>
      )}

      {error && <p className="error">{error}</p>}
    </section>
  )
}

function ProcessingView({
  videoId,
  onDone,
  onError,
}: {
  videoId: string
  onDone: () => void
  onError: () => void
}) {
  const [progress, setProgress] = useState<ProgressResponse | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    startProcessing(videoId, 8).catch((e: unknown) => {
      if (!cancelled)
        setError(e instanceof Error ? e.message : 'Could not start processing')
    })
    let pollFailures = 0
    const timer = setInterval(async () => {
      try {
        const p = await getProgress(videoId)
        if (cancelled) return
        pollFailures = 0
        setProgress(p)
        if (p.error) {
          setError(p.error)
          clearInterval(timer)
          return
        }
        const finished =
          p.stage === 'done' || (p.total > 0 && p.done >= p.total)
        if (finished) {
          clearInterval(timer)
          onDone()
        }
      } catch (e) {
        // A single failed poll is usually a network blip. Only give up
        // after several in a row so a shaky connection cannot kill a run.
        pollFailures += 1
        if (cancelled) return
        if (pollFailures >= 5) {
          setError(e instanceof Error ? e.message : 'Lost contact with server')
          clearInterval(timer)
        }
      }
    }, 2000)
    return () => {
      cancelled = true
      clearInterval(timer)
    }
  }, [videoId, onDone])

  const pct =
    progress && progress.total > 0
      ? Math.min(100, Math.round((progress.done / progress.total) * 100))
      : 0

  return (
    <section className="panel">
      <h2>{progress ? stageLabel(progress.stage) : 'Getting started'}</h2>
      <p className="muted">
        This usually takes a few minutes. You can leave this tab open, we will
        keep working.
      </p>
      <div className="progress-wrap">
        <div className="progress-bar">
          <div className="progress-fill" style={{ width: `${pct}%` }} />
        </div>
        <p className="muted small">
          {progress ? `${progress.done} of ${progress.total} steps` : 'Starting...'}
        </p>
      </div>
      {error && (
        <div>
          <p className="error">{error}</p>
          <button className="btn" onClick={onError}>
            Back to upload
          </button>
        </div>
      )}
    </section>
  )
}

function ClipsView({ videoId }: { videoId: string }) {
  const [clips, setClips] = useState<ClipRecord[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const refresh = useCallback(async () => {
    try {
      setLoading(true)
      setError(null)
      setClips(await getClips(videoId))
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Could not load clips')
    } finally {
      setLoading(false)
    }
  }, [videoId])

  useEffect(() => {
    void refresh()
  }, [refresh])

  if (loading) return <p className="muted">Loading your clips...</p>
  if (error) return <p className="error">{error}</p>
  if (clips.length === 0)
    return <p className="muted">No clips were found for this video.</p>

  return (
    <section>
      <div className="clips-header">
        <h2>
          Your clips <span className="muted">({clips.length})</span>
        </h2>
        <p className="muted">
          Preview each clip, tweak the title or timing, then download or copy
          the caption.
        </p>
      </div>
      <div className="clips-grid">
        {clips.map((clip) => (
          <ClipCard key={clip.id} videoId={videoId} initial={clip} />
        ))}
      </div>
    </section>
  )
}

function ClipCard({ videoId, initial }: { videoId: string; initial: ClipRecord }) {
  const [clip, setClip] = useState<ClipRecord>(initial)
  const [title, setTitle] = useState(initial.candidate.title)
  const [start, setStart] = useState(String(Math.round(initial.candidate.start)))
  const [end, setEnd] = useState(String(Math.round(initial.candidate.end)))
  const [busy, setBusy] = useState(false)
  const [copied, setCopied] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const c = clip.candidate

  const doRerender = async () => {
    setBusy(true)
    setError(null)
    try {
      const updated = await rerenderClip(videoId, clip.id, {
        start: Number(start),
        end: Number(end),
        title,
      })
      setClip(updated)
      setTitle(updated.candidate.title)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Re-render failed')
    } finally {
      setBusy(false)
    }
  }

  const copyCaption = async () => {
    const text = `${title}\n\n${c.hashtags.join(' ')}`
    try {
      await navigator.clipboard.writeText(text)
    } catch {
      const ta = document.createElement('textarea')
      ta.value = text
      document.body.appendChild(ta)
      ta.select()
      document.execCommand('copy')
      document.body.removeChild(ta)
    }
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  return (
    <article className="card">
      <div className="card-video">
        <video
          src={downloadUrl(videoId, clip.id)}
          controls
          preload="metadata"
          playsInline
        />
        <span className={scoreClass(c.score)}>{c.score}</span>
        {c.genre && <span className="genre-badge">{c.genre}</span>}
      </div>

      <div className="card-body">
        <label className="field-label">Title</label>
        <input
          className="title-input"
          value={title}
          onChange={(e) => setTitle(e.target.value)}
        />

        <p className="hook">{c.hook_line}</p>
        <p className="reason muted">{c.reason}</p>
        {c.description && <p className="muted small">{c.description}</p>}
        {c.thumbnail_concept && (
          <p className="muted small">
            <strong>Thumbnail:</strong> {c.thumbnail_concept}
          </p>
        )}

        {c.speaker && <p className="muted small">Speaker: {c.speaker}</p>}

        <div className="chips">
          {c.hashtags.map((tag) => (
            <span key={tag} className="chip">
              {tag}
            </span>
          ))}
        </div>

        <div className="time-row">
          <label>
            Start (s)
            <input
              type="number"
              min={0}
              value={start}
              onChange={(e) => setStart(e.target.value)}
            />
          </label>
          <label>
            End (s)
            <input
              type="number"
              min={0}
              value={end}
              onChange={(e) => setEnd(e.target.value)}
            />
          </label>
          <span className="muted small time-hint">
            Clip runs {formatTime(Number(start) || 0)} to {formatTime(Number(end) || 0)}
          </span>
        </div>

        {clip.status && clip.status !== 'done' && (
          <p className="muted small">Status: {clip.status}</p>
        )}
        {clip.error && <p className="error small">{clip.error}</p>}
        {error && <p className="error small">{error}</p>}

        <div className="actions">
          <button className="btn primary" onClick={doRerender} disabled={busy}>
            {busy ? 'Rendering...' : 'Re-render'}
          </button>
          <a
            className="btn"
            href={downloadUrl(videoId, clip.id)}
            download
          >
            Download
          </a>
          <a
            className="btn ghost"
            href={captionsUrl(videoId, clip.id)}
            download
          >
            SRT
          </a>
          <button className="btn ghost" onClick={copyCaption}>
            {copied ? 'Copied' : 'Copy caption'}
          </button>
        </div>
      </div>
    </article>
  )
}
