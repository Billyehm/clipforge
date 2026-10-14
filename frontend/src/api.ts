// Typed fetch client for the Clipforge FastAPI backend.
// All paths are relative so the Vite dev proxy forwards /api to localhost:8000.

import type {
  ClipRecord,
  ProgressResponse,
  RerenderRequest,
  VideoInfo,
  VideoUploadResponse,
} from './types'

const BASE = '/api'

async function check(res: Response): Promise<Response> {
  if (!res.ok) {
    const text = await res.text().catch(() => '')
    throw new Error(text || `Request failed with status ${res.status}`)
  }
  return res
}

export function uploadVideo(
  file: File,
  onProgress: (percent: number) => void,
): Promise<VideoUploadResponse> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest()
    xhr.open('POST', `${BASE}/videos`)
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) onProgress(Math.round((e.loaded / e.total) * 100))
    }
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        try {
          resolve(JSON.parse(xhr.responseText) as VideoUploadResponse)
        } catch {
          reject(new Error('Could not read the upload response'))
        }
      } else {
        reject(new Error(xhr.responseText || `Upload failed (${xhr.status})`))
      }
    }
    xhr.onerror = () => reject(new Error('Upload failed: network error'))
    const form = new FormData()
    form.append('file', file)
    xhr.send(form)
  })
}

export async function getVideo(id: string): Promise<VideoInfo> {
  const res = await check(await fetch(`${BASE}/videos/${id}`))
  return res.json() as Promise<VideoInfo>
}

export async function startProcessing(id: string, n = 8): Promise<{ job: string }> {
  const res = await check(
    await fetch(`${BASE}/videos/${id}/process?n=${n}`, { method: 'POST' }),
  )
  return res.json() as Promise<{ job: string }>
}

export async function getProgress(id: string): Promise<ProgressResponse> {
  const res = await check(await fetch(`${BASE}/videos/${id}/progress`))
  return res.json() as Promise<ProgressResponse>
}

export async function getClips(id: string): Promise<ClipRecord[]> {
  const res = await check(await fetch(`${BASE}/videos/${id}/clips`))
  return res.json() as Promise<ClipRecord[]>
}

export async function rerenderClip(
  videoId: string,
  clipId: string,
  body: RerenderRequest,
): Promise<ClipRecord> {
  const res = await check(
    await fetch(`${BASE}/videos/${videoId}/clips/${clipId}/rerender`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }),
  )
  return res.json() as Promise<ClipRecord>
}

export function downloadUrl(videoId: string, clipId: string): string {
  return `${BASE}/videos/${videoId}/download/${clipId}`
}

export function captionsUrl(videoId: string, clipId: string): string {
  return `${BASE}/videos/${videoId}/captions/${clipId}`
}
