// TypeScript interfaces matching the FastAPI backend contract.

export interface VideoUploadResponse {
  video_id: string
  filename: string
  trimmed: boolean
  note?: string
}

export interface VideoInfo {
  video_id: string
  filename: string
  stage: string
  [key: string]: unknown
}

export interface ProgressResponse {
  stage: string
  done: number
  total: number
  error: string | null
}

export interface Candidate {
  id: number
  start: number
  end: number
  score: number
  hook_line: string
  reason: string
  title: string
  hashtags: string[]
  speaker: string | null
  genre: string
  description: string
  thumbnail_concept: string
}

export interface ClipRecord {
  id: string
  video_id: string
  candidate: Candidate
  status: string
  output_path: string | null
  caption_path: string | null
  srt_path: string | null
  render_info: Record<string, unknown>
  error: string | null
}

export interface RerenderRequest {
  start?: number
  end?: number
  title?: string
}
