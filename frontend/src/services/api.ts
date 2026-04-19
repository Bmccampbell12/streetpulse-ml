import axios from 'axios'
import type {
  DashboardStats,
  Label,
  PipelineStatus,
  PredictedLabel,
  PredictionResult,
  QueueResponse,
} from '../types'

const API_BASE_URL = import.meta.env.VITE_API_URL ?? 'http://127.0.0.1:8000'
const ADMIN_API_KEY = import.meta.env.VITE_ADMIN_KEY ?? ''

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
})

function adminHeaders() {
  return ADMIN_API_KEY ? { 'x-admin-key': ADMIN_API_KEY } : {}
}

export interface ServerImageItem {
  name: string
  path: string
  folder: string
}

export interface SdIngestResponse {
  status: string
  source_path: string | null
  copied: number
}

export async function checkHealth(): Promise<{ modelAvailable: boolean }> {
  try {
    const { data } = await api.get<{ status: string; model: string }>('/health')
    return { modelAvailable: data.model === 'loaded' }
  } catch {
    return { modelAvailable: false }
  }
}

export async function predictImage(file: File): Promise<PredictionResult> {
  const form = new FormData()
  form.append('file', file)

  const { data } = await api.post<PredictionResult>('/predict', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
  })
  return data
}

export async function fetchImages(source: 'raw' | 'curated' | 'labeled' = 'curated'): Promise<ServerImageItem[]> {
  const { data } = await api.get<{ images: ServerImageItem[] }>('/images', {
    params: { source },
  })
  return data.images
}

export async function ingestSd(path?: string): Promise<SdIngestResponse> {
  const { data } = await api.post<SdIngestResponse>(
    '/ingest/sd',
    {
      path: path?.trim() || null,
    },
    {
      headers: adminHeaders(),
    },
  )
  return data
}

export async function labelStreamImage(
  filename: string,
  label: Label,
  prediction?: { predictedLabel?: PredictedLabel; confidence?: number; modelVersion?: string },
): Promise<void> {
  await api.post(
    '/label',
    {
      filename,
      label,
      predicted_label: prediction?.predictedLabel,
      confidence: prediction?.confidence,
      model_version: prediction?.modelVersion,
    },
    {
      headers: adminHeaders(),
    },
  )
}

export function imageUrlForPath(path: string): string {
  const encoded = encodeURIComponent(path)
  return `${API_BASE_URL}/images/file?path=${encoded}`
}

export function streamWebSocketUrl(): string {
  const httpUrl = new URL(API_BASE_URL)
  const scheme = httpUrl.protocol === 'https:' ? 'wss:' : 'ws:'
  return `${scheme}//${httpUrl.host}/ws/stream`
}

export function toAbsoluteImageUrl(imageUrl: string): string {
  if (imageUrl.startsWith('http://') || imageUrl.startsWith('https://')) {
    return imageUrl
  }
  return `${API_BASE_URL}${imageUrl}`
}

export function extractFilenameFromImageUrl(imageUrl: string): string {
  const rawPath = imageUrl.split('?')[0]
  const part = rawPath.split('/').pop()?.trim() ?? ''
  return decodeURIComponent(part)
}

export async function fetchDashboardStats(): Promise<DashboardStats> {
  const { data } = await api.get<DashboardStats>('/dashboard/stats')
  return data
}

export async function fetchActiveQueue(): Promise<QueueResponse> {
  const { data } = await api.get<QueueResponse>('/queue/active')
  return data
}

export async function fetchAutoLabelQueue(): Promise<QueueResponse> {
  const { data } = await api.get<QueueResponse>('/queue/auto-label')
  return data
}

export async function labelQueueItem(args: {
  filename: string
  label: Label
  sourceId?: string
  imagePath?: string | null
  prediction?: { predictedLabel?: PredictedLabel; confidence?: number; modelVersion?: string }
}): Promise<void> {
  await api.post('/label', {
    filename: args.filename,
    label: args.label,
    source_id: args.sourceId,
    path: args.imagePath,
    predicted_label: args.prediction?.predictedLabel,
    confidence: args.prediction?.confidence,
    model_version: args.prediction?.modelVersion,
  })
}

export async function fetchLatestPipelineStatus(): Promise<PipelineStatus> {
  const { data } = await api.get<PipelineStatus>('/pipeline/status/latest')
  return data
}

export async function fetchPipelineStatus(runId: string): Promise<PipelineStatus> {
  const { data } = await api.get<PipelineStatus>(`/pipeline/status/${encodeURIComponent(runId)}`)
  return data
}
