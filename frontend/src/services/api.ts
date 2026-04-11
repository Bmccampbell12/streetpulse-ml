import axios from 'axios'
import type { Label, PredictionResult } from '../types'

const API_BASE_URL = import.meta.env.VITE_API_URL ?? 'http://127.0.0.1:8000'

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
})

export interface ServerImageItem {
  name: string
  path: string
  folder: string
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

export async function labelStreamImage(filename: string, label: Label): Promise<void> {
  await api.post('/label', {
    filename,
    label,
  })
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
