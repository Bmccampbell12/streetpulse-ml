export type Label = 'pothole' | 'crack' | 'normal' | 'manhole'

export type ImageSource = 'local' | 'server' | 'stream'

export interface DatasetImage {
  id: string
  name: string
  imageUrl: string
  sourcePath?: string
  source: ImageSource
  file?: File
  predictedLabel?: Label
  confidence?: number
}

export interface PredictionResult {
  label: Label
  confidence: number
}

export interface StreamEventMessage {
  type: 'new_image'
  image_url: string
  prediction: PredictionResult
}

export const LABELS: Label[] = ['pothole', 'crack', 'normal', 'manhole']
