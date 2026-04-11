export type Label = 'pothole' | 'crack' | 'normal' | 'manhole'
export type PredictedLabel = Label | 'uncertain'

export type ImageSource = 'local' | 'server' | 'stream'

export interface DatasetImage {
  id: string
  name: string
  imageUrl: string
  sourcePath?: string
  source: ImageSource
  file?: File
  predictedLabel?: PredictedLabel
  confidence?: number
  modelVersion?: string
}

export interface PredictionResult {
  label: PredictedLabel
  confidence: number
  model_version?: string
}

export interface StreamEventMessage {
  type: 'new_image'
  image_url: string
  prediction: PredictionResult
}

export interface LabeledEventMessage {
  type: 'labeled'
  filename: string
}

export type WsMessage = StreamEventMessage | LabeledEventMessage

export const LABELS: Label[] = ['pothole', 'crack', 'normal', 'manhole']
