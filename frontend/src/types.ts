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

// ---------------------------------------------------------------------------
// Dashboard decision-tool types
// ---------------------------------------------------------------------------

export type AlertLevel = 'critical' | 'warning' | 'info'
export type TrendDir = 'up' | 'down' | 'flat'

export interface DashboardAlert {
  level: AlertLevel
  code: string
  label: string
  message: string
  detail: string
}

export interface SeverityEntry {
  label: Label
  severity: number
  count: number
  recent_count: number
  prior_count: number
  avg_confidence: number | null
  trend: TrendDir
}

export interface TrendPoint {
  date: string
  pothole: number
  crack: number
  normal: number
  manhole: number
}

export interface DashboardSummary {
  total_labeled: number
  by_class: Record<string, number>
  correction_rate: number
  model_version: string
}

export interface DashboardStats {
  summary: DashboardSummary
  severity_ranking: SeverityEntry[]
  trends: TrendPoint[]
  alerts: DashboardAlert[]
}

export interface QueueItem {
  source_id: string
  image_path?: string | null
  image_url?: string | null
  prediction?: string | null
  confidence?: number | null
  model_version?: string | null
  timestamp?: string | null
}

export interface QueueResponse {
  count: number
  items: QueueItem[]
}

export interface PipelineProgress {
  completed_stages: number
  total_stages: number
  percent: number
}

export interface PipelineMetrics {
  images_processed: number
  rejected_images: number
  auto_label_count: number
  active_learning_count: number
}

export interface PipelineStatus {
  run_id?: string
  status: string
  current_stage?: string | null
  progress: PipelineProgress
  metrics: PipelineMetrics
  stages: Record<string, unknown>
  started_at?: string | null
  finished_at?: string | null
  error?: string | null
}
