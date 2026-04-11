import { LabelControls } from './LabelControls'
import type { DatasetImage, Label } from '../types'

interface PreviewPanelProps {
  image?: DatasetImage
  loading: boolean
  onLabel: (label: Label) => void
  onAutoLabelAll: () => void
  confidenceThreshold: number
  onThresholdChange: (value: number) => void
}

export function PreviewPanel({
  image,
  loading,
  onLabel,
  onAutoLabelAll,
  confidenceThreshold,
  onThresholdChange,
}: PreviewPanelProps) {
  return (
    <aside className="flex h-full flex-col gap-4 rounded-2xl border border-slate-800 bg-slate-900/60 p-4">
      <div className="rounded-xl border border-slate-800 bg-slate-950 p-2">
        {image ? (
          <img src={image.imageUrl} alt={image.name} className="h-64 w-full rounded-lg object-contain" />
        ) : (
          <div className="flex h-64 items-center justify-center text-slate-500">No image selected</div>
        )}
      </div>

      <div className="rounded-xl border border-slate-800 bg-slate-950/70 p-3 text-sm">
        <p className="text-slate-400">Prediction Suggestion</p>
        <p className="mt-1 text-lg font-semibold text-white capitalize">{image?.predictedLabel ?? 'Pending...'}</p>
        <p className="text-slate-300">
          Confidence:{' '}
          {typeof image?.confidence === 'number' ? `${Math.round(image.confidence * 100)}%` : '--'}
        </p>
        {loading && <p className="mt-2 text-xs text-cyan-300">Running inference...</p>}
      </div>

      <LabelControls onLabel={onLabel} />

      <div className="rounded-xl border border-slate-800 bg-slate-950/70 p-3 text-sm">
        <div className="flex items-center justify-between">
          <span className="text-slate-300">Auto-label threshold</span>
          <span className="font-semibold text-cyan-300">{Math.round(confidenceThreshold * 100)}%</span>
        </div>
        <input
          type="range"
          min={0.5}
          max={0.99}
          step={0.01}
          value={confidenceThreshold}
          onChange={(event) => onThresholdChange(Number(event.target.value))}
          className="mt-2 w-full"
        />
        <button
          type="button"
          onClick={onAutoLabelAll}
          className="mt-3 w-full rounded-lg border border-cyan-500/60 bg-cyan-500/10 px-3 py-2 font-semibold text-cyan-200 transition hover:bg-cyan-500/20"
        >
          Auto-label all above threshold
        </button>
      </div>
    </aside>
  )
}
