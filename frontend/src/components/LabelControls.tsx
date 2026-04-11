import type { Label } from '../types'

const LABEL_CONFIG: { label: Label; keyHint: string; color: string }[] = [
  { label: 'pothole', keyHint: '1', color: 'bg-red-600/85 hover:bg-red-500' },
  { label: 'crack', keyHint: '2', color: 'bg-amber-600/85 hover:bg-amber-500' },
  { label: 'normal', keyHint: '3', color: 'bg-emerald-600/85 hover:bg-emerald-500' },
  { label: 'manhole', keyHint: '4', color: 'bg-sky-600/85 hover:bg-sky-500' },
]

interface LabelControlsProps {
  onLabel: (label: Label) => void
}

export function LabelControls({ onLabel }: LabelControlsProps) {
  return (
    <div className="grid grid-cols-2 gap-2">
      {LABEL_CONFIG.map((entry) => (
        <button
          key={entry.label}
          type="button"
          onClick={() => onLabel(entry.label)}
          className={`rounded-lg px-3 py-2 text-left text-sm font-semibold text-white transition ${entry.color}`}
        >
          <span className="block capitalize">{entry.label}</span>
          <span className="text-xs text-white/80">Key {entry.keyHint}</span>
        </button>
      ))}
    </div>
  )
}
