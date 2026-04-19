import { useCallback, useEffect, useMemo, useState } from 'react'
import {
  fetchActiveQueue,
  fetchAutoLabelQueue,
  fetchLatestPipelineStatus,
  labelQueueItem,
  toAbsoluteImageUrl,
} from '../services/api'
import type { Label, PipelineStatus, QueueItem } from '../types'

interface OpsCenterPageProps {
  onBack: () => void
}

type Tab = 'active' | 'auto'

function queueFilename(item: QueueItem): string {
  const fromSource = (item.source_id ?? '').split('/').pop()
  const fromPath = (item.image_path ?? '').split('/').pop()
  return (fromSource || fromPath || `img-${Date.now()}`).trim()
}

function severityBadge(status: string | undefined): string {
  const s = (status || '').toLowerCase()
  if (s === 'failed') return 'bg-red-900/60 text-red-300 border-red-700'
  if (s === 'success') return 'bg-emerald-900/60 text-emerald-300 border-emerald-700'
  return 'bg-amber-900/60 text-amber-300 border-amber-700'
}

export function OpsCenterPage({ onBack }: OpsCenterPageProps) {
  const [tab, setTab] = useState<Tab>('active')
  const [activeQueue, setActiveQueue] = useState<QueueItem[]>([])
  const [autoQueue, setAutoQueue] = useState<QueueItem[]>([])
  const [activeIndex, setActiveIndex] = useState(0)
  const [autoIndex, setAutoIndex] = useState(0)
  const [pipeline, setPipeline] = useState<PipelineStatus | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const reload = useCallback(async () => {
    setLoading(true)
    try {
      const [active, auto, status] = await Promise.all([
        fetchActiveQueue(),
        fetchAutoLabelQueue(),
        fetchLatestPipelineStatus().catch(() => null),
      ])
      setActiveQueue(active.items)
      setAutoQueue(auto.items)
      setPipeline(status)
      setError(null)
      setActiveIndex((prev) => Math.min(prev, Math.max(active.items.length - 1, 0)))
      setAutoIndex((prev) => Math.min(prev, Math.max(auto.items.length - 1, 0)))
    } catch {
      setError('Failed to load queues or pipeline status. Check backend connection.')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    void reload()
    const id = setInterval(() => void reload(), 2000)
    return () => clearInterval(id)
  }, [reload])

  const activeItem = activeQueue[activeIndex]
  const autoItem = autoQueue[autoIndex]

  const selected = tab === 'active' ? activeItem : autoItem
  const selectedImage = selected?.image_url ? toAbsoluteImageUrl(selected.image_url) : undefined

  const doLabel = useCallback(
    async (item: QueueItem | undefined, label: Label) => {
      if (!item) return
      const filename = queueFilename(item)
      try {
        await labelQueueItem({
          filename,
          label,
          sourceId: item.source_id,
          imagePath: item.image_path,
          prediction: {
            predictedLabel: (item.prediction as Label | 'uncertain' | undefined) ?? undefined,
            confidence: typeof item.confidence === 'number' ? item.confidence : undefined,
            modelVersion: item.model_version ?? undefined,
          },
        })
        await reload()
      } catch {
        setError('Could not label this queue item. It may already be moved or no longer available. Refreshing queue...')
        await reload()
      }
    },
    [reload],
  )

  const onAutoAccept = useCallback(async () => {
    if (!autoItem) return
    const predicted = (autoItem.prediction || '').toLowerCase()
    const mapped: Label | null =
      predicted === 'pothole' || predicted === 'crack' || predicted === 'manhole' || predicted === 'normal'
        ? (predicted as Label)
        : null
    if (!mapped) return
    try {
      await doLabel(autoItem, mapped)
    } catch {
      setError('Auto-accept failed for this item. It may be stale; queue was refreshed.')
    }
  }, [autoItem, doLabel])

  const onAutoReject = useCallback(() => {
    // Send to active-learning tab for manual choice; keep it visible there this session.
    if (!autoItem) return
    setActiveQueue((prev) => [autoItem, ...prev.filter((x) => x.source_id !== autoItem.source_id)])
    setAutoQueue((prev) => prev.filter((x) => x.source_id !== autoItem.source_id))
    setTab('active')
    setActiveIndex(0)
  }, [autoItem])

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (tab !== 'active' || !activeItem) return
      if (event.key === '1') void doLabel(activeItem, 'pothole')
      if (event.key === '2') void doLabel(activeItem, 'crack')
      if (event.key === '3') void doLabel(activeItem, 'normal')
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [activeItem, doLabel, tab])

  const metrics = pipeline?.metrics
  const progressPct = useMemo(() => pipeline?.progress.percent ?? 0, [pipeline?.progress.percent])

  return (
    <div className="min-h-screen bg-[#090b12] text-slate-100">
      <header className="border-b border-slate-800 bg-slate-950/60 px-4 py-3 backdrop-blur">
        <div className="mx-auto flex max-w-6xl items-center gap-3">
          <button
            type="button"
            onClick={onBack}
            className="rounded-lg border border-slate-700 px-3 py-1.5 text-sm hover:border-slate-500"
          >
            Home
          </button>
          <h1 className="flex-1 text-lg font-semibold tracking-tight">Ops Center: Labeling + Live Pipeline</h1>
          <button
            type="button"
            onClick={() => void reload()}
            className="rounded-lg border border-slate-700 px-3 py-1.5 text-xs hover:border-slate-500"
          >
            Refresh
          </button>
        </div>
      </header>

      <main className="mx-auto grid max-w-6xl gap-4 px-4 py-4 lg:grid-cols-[1.2fr_1fr]">
        <section className="rounded-xl border border-slate-800 bg-slate-900/40 p-4">
          <div className="mb-3 flex items-center justify-between">
            <div className="flex gap-2">
              <button
                type="button"
                onClick={() => setTab('active')}
                className={`rounded px-3 py-1.5 text-sm ${tab === 'active' ? 'bg-cyan-700/40 text-cyan-200' : 'bg-slate-800 text-slate-300'}`}
              >
                Active Learning ({activeQueue.length})
              </button>
              <button
                type="button"
                onClick={() => setTab('auto')}
                className={`rounded px-3 py-1.5 text-sm ${tab === 'auto' ? 'bg-cyan-700/40 text-cyan-200' : 'bg-slate-800 text-slate-300'}`}
              >
                Auto Label ({autoQueue.length})
              </button>
            </div>
            {loading && <span className="text-xs text-slate-500">Updating...</span>}
          </div>

          {error && <div className="mb-3 rounded border border-red-800 bg-red-950/50 px-3 py-2 text-sm text-red-300">{error}</div>}

          {!selected ? (
            <div className="flex h-[420px] items-center justify-center rounded border border-slate-800 text-slate-500">
              Queue is empty.
            </div>
          ) : (
            <div className="grid gap-3 md:grid-cols-[1fr_260px]">
              <div className="rounded border border-slate-800 bg-slate-950 p-2">
                {selectedImage ? (
                  <img src={selectedImage} alt={queueFilename(selected)} className="h-[410px] w-full rounded object-contain" />
                ) : (
                  <div className="flex h-[410px] items-center justify-center text-slate-500">Image unavailable</div>
                )}
              </div>
              <div className="space-y-3">
                <div className="rounded border border-slate-800 bg-slate-950/70 p-3 text-sm">
                  <p className="text-slate-400">Prediction</p>
                  <p className="mt-1 text-lg font-semibold capitalize text-white">{selected.prediction ?? 'unknown'}</p>
                  <p className="text-slate-300">Confidence: {typeof selected.confidence === 'number' ? `${Math.round(selected.confidence * 100)}%` : '--'}</p>
                  <p className="mt-1 text-xs text-slate-500 break-all">Source: {selected.source_id}</p>
                </div>

                {tab === 'active' ? (
                  <>
                    <button type="button" onClick={() => void doLabel(selected, 'pothole')} className="w-full rounded bg-red-600/90 px-3 py-2 text-sm font-semibold hover:bg-red-500">
                      1 - Pothole
                    </button>
                    <button type="button" onClick={() => void doLabel(selected, 'crack')} className="w-full rounded bg-amber-600/90 px-3 py-2 text-sm font-semibold hover:bg-amber-500">
                      2 - Crack
                    </button>
                    <button type="button" onClick={() => void doLabel(selected, 'normal')} className="w-full rounded bg-emerald-600/90 px-3 py-2 text-sm font-semibold hover:bg-emerald-500">
                      3 - No Defect
                    </button>
                  </>
                ) : (
                  <>
                    <button type="button" onClick={() => void onAutoAccept()} className="w-full rounded bg-cyan-700/80 px-3 py-2 text-sm font-semibold hover:bg-cyan-600">
                      Accept Prediction
                    </button>
                    <button type="button" onClick={onAutoReject} className="w-full rounded bg-slate-700 px-3 py-2 text-sm font-semibold hover:bg-slate-600">
                      Reject (Send to Active)
                    </button>
                  </>
                )}

                <div className="flex gap-2">
                  <button
                    type="button"
                    onClick={() => (tab === 'active' ? setActiveIndex((i) => Math.max(i - 1, 0)) : setAutoIndex((i) => Math.max(i - 1, 0)))}
                    className="flex-1 rounded border border-slate-700 px-3 py-1.5 text-sm hover:border-slate-500"
                  >
                    Prev
                  </button>
                  <button
                    type="button"
                    onClick={() =>
                      tab === 'active'
                        ? setActiveIndex((i) => Math.min(i + 1, Math.max(activeQueue.length - 1, 0)))
                        : setAutoIndex((i) => Math.min(i + 1, Math.max(autoQueue.length - 1, 0)))
                    }
                    className="flex-1 rounded border border-slate-700 px-3 py-1.5 text-sm hover:border-slate-500"
                  >
                    Next
                  </button>
                </div>
              </div>
            </div>
          )}
        </section>

        <section className="rounded-xl border border-slate-800 bg-slate-900/40 p-4">
          <h2 className="mb-3 text-sm font-semibold uppercase tracking-widest text-slate-400">Pipeline Status</h2>

          {!pipeline ? (
            <div className="rounded border border-slate-800 bg-slate-950/60 px-3 py-3 text-sm text-slate-500">No run state yet.</div>
          ) : (
            <>
              <div className="mb-3 flex items-center gap-2">
                <span className={`rounded border px-2 py-0.5 text-xs ${severityBadge(pipeline.status)}`}>{pipeline.status}</span>
                <span className="text-sm text-slate-300">Run: {pipeline.run_id ?? 'latest'}</span>
              </div>
              <p className="text-sm text-slate-300">Current stage: <span className="font-semibold text-cyan-200">{pipeline.current_stage ?? 'idle'}</span></p>

              <div className="mt-3 h-2 overflow-hidden rounded-full bg-slate-800">
                <div className="h-2 rounded-full bg-cyan-500 transition-all" style={{ width: `${progressPct}%` }} />
              </div>
              <p className="mt-1 text-xs text-slate-400">Progress: {progressPct.toFixed(1)}%</p>

              <div className="mt-4 grid grid-cols-2 gap-3 text-sm">
                <Metric label="Images Processed" value={metrics?.images_processed ?? 0} />
                <Metric label="Rejected Images" value={metrics?.rejected_images ?? 0} />
                <Metric label="Auto-Label Queue" value={metrics?.auto_label_count ?? 0} />
                <Metric label="Active Queue" value={metrics?.active_learning_count ?? 0} />
              </div>

              {pipeline.error && (
                <div className="mt-4 rounded border border-red-800 bg-red-950/50 px-3 py-2 text-xs text-red-300">
                  {pipeline.error}
                </div>
              )}
            </>
          )}
        </section>
      </main>
    </div>
  )
}

function Metric({ label, value }: { label: string; value: number }) {
  return (
    <div className="rounded border border-slate-800 bg-slate-950/60 px-3 py-2">
      <p className="text-xs text-slate-500 uppercase tracking-wide">{label}</p>
      <p className="mt-1 text-xl font-semibold text-white">{value}</p>
    </div>
  )
}
