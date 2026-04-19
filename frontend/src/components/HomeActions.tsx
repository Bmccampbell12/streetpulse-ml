import type { ChangeEvent } from 'react'

interface HomeActionsProps {
  onSelectLocalFolder: (event: ChangeEvent<HTMLInputElement>) => void
  onSelectSdFolder: () => void
  onViewDataset: () => void
  onOpenDashboard: () => void
  onOpenOpsCenter: () => void
}

export function HomeActions({
  onSelectLocalFolder,
  onSelectSdFolder,
  onViewDataset,
  onOpenDashboard,
  onOpenOpsCenter,
}: HomeActionsProps) {
  return (
    <div className="min-h-screen bg-slate-950 text-slate-100">
      <div className="mx-auto flex min-h-screen max-w-4xl flex-col justify-center px-6 py-12">
        <h1 className="font-[Space_Grotesk] text-4xl font-semibold tracking-tight text-white md:text-6xl">
          StreetPulse Dataset Control Panel
        </h1>
        <p className="mt-4 max-w-2xl text-slate-400">
          Local-first image curation, labeling, and inference preview for rapid dataset operations.
        </p>

        <div className="mt-10 grid gap-4 md:grid-cols-3">
          <label className="cursor-pointer rounded-xl border border-slate-700 bg-slate-900/60 p-5 text-left transition hover:border-cyan-400 hover:bg-slate-900">
            <span className="block text-lg font-semibold">Select Local Folder</span>
            <span className="mt-1 block text-sm text-slate-400">Pick a local image directory.</span>
            <input
              type="file"
              multiple
              accept="image/*"
              className="hidden"
              // @ts-expect-error Non-standard folder selection API used by Chromium browsers.
              webkitdirectory=""
              onChange={onSelectLocalFolder}
            />
          </label>

          <button
            type="button"
            onClick={onSelectSdFolder}
            className="rounded-xl border border-slate-700 bg-slate-900/60 p-5 text-left transition hover:border-cyan-400 hover:bg-slate-900"
          >
            <span className="block text-lg font-semibold">Scan SD Card</span>
            <span className="mt-1 block text-sm text-slate-400">Ingest images from mounted SD path (for example D:/ride_002).</span>
          </button>

          <button
            type="button"
            onClick={onViewDataset}
            className="rounded-xl border border-slate-700 bg-slate-900/60 p-5 text-left transition hover:border-cyan-400 hover:bg-slate-900"
          >
            <span className="block text-lg font-semibold">Sort &amp; Label</span>
            <span className="mt-1 block text-sm text-slate-400">Load curated images from backend storage.</span>
          </button>

          <button
            type="button"
            onClick={onOpenDashboard}
            className="rounded-xl border border-cyan-800 bg-cyan-950/40 p-5 text-left transition hover:border-cyan-400 hover:bg-cyan-950/70"
          >
            <span className="block text-lg font-semibold text-cyan-200">Decision Dashboard</span>
            <span className="mt-1 block text-sm text-slate-400">
              Trends, alerts, and severity rankings from labeled data.
            </span>
          </button>

          <button
            type="button"
            onClick={onOpenOpsCenter}
            className="rounded-xl border border-emerald-800 bg-emerald-950/30 p-5 text-left transition hover:border-emerald-400 hover:bg-emerald-950/60"
          >
            <span className="block text-lg font-semibold text-emerald-200">Labeling Ops Center</span>
            <span className="mt-1 block text-sm text-slate-400">
              Active-learning queue, auto-label review, and live pipeline status.
            </span>
          </button>
        </div>
      </div>
    </div>
  )
}
