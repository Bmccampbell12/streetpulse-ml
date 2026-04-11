import { useCallback, useEffect, useMemo, useState } from 'react'
import {
  checkHealth,
  extractFilenameFromImageUrl,
  fetchImages,
  imageUrlForPath,
  labelStreamImage,
  predictImage,
  streamWebSocketUrl,
  toAbsoluteImageUrl,
} from '../services/api'
import { useDatasetStore } from '../store/useDatasetStore'
import type { DatasetImage, Label, WsMessage } from '../types'
import { LABELS } from '../types'
import { ImageGrid } from '../components/ImageGrid'
import { PreviewPanel } from '../components/PreviewPanel'

interface SortingPageProps {
  onBack: () => void
}

function asDatasetImages(files: FileList): DatasetImage[] {
  return Array.from(files)
    .filter((file) => file.type.startsWith('image/'))
    .map((file) => ({
      id: `${file.name}-${file.size}-${file.lastModified}`,
      name: file.name,
      imageUrl: URL.createObjectURL(file),
      source: 'local' as const,
      file,
    }))
}

export function SortingPage({ onBack }: SortingPageProps) {
  const {
    images,
    selectedIndex,
    loading,
    confidenceThreshold,
    setImages,
    setLoading,
    setSelectedIndex,
    updateImage,
    removeImage,
    setConfidenceThreshold,
  } = useDatasetStore()
  const [datasetSource, setDatasetSource] = useState<'curated' | 'raw' | 'labeled'>('curated')
  const [modelAvailable, setModelAvailable] = useState(false)

  useEffect(() => {
    checkHealth().then(({ modelAvailable: available }) => setModelAvailable(available))
  }, [])

  const selectedImage = images[selectedIndex]

  const loadBackendDataset = useCallback(
    async (source: 'curated' | 'raw' | 'labeled' = datasetSource) => {
      setLoading(true)
      try {
        const rows = await fetchImages(source)
        const mapped: DatasetImage[] = rows.map((row) => ({
          id: `${row.path}`,
          name: row.name,
          source: 'server',
          sourcePath: row.path,
          imageUrl: imageUrlForPath(row.path),
        }))
        setImages(mapped)
      } finally {
        setLoading(false)
      }
    },
    [datasetSource, setImages, setLoading],
  )

  const handleSelectFiles = useCallback(
    (files: FileList | null) => {
      if (!files?.length) {
        return
      }
      const mapped = asDatasetImages(files)
      setImages(mapped)
    },
    [setImages],
  )

  const predictCurrent = useCallback(async () => {
    // Skip: no image, already predicted, stream image, or model not loaded
    if (!selectedImage || selectedImage.predictedLabel || selectedImage.source === 'stream' || !modelAvailable) {
      return
    }

    setLoading(true)
    try {
      let fileToPredict: File
      if (selectedImage.file) {
        fileToPredict = selectedImage.file
      } else {
        const blob = await fetch(selectedImage.imageUrl).then((res) => res.blob())
        fileToPredict = new File([blob], selectedImage.name, { type: blob.type || 'image/jpeg' })
      }

      const result = await predictImage(fileToPredict)
      updateImage(selectedImage.id, {
        predictedLabel: result.label,
        confidence: result.confidence,
        modelVersion: result.model_version,
      })
    } catch {
      // 503 = model not loaded; other errors are transient — degrade silently.
    } finally {
      setLoading(false)
    }
  }, [modelAvailable, selectedImage, setLoading, updateImage])

  useEffect(() => {
    void predictCurrent()
  }, [predictCurrent])

  const navigate = useCallback(
    (direction: number) => {
      if (!images.length) {
        return
      }
      const next = Math.min(Math.max(selectedIndex + direction, 0), images.length - 1)
      setSelectedIndex(next)
    },
    [images.length, selectedIndex, setSelectedIndex],
  )

  // An image can be labeled if it lives in curated/stream on the server.
  // This covers live WS stream images (source='stream') and images loaded from
  // the backend 'curated' view that resolve to curated/stream/ (source='server').
  function isLabelable(image: DatasetImage): boolean {
    return (
      image.source === 'stream' ||
      (image.source === 'server' && (image.sourcePath?.startsWith('curated/stream/') ?? false))
    )
  }

  const labelImage = useCallback(
    async (label: Label) => {
      if (!selectedImage || !isLabelable(selectedImage)) {
        return
      }

      const toRemove = selectedImage.id
      removeImage(toRemove)

      try {
        // Use .name (reliable for all sources) rather than parsing the image URL.
        await labelStreamImage(selectedImage.name, label, {
          predictedLabel: selectedImage.predictedLabel,
          confidence: selectedImage.confidence,
          modelVersion: selectedImage.modelVersion,
        })
      } catch {
        const current = useDatasetStore.getState().images
        useDatasetStore.getState().setImages([selectedImage, ...current])
      }
    },
    [removeImage, selectedImage],
  )

  const autoLabelAll = useCallback(async () => {
    const streamCandidates = images.filter(
      (image) =>
        isLabelable(image) &&
        image.predictedLabel &&
        image.predictedLabel !== 'uncertain' &&
        image.confidence &&
        image.confidence >= confidenceThreshold,
    )

    for (const image of streamCandidates) {
      if (!image.predictedLabel || image.predictedLabel === 'uncertain') {
        continue
      }

      removeImage(image.id)
      void labelStreamImage(image.name, image.predictedLabel, {
        predictedLabel: image.predictedLabel,
        confidence: image.confidence,
        modelVersion: image.modelVersion,
      })
    }
  }, [confidenceThreshold, images, removeImage])

  useEffect(() => {
    let destroyed = false
    let ws: WebSocket | null = null
    let reconnectTimer: ReturnType<typeof setTimeout> | null = null

    function connect() {
      if (destroyed) return

      ws = new WebSocket(streamWebSocketUrl())

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data) as WsMessage

          if (data.type === 'new_image') {
            if (!data.image_url || !data.prediction) return

            const filename = extractFilenameFromImageUrl(data.image_url) || `stream-${Date.now()}`
            const next: DatasetImage = {
              id: `${filename}-${Date.now()}`,
              name: filename,
              imageUrl: toAbsoluteImageUrl(data.image_url),
              source: 'stream',
              predictedLabel: data.prediction.label,
              confidence: data.prediction.confidence,
              modelVersion: data.prediction.model_version,
            }

            const current = useDatasetStore.getState().images
            useDatasetStore.getState().setImages([next, ...current])
            return
          }

          if (data.type === 'labeled') {
            const current = useDatasetStore.getState().images
            // Use img.name which is reliable for both stream and server images.
            const updated = current.filter((img) => img.name !== data.filename)
            useDatasetStore.getState().setImages(updated)
          }
        } catch {
          // Ignore malformed websocket payloads.
        }
      }

      ws.onerror = () => {
        ws?.close()
      }

      ws.onclose = () => {
        if (destroyed) return
        reconnectTimer = setTimeout(connect, 3000)
      }
    }

    // Delay by one tick: StrictMode cleanup runs synchronously and cancels
    // this timer before the socket is ever created, suppressing the
    // "WebSocket closed before established" browser warning in dev.
    reconnectTimer = setTimeout(connect, 0)

    return () => {
      destroyed = true
      if (reconnectTimer !== null) clearTimeout(reconnectTimer)
      ws?.close()
    }
  }, [])

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'ArrowRight') {
        navigate(1)
      }
      if (event.key === 'ArrowLeft') {
        navigate(-1)
      }
      if (['1', '2', '3', '4'].includes(event.key)) {
        const index = Number(event.key) - 1
        const label = LABELS[index]
        void labelImage(label)
      }
    }

    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [labelImage, navigate])

  const stats = useMemo(() => {
    const predicted = images.filter((img) => img.predictedLabel).length
    return {
      total: images.length,
      predicted,
      pending: images.length - predicted,
    }
  }, [images])

  return (
    <div className="min-h-screen bg-[#090b12] text-slate-100">
      <header className="border-b border-slate-800 bg-slate-950/60 px-4 py-3 backdrop-blur">
        <div className="mx-auto flex max-w-[1500px] flex-wrap items-center gap-3">
          <button
            type="button"
            onClick={onBack}
            className="rounded-lg border border-slate-700 px-3 py-1.5 text-sm hover:border-slate-500"
          >
            Home
          </button>
          <label className="cursor-pointer rounded-lg border border-slate-700 px-3 py-1.5 text-sm hover:border-slate-500">
            Select Folder
            <input
              type="file"
              multiple
              accept="image/*"
              className="hidden"
              // @ts-expect-error Non-standard folder selection API used by Chromium browsers.
              webkitdirectory=""
              onChange={(event) => handleSelectFiles(event.target.files)}
            />
          </label>
          <div className="flex gap-2">
            {(['curated', 'raw', 'labeled'] as const).map((source) => (
              <button
                key={source}
                type="button"
                onClick={() => {
                  setDatasetSource(source)
                  void loadBackendDataset(source)
                }}
                className={`rounded-lg px-3 py-1.5 text-sm capitalize ${
                  source === datasetSource
                    ? 'bg-cyan-500/20 text-cyan-200'
                    : 'border border-slate-700 text-slate-300 hover:border-slate-500'
                }`}
              >
                {source}
              </button>
            ))}
          </div>
          <div className="ml-auto flex items-center gap-4 text-sm text-slate-400">
            {!modelAvailable && (
              <span className="rounded bg-yellow-900/60 px-2 py-0.5 text-xs text-yellow-300">
                model not loaded — predictions disabled
              </span>
            )}
            <span>Total: {stats.total}</span>
            <span>Predicted: {stats.predicted}</span>
            <span>Pending: {stats.pending}</span>
          </div>
        </div>
      </header>

      <main className="mx-auto grid max-w-[1500px] gap-4 px-4 py-4 lg:grid-cols-[1fr_380px]">
        <div className="h-[calc(100vh-110px)]">
          <ImageGrid images={images} selectedIndex={selectedIndex} onSelect={setSelectedIndex} />
        </div>
        <div className="h-[calc(100vh-110px)]">
          <PreviewPanel
            image={selectedImage}
            loading={loading}
            onLabel={(label) => void labelImage(label)}
            onAutoLabelAll={() => void autoLabelAll()}
            confidenceThreshold={confidenceThreshold}
            onThresholdChange={setConfidenceThreshold}
          />
        </div>
      </main>
    </div>
  )
}
