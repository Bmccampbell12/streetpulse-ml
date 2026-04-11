import clsx from 'clsx'
import type { DatasetImage } from '../types'

interface ImageGridProps {
  images: DatasetImage[]
  selectedIndex: number
  onSelect: (index: number) => void
}

export function ImageGrid({ images, selectedIndex, onSelect }: ImageGridProps) {
  return (
    <div className="h-full overflow-y-auto rounded-2xl border border-slate-800 bg-slate-900/50 p-3">
      <div className="grid grid-cols-2 gap-3 md:grid-cols-3 xl:grid-cols-4">
        {images.map((image, index) => (
          <button
            type="button"
            key={image.id}
            onClick={() => onSelect(index)}
            className={clsx(
              'group relative overflow-hidden rounded-lg border transition',
              index === selectedIndex
                ? 'border-cyan-400 shadow-[0_0_0_1px_rgba(34,211,238,0.4)]'
                : 'border-slate-800 hover:border-slate-600',
            )}
          >
            <img
              loading="lazy"
              src={image.imageUrl}
              alt={image.name}
              className="aspect-square w-full object-cover"
            />
            <div className="absolute inset-x-0 bottom-0 bg-gradient-to-t from-black/80 to-transparent px-2 py-1 text-left text-xs text-slate-200">
              {image.name}
            </div>
          </button>
        ))}
      </div>
    </div>
  )
}
