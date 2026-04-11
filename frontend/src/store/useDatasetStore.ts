import { create } from 'zustand'
import type { DatasetImage } from '../types'

type Screen = 'home' | 'sorting'

interface DatasetState {
  screen: Screen
  images: DatasetImage[]
  selectedIndex: number
  loading: boolean
  confidenceThreshold: number
  setScreen: (screen: Screen) => void
  setImages: (images: DatasetImage[]) => void
  setLoading: (loading: boolean) => void
  setSelectedIndex: (index: number) => void
  setConfidenceThreshold: (value: number) => void
  updateImage: (id: string, patch: Partial<DatasetImage>) => void
  removeImage: (id: string) => void
  clearImages: () => void
}

export const useDatasetStore = create<DatasetState>((set) => ({
  screen: 'home',
  images: [],
  selectedIndex: 0,
  loading: false,
  confidenceThreshold: 0.75,
  setScreen: (screen) => set({ screen }),
  setImages: (images) => set({ images, selectedIndex: images.length ? 0 : -1 }),
  setLoading: (loading) => set({ loading }),
  setSelectedIndex: (selectedIndex) => set({ selectedIndex }),
  setConfidenceThreshold: (confidenceThreshold) => set({ confidenceThreshold }),
  updateImage: (id, patch) =>
    set((state) => ({
      images: state.images.map((img) => (img.id === id ? { ...img, ...patch } : img)),
    })),
  removeImage: (id) =>
    set((state) => {
      const next = state.images.filter((img) => img.id !== id)
      const nextIndex = Math.min(Math.max(state.selectedIndex, 0), Math.max(next.length - 1, 0))
      return {
        images: next,
        selectedIndex: next.length ? nextIndex : -1,
      }
    }),
  clearImages: () => set({ images: [], selectedIndex: -1 }),
}))
