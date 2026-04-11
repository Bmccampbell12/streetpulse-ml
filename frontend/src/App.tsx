import type { ChangeEvent } from 'react'
import { HomeActions } from './components/HomeActions'
import { SortingPage } from './pages/SortingPage'
import { useDatasetStore } from './store/useDatasetStore'

function App() {
  const { screen, setScreen, setImages } = useDatasetStore()

  const handleFolderSelection = (event: ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files
    if (!files?.length) {
      return
    }

    const mapped = Array.from(files)
      .filter((file) => file.type.startsWith('image/'))
      .map((file) => ({
        id: `${file.name}-${file.size}-${file.lastModified}`,
        name: file.name,
        imageUrl: URL.createObjectURL(file),
        source: 'local' as const,
        file,
      }))

    setImages(mapped)
    setScreen('sorting')
  }

  if (screen === 'sorting') {
    return <SortingPage onBack={() => setScreen('home')} />
  }

  return (
    <HomeActions
      onSelectLocalFolder={handleFolderSelection}
      onSelectSdFolder={handleFolderSelection}
      onViewDataset={() => setScreen('sorting')}
    />
  )
}

export default App
