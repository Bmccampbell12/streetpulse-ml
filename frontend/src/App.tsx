import type { ChangeEvent } from 'react'
import { HomeActions } from './components/HomeActions'
import { DashboardPage } from './pages/DashboardPage'
import { OpsCenterPage } from './pages/OpsCenterPage'
import { SortingPage } from './pages/SortingPage'
import { fetchImages, imageUrlForPath, ingestSd } from './services/api'
import { useDatasetStore } from './store/useDatasetStore'
import type { DatasetImage } from './types'

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

  const handleSdIngest = async () => {
    const pathInput = window.prompt('Enter mounted SD folder path', 'D:/ride_002')
    if (pathInput === null) {
      return
    }

    const sdPath = pathInput.trim()
    if (!sdPath) {
      window.alert('Path is required to scan SD card.')
      return
    }

    try {
      const ingest = await ingestSd(sdPath)
      const rows = await fetchImages('raw')
      const mapped: DatasetImage[] = rows.map((row) => ({
        id: row.path,
        name: row.name,
        source: 'server',
        sourcePath: row.path,
        imageUrl: imageUrlForPath(row.path),
      }))
      setImages(mapped)
      setScreen('sorting')
      window.alert(`Imported ${ingest.copied} image(s) from ${ingest.source_path ?? sdPath}.`)
    } catch {
      window.alert('SD ingest failed. Verify backend is running and the path exists.')
    }
  }

  if (screen === 'sorting') {
    return <SortingPage onBack={() => setScreen('home')} />
  }

  if (screen === 'dashboard') {
    return <DashboardPage onBack={() => setScreen('home')} />
  }

  if (screen === 'ops') {
    return <OpsCenterPage onBack={() => setScreen('home')} />
  }

  return (
    <HomeActions
      onSelectLocalFolder={handleFolderSelection}
      onSelectSdFolder={() => void handleSdIngest()}
      onViewDataset={() => setScreen('sorting')}
      onOpenDashboard={() => setScreen('dashboard')}
      onOpenOpsCenter={() => setScreen('ops')}
    />
  )
}

export default App
