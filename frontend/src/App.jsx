import { useState } from 'react'
import UploadViewer    from './components/tabs/UploadViewer.jsx'
import ProcessingLab   from './components/tabs/ProcessingLab.jsx'
import FeatureExplorer from './components/tabs/FeatureExplorer.jsx'
import MLLab           from './components/tabs/MLLab.jsx'
import AnalysisReport  from './components/tabs/AnalysisReport.jsx'

const TABS = [
  { id: 'upload',     icon: '📤', label: 'Upload & Viewer'  },
  { id: 'processing', icon: '⚗️',  label: 'Processing Lab'  },
  { id: 'features',   icon: '📊', label: 'Feature Explorer' },
  { id: 'ml',         icon: '🤖', label: 'ML Lab'           },
  { id: 'report',     icon: '📋', label: 'Analysis Report'  },
]

export default function App() {
  const [activeTab,  setActiveTab]  = useState('upload')
  const [imageId,    setImageId]    = useState(null)
  const [metadata,   setMetadata]   = useState(null)

  function handleUpload(id, meta) {
    setImageId(id)
    setMetadata(meta)
    setActiveTab('processing')
  }

  const tabProps = { imageId, metadata }

  return (
    <div className="app">
      {/* ── Header ── */}
      <header className="header">
        <div className="brand">
          <span className="brand-icon">🔬</span>
          <span className="brand-name">MedVision</span>
          <span className="brand-badge">v1.0</span>
        </div>
        {imageId && (
          <div className="header-meta">
            <span className="chip">
              {metadata?.modality !== 'unknown' ? metadata?.modality : metadata?.file_type?.toUpperCase()}
            </span>
            <span className="chip dim">{metadata?.shape?.join(' × ')} px</span>
            {metadata?.is_3d && <span className="chip accent">3-D Volume</span>}
          </div>
        )}
      </header>

      {/* ── Tab navigation ── */}
      <nav className="tab-nav" role="tablist">
        {TABS.map(t => (
          <button
            key={t.id}
            role="tab"
            aria-selected={activeTab === t.id}
            className={`tab-btn${activeTab === t.id ? ' active' : ''}${!imageId && t.id !== 'upload' ? ' disabled' : ''}`}
            onClick={() => (imageId || t.id === 'upload') && setActiveTab(t.id)}
            title={!imageId && t.id !== 'upload' ? 'Upload an image first' : undefined}
          >
            <span className="tab-icon">{t.icon}</span>
            <span className="tab-label">{t.label}</span>
          </button>
        ))}
      </nav>

      {/* ── Tab panels ── */}
      <main className="main">
        {activeTab === 'upload'     && <UploadViewer    {...tabProps} onUpload={handleUpload} />}
        {activeTab === 'processing' && <ProcessingLab   {...tabProps} />}
        {activeTab === 'features'   && <FeatureExplorer {...tabProps} />}
        {activeTab === 'ml'         && <MLLab           {...tabProps} />}
        {activeTab === 'report'     && <AnalysisReport  {...tabProps} />}
      </main>
    </div>
  )
}
