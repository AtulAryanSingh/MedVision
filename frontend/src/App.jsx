import { useState } from 'react'
import Home          from './pages/Home.jsx'
import DataManager   from './pages/DataManager.jsx'
import Workspace     from './pages/Workspace.jsx'
import QCPlots       from './pages/QCPlots.jsx'
import Tools         from './pages/Tools.jsx'
import Patchify      from './pages/Patchify.jsx'
import DeepLearning  from './pages/DeepLearning.jsx'
import Downloads     from './pages/Downloads.jsx'
import Registration  from './pages/Registration.jsx'
import Login         from './pages/Login.jsx'
import Signup        from './pages/Signup.jsx'
import { getStoredAuth, saveAuth, clearAuth } from './auth.js'

const NAV = [
  { id: 'home',      icon: '🏠', label: 'Home',          section: 'main' },
  { id: 'data',      icon: '📂', label: 'Data Manager',  section: 'main' },
  { id: 'workspace', icon: '🔬', label: 'Workspace',     section: 'viewer' },
  { id: 'qc',        icon: '📊', label: 'QC & Plots',    section: 'viewer' },
  { id: 'tools',     icon: '⚗️',  label: 'Tools',        section: 'tools' },
  { id: 'patchify',  icon: '🧩', label: 'Patchify 3D',   section: 'tools' },
  { id: 'register',  icon: '📐', label: 'Registration',  section: 'tools' },
  { id: 'dl',        icon: '🧬', label: 'Deep Learning', section: 'tools' },
  { id: 'exports',   icon: '📥', label: 'Downloads',     section: 'exports' },
]

const SECTION_LABELS = { main: 'Explorer', viewer: 'Viewer', tools: 'Tools', exports: 'Export' }

function AuthGate({ mode, onSwitchMode, onAuthSuccess }) {
  return (
    <div style={{ minHeight: '100vh', display: 'grid', placeItems: 'center', padding: '1.5rem' }}>
      <div style={{ width: '100%', maxWidth: 820, display: 'grid', gridTemplateColumns: '1fr', gap: '1.25rem' }}>
        <div className="card">
          <div className="card-body" style={{ padding: '1.25rem 1.5rem' }}>
            <h1 style={{ fontSize: '1.5rem', fontWeight: 800, letterSpacing: '-.5px' }}>🔬 MedVision</h1>
            <p className="text-muted" style={{ marginTop: '.35rem' }}>
              Sign in to access protected imaging APIs (upload, preview, MPR, processing, exports).
            </p>
          </div>
        </div>
        {mode === 'signup' ? (
          <Signup onSuccess={onAuthSuccess} onSwitchToLogin={() => onSwitchMode('login')} />
        ) : (
          <Login onSuccess={onAuthSuccess} onSwitchToSignup={() => onSwitchMode('signup')} />
        )}
      </div>
    </div>
  )
}

export default function App() {
  const [auth, setAuth] = useState(() => getStoredAuth())
  const [authMode, setAuthMode] = useState('login')
  const [page,     setPage]     = useState('home')
  const [imageId,  setImageId]  = useState(null)
  const [metadata, setMetadata] = useState(null)

  function handleReset() {
    setImageId(null)
    setMetadata(null)
    setPage('data')
  }

  function handleUpload(id, meta) {
    setImageId(id)
    setMetadata(meta)
    setPage('workspace')
  }

  function handleAuthSuccess({ token, username }) {
    saveAuth(token, username)
    setAuth({ token, username })
    setPage('data')
  }

  function handleLogout() {
    clearAuth()
    setAuth({ token: null, username: null })
    handleReset()
    setAuthMode('login')
  }

  if (!auth.token) {
    return <AuthGate mode={authMode} onSwitchMode={setAuthMode} onAuthSuccess={handleAuthSuccess} />
  }

  function renderPage() {
    const props = { imageId, metadata }
    switch (page) {
      case 'home':      return <Home         onNavigate={setPage} />
      case 'data':      return <DataManager  {...props} onUpload={handleUpload} onReset={handleReset} />
      case 'workspace': return <Workspace    {...props} />
      case 'qc':        return <QCPlots      {...props} />
      case 'tools':     return <Tools        {...props} />
      case 'patchify':  return <Patchify     {...props} />
      case 'register':  return <Registration {...props} />
      case 'dl':        return <DeepLearning {...props} />
      case 'exports':   return <Downloads    {...props} />
      default:          return <Home         onNavigate={setPage} />
    }
  }

  const activeNav = NAV.find(n => n.id === page)
  const sections  = [...new Set(NAV.map(n => n.section))]

  return (
    <div style={{ display: 'flex', minHeight: '100vh' }}>
      <aside className="sidebar">
        <div className="sidebar-brand">
          <span className="brand-icon">🔬</span>
          <span className="brand-name">MedVision</span>
          <span className="brand-badge">v2</span>
        </div>
        <nav style={{ flex: 1, overflowY: 'auto' }}>
          {sections.map((sec, si) => {
            const items = NAV.filter(n => n.section === sec)
            return (
              <div key={sec} className="sidebar-section">
                <div className="sidebar-section-label">{SECTION_LABELS[sec]}</div>
                {items.map(item => (
                  <button
                    key={item.id}
                    className={`nav-item${page === item.id ? ' active' : ''}`}
                    onClick={() => setPage(item.id)}
                  >
                    <span className="nav-icon">{item.icon}</span>
                    <span className="nav-label">{item.label}</span>
                  </button>
                ))}
                {si < sections.length - 1 && <div className="nav-divider" />}
              </div>
            )
          })}
        </nav>
        <div className="sidebar-footer">Medical Imaging Platform</div>
        <div style={{ padding: '.7rem 1rem', borderTop: '1px solid var(--border)' }}>
          {auth.username && <div className="text-muted" style={{ marginBottom: '.4rem' }}>Signed in as <strong>{auth.username}</strong></div>}
          <button className="btn btn-outline btn-sm w-full" onClick={handleLogout}>Log out</button>
        </div>
      </aside>

      <div className="app-content">
        <header className="topbar">
          <span className="topbar-title">{activeNav?.icon}&nbsp; {activeNav?.label}</span>
          <div className="topbar-chips">
            {imageId ? (
              <>
                <span className="chip blue">
                  {metadata?.modality !== 'unknown' ? metadata?.modality : metadata?.file_type?.toUpperCase()}
                </span>
                <span className="chip">{metadata?.shape?.join(' × ')} px</span>
                {metadata?.spacing && (
                  <span className="chip teal">{metadata.spacing.map(s => s.toFixed(2)).join('×')} mm</span>
                )}
                {metadata?.is_3d && <span className="chip green">3-D Volume</span>}
              </>
            ) : (
              <span className="chip">No image loaded</span>
            )}
          </div>
        </header>
        <main style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
          {renderPage()}
        </main>
      </div>
    </div>
  )
}
