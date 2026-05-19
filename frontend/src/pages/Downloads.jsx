/**
 * pages/Downloads.jsx – Export centre (PNG / NPY / CSV)
 */
import { useState } from 'react'
import { api } from '../api/client.js'

export default function Downloads({ imageId }) {
  const [npyStatus, setNpyStatus] = useState(null) // null | 'loading' | 'done'
  const [npyError,  setNpyError]  = useState(null)

  function download(url, filename) {
    const a = Object.assign(document.createElement('a'), { href: url, download: filename })
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
  }

  async function downloadNpy() {
    if (!imageId) return
    setNpyStatus('loading'); setNpyError(null)
    try {
      download(api.exportUrl.npyStream(imageId), `medvision_${imageId.slice(0, 8)}.npy`)
      setNpyStatus('done')
    } catch (e) {
      setNpyError(e.message); setNpyStatus(null)
    }
  }

  const CARDS = [
    {
      icon: '🖼️',
      title: 'PNG Image',
      desc: 'Middle axial slice (or 2-D image) normalised to 8-bit and saved as PNG.',
      action: () => download(api.exportUrl.png(imageId), `medvision_${imageId?.slice(0, 8)}.png`),
      label: '⬇ Download PNG',
      disabled: !imageId,
    },
    {
      icon: '🔢',
      title: 'NumPy Array (.npy)',
      desc: 'Full float32 array in native NumPy format. Load with np.load(). For 3-D volumes the complete volume is included.',
      action: downloadNpy,
      label: npyStatus === 'loading' ? 'Preparing…' : '⬇ Download .npy',
      disabled: !imageId || npyStatus === 'loading',
      loading: npyStatus === 'loading',
    },
    {
      icon: '📋',
      title: 'Component Metrics (.csv)',
      desc: 'Connected-component measurements: label, area in pixels, area in mm² (using voxel spacing), centroid, and bounding box.',
      action: () => download(api.exportUrl.csv(imageId), `medvision_metrics_${imageId?.slice(0, 8)}.csv`),
      label: '⬇ Download CSV',
      disabled: !imageId,
    },
  ]

  return (
    <div className="page">
      <div className="page-header">
        <h2 className="page-title">📥 Downloads</h2>
        <p className="page-desc">
          Export the loaded image and computed metrics in standard formats for use in
          external tools, Python notebooks, and spreadsheets.
        </p>
      </div>

      {!imageId && (
        <div className="empty-state">
          <div className="empty-icon">📥</div>
          <div className="empty-title">No image loaded</div>
          <div className="empty-hint">Upload an image in Data Manager first.</div>
        </div>
      )}

      {imageId && (
        <>
          <div className="downloads-grid">
            {CARDS.map(c => (
              <div key={c.title} className="dl-card">
                <div className="dl-card-icon">{c.icon}</div>
                <div className="dl-card-title">{c.title}</div>
                <div className="dl-card-desc">{c.desc}</div>
                <button className="btn btn-outline" onClick={c.action} disabled={c.disabled}>
                  {c.loading ? <><span className="spinner" /> {c.label}</> : c.label}
                </button>
              </div>
            ))}
          </div>

          {npyError && (
            <div className="alert alert-error" style={{ maxWidth: 460, marginTop: '1rem' }}>
              ⚠️ {npyError}
            </div>
          )}

          <div style={{ marginTop: '2rem' }}>
            <div className="section-label">Patch archive (.npz)</div>
            <div className="alert alert-info" style={{ maxWidth: 480 }}>
              💡 To download volume patches as a compressed .npz file, go to
              <strong> Patchify 3D</strong> and click "Download .npz" after extracting patches.
            </div>
          </div>

          {/* Python usage */}
          <div style={{ marginTop: '1.75rem' }}>
            <div className="section-label">Python usage</div>
            <pre className="code-block">
{`import numpy as np
import pandas as pd

# .npy – load full array
arr = np.load("medvision_*.npy")
print(arr.shape, arr.dtype)

# .csv – load component metrics
df = pd.read_csv("medvision_metrics_*.csv")
print(df.head())`}
            </pre>
          </div>
        </>
      )}
    </div>
  )
}
