/**
 * pages/Patchify.jsx – 3D volume → patches + NPZ download
 */
import { useState } from 'react'
import { api } from '../api/client.js'

function estimatePatches(shape, ps, stride) {
  if (!shape || shape.length < 3) return null
  const [D, H, W] = shape
  const nz = Math.floor((D - ps) / stride) + 1
  const ny = Math.floor((H - ps) / stride) + 1
  const nx = Math.floor((W - ps) / stride) + 1
  if (nz <= 0 || ny <= 0 || nx <= 0) return 0
  return nz * ny * nx
}

export default function Patchify({ imageId, metadata }) {
  const [patchSize, setPatchSize] = useState(32)
  const [stride,    setStride]    = useState(16)
  const [result,    setResult]    = useState(null)
  const [loading,   setLoading]   = useState(false)
  const [error,     setError]     = useState(null)

  const shape   = metadata?.shape || []
  const is3d    = metadata?.is_3d || false
  const est     = is3d ? estimatePatches(shape, patchSize, stride) : null
  const overlap = patchSize - stride

  async function run() {
    if (!imageId) return
    setLoading(true); setError(null)
    try {
      const r = await api.patchify(imageId, patchSize, stride)
      setResult(r)
    } catch (e) { setError(e.message) }
    finally     { setLoading(false) }
  }

  function downloadNpz(npzB64, filename) {
    const bytes  = atob(npzB64)
    const buf    = new Uint8Array(bytes.length)
    for (let i = 0; i < bytes.length; i++) buf[i] = bytes.charCodeAt(i)
    const blob   = new Blob([buf], { type: 'application/octet-stream' })
    const url    = URL.createObjectURL(blob)
    const a      = Object.assign(document.createElement('a'), { href: url, download: filename })
    a.click()
    URL.revokeObjectURL(url)
  }

  return (
    <div className="page">
      <div className="page-header">
        <h2 className="page-title">🧩 Patchify 3D</h2>
        <p className="page-desc">
          Divide a 3-D volume into cubic patches for deep-learning training.
          Configure patch size and stride, then download the result as a compressed NumPy archive (.npz)
          containing the patch array, coordinates, and voxel spacing.
        </p>
      </div>

      {!imageId && (
        <div className="empty-state">
          <div className="empty-icon">🧩</div>
          <div className="empty-title">No image loaded</div>
          <div className="empty-hint">Upload an image in Data Manager first.</div>
        </div>
      )}

      {imageId && !is3d && (
        <div className="alert alert-warning">
          ⚠️ Patchify requires a 3-D volume. The current image is 2-D.
          Upload a DICOM series or NIfTI file.
        </div>
      )}

      {imageId && is3d && (
        <>
          <div className="controls-bar">
            <div className="ctrl-group" style={{ minWidth: 200 }}>
              <span className="ctrl-label">Patch size (voxels): {patchSize}</span>
              <div className="ctrl-range-row">
                <input type="range" min={8} max={128} step={8} value={patchSize}
                  onChange={e => { setPatchSize(Number(e.target.value)); setResult(null) }}
                  style={{ flex: 1 }} />
                <span className="ctrl-val">{patchSize}</span>
              </div>
            </div>

            <div className="ctrl-group" style={{ minWidth: 200 }}>
              <span className="ctrl-label">Stride (voxels): {stride}</span>
              <div className="ctrl-range-row">
                <input type="range" min={4} max={patchSize} step={4} value={Math.min(stride, patchSize)}
                  onChange={e => { setStride(Number(e.target.value)); setResult(null) }}
                  style={{ flex: 1 }} />
                <span className="ctrl-val">{stride}</span>
              </div>
            </div>

            <button className="btn btn-primary" onClick={run} disabled={loading}>
              {loading ? <><span className="spinner" /> Patching…</> : '🧩 Extract Patches'}
            </button>
          </div>

          {/* Estimate preview */}
          <div style={{ display: 'flex', gap: '.5rem', flexWrap: 'wrap', marginBottom: '1rem' }}>
            <span className="chip">Volume: {shape.join(' × ')} vx</span>
            <span className="chip blue">Patch size: {patchSize}³</span>
            <span className={`chip ${overlap > 0 ? 'teal' : ''}`}>Overlap: {overlap} vx</span>
            {est !== null && (
              <span className={`chip ${est > 0 ? 'green' : 'red'}`}>
                ~{est > 0 ? est.toLocaleString() : '0 (volume too small)'} patches
              </span>
            )}
          </div>

          {error && <div className="alert alert-error" style={{ marginBottom: '1rem' }}>⚠️ {error}</div>}

          {result && (
            <div className="card">
              <div className="card-header">
                <span className="card-title">✅ Patchify complete</span>
                <button
                  className="btn btn-teal btn-sm"
                  onClick={() => downloadNpz(result.npz_b64, `patches_${imageId.slice(0, 8)}.npz`)}
                >
                  ⬇ Download .npz
                </button>
              </div>
              <div className="card-body">
                <div className="meta-table">
                  <span className="meta-key">Volume shape</span>
                  <span className="meta-val">{result.volume_shape?.join(' × ')} vx</span>
                  <span className="meta-key">Patch size</span>
                  <span className="meta-val">{result.patch_size}³ vx</span>
                  <span className="meta-key">Stride</span>
                  <span className="meta-val">{result.stride} vx</span>
                  <span className="meta-key">Patches extracted</span>
                  <span className="meta-val"><strong>{result.n_patches?.toLocaleString()}</strong></span>
                  <span className="meta-key">Output array shape</span>
                  <span className="meta-val">{result.patches_shape?.join(' × ')}</span>
                  <span className="meta-key">Voxel spacing</span>
                  <span className="meta-val">{result.spacing_mm?.map(s => s.toFixed(3)).join(' × ')} mm</span>
                </div>
              </div>
              <div className="card-footer" style={{ fontSize: '.78rem', color: 'var(--text-secondary)' }}>
                <span>📦 .npz contains: <code>patches</code> (N×{result.patch_size}³), <code>coords</code> (N×3), <code>spacing</code> (3,)</span>
              </div>
            </div>
          )}

          {/* Usage example */}
          <div style={{ marginTop: '1.75rem' }}>
            <div className="section-label">Python usage</div>
            <pre className="code-block">
{`import numpy as np, io, base64

# After downloading patches_*.npz:
data    = np.load("patches_*.npz")
patches = data["patches"]   # float32  (N, P, P, P)
coords  = data["coords"]    # int32    (N, 3)  [z, y, x]
spacing = data["spacing"]   # float32  (3,)  mm/voxel

print(f"{len(patches)} patches of shape {patches[0].shape}")`}
            </pre>
          </div>
        </>
      )}
    </div>
  )
}
