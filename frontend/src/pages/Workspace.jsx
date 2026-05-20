/**
 * pages/Workspace.jsx – MPR viewer with spacing-correct aspect ratios
 * Includes real-time Manual Spacing Override for bad DICOM/NIfTI metadata.
 */
import { useState, useEffect, useCallback } from 'react'
import { api } from '../api/client.js'

/* Window / Level presets */
const WL_PRESETS = [
  { label: 'Auto (full range)',   wc: null,  ww: null },
  { label: 'CT – Brain',          wc: 40,    ww: 80   },
  { label: 'CT – Bone',           wc: 400,   ww: 1500 },
  { label: 'CT – Lung',           wc: -600,  ww: 1500 },
  { label: 'CT – Abdomen',        wc: 60,    ww: 400  },
  { label: 'CT – Soft tissue',    wc: 50,    ww: 350  },
  { label: 'MR – T1',             wc: 500,   ww: 1000 },
  { label: 'MR – T2 (FLAIR)',     wc: 1000,  ww: 2000 },
]

export default function Workspace({ imageId, metadata }) {
  const [mpr,        setMpr]        = useState(null)
  const [slices,     setSlices]     = useState({ axial: 0, coronal: 0, sagittal: 0 })
  const [preset,     setPreset]     = useState(0)
  const [loading,    setLoading]    = useState(false)
  const [error,      setError]      = useState(null)

  // NEW: State to hold the user's manual spacing overrides
  const [manualSpacing, setManualSpacing] = useState({ z: null, y: null, x: null })

  // 1. Extract base data from the file
  const shape  = metadata?.shape  || []
  const is3d   = metadata?.is_3d  || false
  const sp     = metadata?.spacing || [1, 1, 1]
  const baseSpacing = (Array.isArray(mpr?.spacing_mm) && mpr.spacing_mm.length >= 3) ? mpr.spacing_mm : sp
  
  const [zDim = 1, yDim = 1, xDim = 1] = shape;
  const [fileZ = 1, fileY = 1, fileX = 1] = baseSpacing;
  
  // 2. Determine effective spacing (Prefer Manual over File)
  const zSpacing = manualSpacing.z !== null ? manualSpacing.z : fileZ;
  const ySpacing = manualSpacing.y !== null ? manualSpacing.y : fileY;
  const xSpacing = manualSpacing.x !== null ? manualSpacing.x : fileX;
  
  // 3. Calculate true physical aspect ratios (Width / Height)
  const axialRatio = (xDim * xSpacing) / (yDim * ySpacing) || 1;
  const coronalRatio = (xDim * xSpacing) / (zDim * zSpacing) || 1;
  const sagittalRatio = (yDim * ySpacing) / (zDim * zSpacing) || 1;

  const fov    = is3d && shape.length >= 3
    ? { z: (shape[0] * zSpacing).toFixed(1), y: (shape[1] * ySpacing).toFixed(1), x: (shape[2] * xSpacing).toFixed(1) }
    : null

  const fetchMpr = useCallback(async (sl = slices, p = preset) => {
    if (!imageId) return
    setLoading(true)
    setError(null)
    try {
      const wl = WL_PRESETS[p]
      const data = await api.mpr(imageId, {
        axialIdx:    is3d ? sl.axial    : undefined,
        coronalIdx:  is3d ? sl.coronal  : undefined,
        sagittalIdx: is3d ? sl.sagittal : undefined,
        windowCenter: wl.wc ?? undefined,
        windowWidth:  wl.ww ?? undefined,
      })
      setMpr(data)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }, [imageId, is3d]) // eslint-disable-line

  useEffect(() => {
    if (imageId) fetchMpr()
  }, [imageId]) // eslint-disable-line

  async function onSlice(axis, val) {
    const next = { ...slices, [axis]: Number(val) }
    setSlices(next)
    await fetchMpr(next, preset)
  }

  async function onPreset(p) {
    setPreset(p)
    await fetchMpr(slices, p)
  }

  const PANELS = [
    { key: 'axial',    label: 'Axial (Z)',    axis: 'axial',    max: zDim - 1, sp_row: ySpacing, sp_col: xSpacing, aspectRatio: axialRatio },
    { key: 'coronal',  label: 'Coronal (Y)',  axis: 'coronal',  max: yDim - 1, sp_row: zSpacing, sp_col: xSpacing, aspectRatio: coronalRatio },
    { key: 'sagittal', label: 'Sagittal (X)', axis: 'sagittal', max: xDim - 1, sp_row: zSpacing, sp_col: ySpacing, aspectRatio: sagittalRatio },
  ]

  return (
    <div className="page">
      <div className="page-header">
        <h2 className="page-title">🔬 Workspace</h2>
        <p className="page-desc">
          Multi-Planar Reconstruction with geometrically-correct aspect ratios.
          Use the manual override panel if the file's metadata is incorrect.
        </p>
      </div>

      {!imageId && (
        <div className="empty-state">
          <div className="empty-icon">🔬</div>
          <div className="empty-title">No image loaded</div>
          <div className="empty-hint">Go to Data Manager and upload a file first.</div>
        </div>
      )}

      {imageId && (
        <>
          {/* Controls */}
          <div className="controls-bar" style={{ display: 'flex', flexWrap: 'wrap', gap: '1rem', alignItems: 'center' }}>
            <div className="ctrl-group">
              <span className="ctrl-label">Window / Level Preset</span>
              <select
                className="form-select"
                value={preset}
                onChange={e => onPreset(Number(e.target.value))}
                style={{ minWidth: 200 }}
              >
                {WL_PRESETS.map((p, i) => <option key={i} value={i}>{p.label}</option>)}
              </select>
            </div>

            {/* NEW: Manual Spacing Override Panel */}
            <div className="ctrl-group" style={{ paddingLeft: '1rem', borderLeft: '1px solid var(--border-color)' }}>
              <span className="ctrl-label">Override Spacing (mm)</span>
              <div style={{ display: 'flex', gap: '0.5rem' }}>
                <input 
                  type="number" step="0.1" placeholder={`Z (${fileZ.toFixed(2)})`}
                  className="form-input" style={{ width: '80px', padding: '0.25rem 0.5rem' }}
                  onChange={e => setManualSpacing(prev => ({ ...prev, z: e.target.value ? parseFloat(e.target.value) : null }))}
                />
                <input 
                  type="number" step="0.1" placeholder={`Y (${fileY.toFixed(2)})`}
                  className="form-input" style={{ width: '80px', padding: '0.25rem 0.5rem' }}
                  onChange={e => setManualSpacing(prev => ({ ...prev, y: e.target.value ? parseFloat(e.target.value) : null }))}
                />
                <input 
                  type="number" step="0.1" placeholder={`X (${fileX.toFixed(2)})`}
                  className="form-input" style={{ width: '80px', padding: '0.25rem 0.5rem' }}
                  onChange={e => setManualSpacing(prev => ({ ...prev, x: e.target.value ? parseFloat(e.target.value) : null }))}
                />
              </div>
            </div>

            {fov && (
              <div style={{ marginLeft: 'auto', display: 'flex', gap: '1rem', flexWrap: 'wrap', alignItems: 'flex-end' }}>
                <div style={{ textAlign: 'center' }}>
                  <div style={{ fontSize: '.68rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '.5px' }}>FOV Z</div>
                  <span className="chip teal">{fov.z} mm</span>
                </div>
                <div style={{ textAlign: 'center' }}>
                  <div style={{ fontSize: '.68rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '.5px' }}>FOV Y</div>
                  <span className="chip teal">{fov.y} mm</span>
                </div>
                <div style={{ textAlign: 'center' }}>
                  <div style={{ fontSize: '.68rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '.5px' }}>FOV X</div>
                  <span className="chip teal">{fov.x} mm</span>
                </div>
              </div>
            )}

            <button className="btn btn-outline" onClick={() => fetchMpr()} disabled={loading}>
              {loading ? <><span className="spinner" /> Loading…</> : '↺ Refresh'}
            </button>
          </div>

          {error && <div className="alert alert-error" style={{ marginBottom: '1rem' }}>⚠️ {error}</div>}

          {/* MPR panels */}
          <div className="mpr-grid">
            {PANELS.map(panel => (
              <div key={panel.key} className="mpr-panel" style={{ display: 'flex', flexDirection: 'column' }}>
                <div className="mpr-header">
                  <span className="mpr-header-label">{panel.label}</span>
                  <span className="mpr-header-info">
                    {panel.sp_row.toFixed(2)} × {panel.sp_col.toFixed(2)} mm/px
                  </span>
                </div>

                {/* THE FIX: Inner container enforces aspect ratio, image uses 'fill' */}
                <div className="mpr-img-wrap" style={{ flexGrow: 1, display: 'flex', justifyContent: 'center', alignItems: 'center', overflow: 'hidden', backgroundColor: '#000' }}>
                  {mpr?.[panel.key] ? (
                    <div style={{ width: '100%', aspectRatio: panel.aspectRatio }}>
                      <img 
                        className="mpr-img" 
                        src={mpr[panel.key]} 
                        alt={panel.label} 
                        style={{ width: '100%', height: '100%', objectFit: 'fill', display: 'block' }} 
                      />
                    </div>
                  ) : (
                    <div className="mpr-no-img" style={{ color: '#fff' }}>{loading ? 'Loading…' : '—'}</div>
                  )}
                </div>

                {is3d && (
                  <div className="mpr-footer">
                    <div className="mpr-slider-label">
                      <span>Slice {slices[panel.axis]}</span>
                      <span>/ {panel.max}</span>
                    </div>
                    <input
                      type="range" min={0} max={Math.max(0, panel.max)} value={slices[panel.axis]}
                      onChange={e => onSlice(panel.axis, e.target.value)}
                      style={{ width: '100%', accentColor: 'var(--accent)' }}
                    />
                  </div>
                )}
              </div>
            ))}
          </div>

          {/* Metadata */}
          {mpr && (
            <div style={{ marginTop: '1.5rem' }}>
              <div className="section-label">Volume Info</div>
              <div style={{ display: 'flex', gap: '.5rem', flexWrap: 'wrap' }}>
                <span className="chip">Shape: {mpr.shape?.join(' × ')}</span>
                <span className="chip teal">Effective Spacing: {zSpacing.toFixed(2)} × {ySpacing.toFixed(2)} × {xSpacing.toFixed(2)} mm</span>
              </div>
            </div>
          )}
        </>
      )}
    </div>
  )
}
