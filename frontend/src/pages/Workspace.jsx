/**
 * pages/Workspace.jsx – MPR viewer with spacing-correct aspect ratios
 *
 * Calls GET /api/mpr/{image_id} which returns Axial / Coronal / Sagittal slices
 * resampled to the correct physical aspect ratio using voxel spacing.
 * Includes window/level presets and FOV display.
 */
import { useState, useEffect, useCallback } from 'react'
import { api } from '../api/client.js'

/* Window / Level presets (CT-oriented, also usable for MRI) */
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

  const shape  = metadata?.shape  || []
  const is3d   = metadata?.is_3d  || false
  const sp     = metadata?.spacing || [1, 1, 1]
  const resolvedSpacing = (Array.isArray(mpr?.spacing_mm) && mpr.spacing_mm.length >= 3 ? mpr.spacing_mm : sp)
  const [zSpacing = 1, ySpacing = 1, xSpacing = 1] = resolvedSpacing
  const calculateScaleY = (axisSpacing, referenceSpacing) => (
    referenceSpacing > 0 && axisSpacing > 0 ? (axisSpacing / referenceSpacing) : 1
  )
  const coronalScaleY = calculateScaleY(zSpacing, ySpacing)
  const sagittalScaleY = calculateScaleY(zSpacing, xSpacing)
  const fov    = is3d && shape.length >= 3
    ? { z: (shape[0] * sp[0]).toFixed(1), y: (shape[1] * (sp[1]??1)).toFixed(1), x: (shape[2] * (sp[2]??1)).toFixed(1) }
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

  /* Load on mount / imageId change */
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
    { key: 'axial',    label: 'Axial (Z)',    axis: 'axial',    max: shape[0] - 1, sp_row: sp[1]??1, sp_col: sp[2]??1 },
    { key: 'coronal',  label: 'Coronal (Y)',  axis: 'coronal',  max: shape[1] - 1, sp_row: sp[0]??1, sp_col: sp[2]??1 },
    { key: 'sagittal', label: 'Sagittal (X)', axis: 'sagittal', max: shape[2] - 1, sp_row: sp[0]??1, sp_col: sp[1]??1 },
  ]
  const panelStyles = {
    axial: undefined,
    coronal: { transform: `scaleY(${coronalScaleY})`, transformOrigin: 'center center' },
    sagittal: { transform: `scaleY(${sagittalScaleY})`, transformOrigin: 'center center' },
  }

  return (
    <div className="page">
      <div className="page-header">
        <h2 className="page-title">🔬 Workspace</h2>
        <p className="page-desc">
          Multi-Planar Reconstruction with geometrically-correct aspect ratios.
          Each slice is resampled using voxel spacing so anatomical proportions are preserved.
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
          <div className="controls-bar">
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
              <div key={panel.key} className="mpr-panel">
                <div className="mpr-header">
                  <span className="mpr-header-label">{panel.label}</span>
                  {mpr?.spacing_mm && (
                    <span className="mpr-header-info">
                      {panel.sp_row.toFixed(2)} × {panel.sp_col.toFixed(2)} mm/px
                    </span>
                  )}
                </div>

                <div className="mpr-img-wrap">
                  {mpr?.[panel.key]
                    ? <img className="mpr-img" src={mpr[panel.key]} alt={panel.label} style={panelStyles[panel.key]} />
                    : <div className="mpr-no-img">{loading ? 'Loading…' : '—'}</div>
                  }
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
                <span className="chip teal">Spacing: {mpr.spacing_mm?.map(s => s.toFixed(2)).join(' × ')} mm</span>
                {mpr.fov_mm && (
                  <>
                    {mpr.fov_mm.z_mm && <span className="chip">FOV Z: {mpr.fov_mm.z_mm} mm</span>}
                    <span className="chip">FOV Y: {mpr.fov_mm.y_mm} mm</span>
                    <span className="chip">FOV X: {mpr.fov_mm.x_mm} mm</span>
                  </>
                )}
              </div>
            </div>
          )}
        </>
      )}
    </div>
  )
}
