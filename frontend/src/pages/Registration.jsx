/**
 * pages/Registration.jsx – 2-D Registration & Resampling
 *
 * Applies geometric transforms to the loaded image on-demand:
 *   • Translate   – shift horizontally / vertically
 *   • Rotate      – rotate around image centre
 *   • Zoom        – rescale (anisotropic)
 *   • Affine      – free 2×2 matrix + translation
 *
 * Interpolation: bicubic (order=3) for images, nearest-neighbour (order=0)
 * for binary masks.
 */
import { useEffect, useState } from 'react'
import { BarChart, Bar, XAxis, YAxis, Tooltip, CartesianGrid, ResponsiveContainer } from 'recharts'
import { api } from '../api/client.js'
import AsyncJobPanel from '../components/AsyncJobPanel.jsx'
import { JOB_POLL_INTERVAL_MS, JOB_TERMINAL_STATES } from '../constants/async.js'

const TOOLTIP_STYLE = { background: '#fff', border: '1px solid #e2e8f0', fontSize: 12, borderRadius: 6 }

const TRANSFORMS = [
  {
    id: 'translate',
    icon: '↔️',
    name: 'Translation',
    desc: 'Shift the image horizontally and/or vertically by a fixed number of pixels.',
    params: [
      { id: 'shift_x', label: 'Shift X (px)', min: -200, max: 200, step: 1,  default: 0 },
      { id: 'shift_y', label: 'Shift Y (px)', min: -200, max: 200, step: 1,  default: 0 },
    ],
  },
  {
    id: 'rotate',
    icon: '🔄',
    name: 'Rotation',
    desc: 'Rotate the image around its centre. Positive angle = counter-clockwise.',
    params: [
      { id: 'angle_deg', label: 'Angle (°)', min: -180, max: 180, step: 1, default: 0 },
    ],
  },
  {
    id: 'zoom',
    icon: '🔍',
    name: 'Zoom / Rescale',
    desc: 'Scale the image along each axis independently. 1.0 = no change.',
    params: [
      { id: 'zoom_x', label: 'Zoom X', min: 0.1, max: 4, step: 0.05, default: 1 },
      { id: 'zoom_y', label: 'Zoom Y', min: 0.1, max: 4, step: 0.05, default: 1 },
    ],
  },
  {
    id: 'affine',
    icon: '🔀',
    name: 'Affine Transform',
    desc: 'Apply a custom 2×2 linear matrix (shear, scale, flip) plus translation.',
    params: [
      { id: 'm00', label: 'M[0,0]', min: -4, max: 4, step: 0.05, default: 1 },
      { id: 'm01', label: 'M[0,1]', min: -4, max: 4, step: 0.05, default: 0 },
      { id: 'm10', label: 'M[1,0]', min: -4, max: 4, step: 0.05, default: 0 },
      { id: 'm11', label: 'M[1,1]', min: -4, max: 4, step: 0.05, default: 1 },
      { id: 'tx',  label: 'Tx (px)', min: -200, max: 200, step: 1, default: 0 },
      { id: 'ty',  label: 'Ty (px)', min: -200, max: 200, step: 1, default: 0 },
    ],
  },
]

function histData(hist) {
  if (!hist) return []
  return hist.bins.map((b, i) => ({ bin: Math.round(b), count: hist.counts[i] }))
}

function TransformCard({ tf, imageId }) {
  const initParams = Object.fromEntries(tf.params.map(p => [p.id, p.default]))
  const [params,  setParams]  = useState(initParams)
  const [isMask,  setIsMask]  = useState(false)
  const [result,  setResult]  = useState(null)
  const [loading, setLoading] = useState(false)
  const [error,   setError]   = useState(null)
  const [job,     setJob]     = useState(null)

  useEffect(() => {
    if (!job || JOB_TERMINAL_STATES.includes(job.status)) return
    const t = setInterval(async () => {
      try {
        const j = await api.getJob(job.job_id)
        setJob(j)
        if (j.status === 'succeeded') {
          const out = await api.getJobResult(j.job_id)
          setResult(out)
        }
      } catch (e) {
        setError(e.message)
      }
    }, JOB_POLL_INTERVAL_MS)
    return () => clearInterval(t)
  }, [job])

  function buildPayload() {
    if (tf.id === 'affine') {
      return {
        is_mask: isMask,
        matrix: [params.m00, params.m01, params.m10, params.m11],
        tx: params.tx,
        ty: params.ty,
      }
    }
    return { ...params, is_mask: isMask }
  }

  async function run() {
    if (!imageId) return
    setLoading(true); setError(null)
    try {
      setResult(null)
      const created = await api.createRegisterJob(imageId, tf.id, buildPayload())
      setJob(created)
    } catch (e) { setError(e.message) }
    finally     { setLoading(false) }
  }

  async function refreshJob() {
    if (!job?.job_id) return
    const j = await api.getJob(job.job_id)
    setJob(j)
    if (j.status === 'succeeded') {
      const out = await api.getJobResult(j.job_id)
      setResult(out)
    }
  }

  async function cancelJob() {
    if (!job?.job_id) return
    const j = await api.cancelJob(job.job_id)
    setJob(j)
  }

  return (
    <div className="tool-card">
      <div className="tool-card-header">
        <span className="tool-card-title">{tf.icon} {tf.name}</span>
        <button className="btn btn-primary btn-sm" onClick={run} disabled={loading || !imageId}>
          {loading ? <span className="spinner" /> : '▶ Apply'}
        </button>
      </div>

      <div className="tool-card-body">
        <p style={{ fontSize: '.77rem', color: 'var(--text-secondary)', marginBottom: '.85rem', lineHeight: 1.5 }}>
          {tf.desc}
        </p>

        {tf.params.map(p => (
          <div key={p.id} style={{ marginBottom: '.6rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '.73rem', color: 'var(--text-secondary)', marginBottom: '.2rem' }}>
              <span>{p.label}</span>
              <strong style={{ color: 'var(--accent-text)' }}>{params[p.id]}</strong>
            </div>
            <input
              type="range" min={p.min} max={p.max} step={p.step} value={params[p.id]}
              onChange={e => setParams(prev => ({ ...prev, [p.id]: Number(e.target.value) }))}
              style={{ width: '100%', accentColor: 'var(--accent)' }}
            />
          </div>
        ))}

        {/* Mask toggle */}
        <label style={{ display: 'flex', alignItems: 'center', gap: '.5rem', marginTop: '.4rem', fontSize: '.76rem', color: 'var(--text-secondary)', cursor: 'pointer' }}>
          <input
            type="checkbox" checked={isMask}
            onChange={e => setIsMask(e.target.checked)}
            style={{ accentColor: 'var(--accent)' }}
          />
          Treat as binary mask (nearest-neighbour interpolation)
        </label>

        {error && <div className="alert alert-error" style={{ marginTop: '.5rem', fontSize: '.78rem' }}>⚠️ {error}</div>}
        {job && (
          <div style={{ marginTop: '.6rem' }}>
            <AsyncJobPanel
              title={`${tf.name} job`}
              job={job}
              onRefresh={refreshJob}
              onCancel={cancelJob}
              onRetry={run}
            />
          </div>
        )}
      </div>

      {result && (
        <div className="tool-result">
          <img src={result.result_image} alt="result" style={{ display: 'block', width: '100%', maxHeight: 300, objectFit: 'contain', background: '#000' }} />

          {result.output_shape && (
            <div style={{ padding: '.5rem .75rem', fontSize: '.74rem', color: 'var(--text-muted)', borderTop: '1px solid var(--border)' }}>
              Output shape: {result.output_shape.join(' × ')} px
            </div>
          )}

          {/* Before / After histograms */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 0, borderTop: '1px solid var(--border)' }}>
            {[
              { title: 'Before', data: histData(result.histogram_before), color: '#94a3b8' },
              { title: 'After',  data: histData(result.histogram_after),  color: '#2563eb' },
            ].map(h => (
              <div key={h.title} style={{ padding: '.6rem', borderRight: h.title === 'Before' ? '1px solid var(--border)' : 'none' }}>
                <div style={{ fontSize: '.68rem', color: 'var(--text-muted)', textAlign: 'center', marginBottom: '.3rem' }}>{h.title}</div>
                <ResponsiveContainer width="100%" height={70}>
                  <BarChart data={h.data} margin={{ top: 0, right: 0, bottom: 0, left: 0 }}>
                    <XAxis hide /><YAxis hide />
                    <Tooltip contentStyle={TOOLTIP_STYLE} formatter={v => [v.toLocaleString(), 'count']} />
                    <Bar dataKey="count" fill={h.color} radius={[1,1,0,0]} isAnimationActive={false} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

export default function Registration({ imageId }) {
  return (
    <div className="page">
      <div className="page-header">
        <h2 className="page-title">📐 Registration & Resampling</h2>
        <p className="page-desc">
          Apply geometric transforms to the loaded image for alignment and resampling.
          Each transform is independent — run them in any order on demand.
          Use bicubic interpolation for images; switch to nearest-neighbour for binary masks.
        </p>
      </div>

      {!imageId ? (
        <div className="empty-state">
          <div className="empty-icon">📐</div>
          <div className="empty-title">No image loaded</div>
          <div className="empty-hint">Upload an image in Data Manager first.</div>
        </div>
      ) : (
        <>
          <div style={{ marginBottom: '1rem' }}>
            <div className="section-label">Available Transforms</div>
            <div style={{ display: 'flex', gap: '.5rem', flexWrap: 'wrap', marginTop: '.4rem', marginBottom: '1.2rem' }}>
              {['Translation', 'Rotation', 'Zoom / Rescale', 'Affine Matrix'].map(t => (
                <span key={t} className="chip">{t}</span>
              ))}
            </div>
          </div>

          <div className="tools-grid">
            {TRANSFORMS.map(tf => <TransformCard key={tf.id} tf={tf} imageId={imageId} />)}
          </div>

          {/* Usage note */}
          <div style={{ marginTop: '1.75rem' }}>
            <div className="section-label">Python equivalents</div>
            <pre className="code-block">
{`from scipy.ndimage import affine_transform, rotate, zoom
import numpy as np

# Translation
shifted = affine_transform(arr, np.eye(2), offset=[-shift_y, -shift_x])

# Rotation (around centre)
rotated = rotate(arr, angle=15.0, reshape=False)

# Zoom
rescaled = zoom(arr, zoom=(zoom_y, zoom_x))   # order=3 for images, order=0 for masks

# Affine (2×2 matrix + offset)
M = np.array([[1, 0.2], [0, 1]])  # shear example
transformed = affine_transform(arr, M, offset=[0, 0])`}
            </pre>
          </div>
        </>
      )}
    </div>
  )
}
