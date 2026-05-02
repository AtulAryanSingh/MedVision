/**
 * pages/Tools.jsx – Fully modular toolbox
 *
 * Each tool card is independent.  The user picks parameters and runs the tool
 * on demand — no forced pipeline order.  Results appear inline in the card.
 */
import { useState } from 'react'
import { BarChart, Bar, XAxis, YAxis, Tooltip, CartesianGrid, ResponsiveContainer } from 'recharts'
import { api } from '../api/client.js'

const TOOLTIP_STYLE = { background: '#fff', border: '1px solid #e2e8f0', fontSize: 12, borderRadius: 6 }

/* Tool definitions */
const TOOLS = [
  {
    id: 'gaussian',
    icon: '🌀',
    name: 'Gaussian Blur',
    desc: 'Smooth the image to reduce noise while preserving large structures.',
    params: [{ id: 'sigma', label: 'Sigma', min: 0.5, max: 10, step: 0.5, default: 2 }],
  },
  {
    id: 'median',
    icon: '🔲',
    name: 'Median Filter',
    desc: 'Remove salt-and-pepper noise. Preserves edges better than Gaussian.',
    params: [{ id: 'kernel_size', label: 'Kernel size', min: 3, max: 21, step: 2, default: 5 }],
  },
  {
    id: 'sobel',
    icon: '✏️',
    name: 'Sobel Edge Detection',
    desc: 'Compute gradient magnitude to highlight structural boundaries.',
    params: [],
  },
  {
    id: 'cdf_threshold',
    icon: '⬜',
    name: 'CDF Threshold',
    desc: 'Binarise at the intensity value matching a chosen CDF percentile.',
    params: [{ id: 'percentile', label: 'Percentile (%)', min: 50, max: 99, step: 1, default: 95 }],
  },
  {
    id: 'erosion',
    icon: '⬛',
    name: 'Morphological Erosion',
    desc: 'Shrink bright foreground regions. Removes thin protrusions.',
    params: [{ id: 'kernel_size', label: 'Kernel size', min: 3, max: 21, step: 2, default: 5 }],
  },
  {
    id: 'dilation',
    icon: '🔲',
    name: 'Morphological Dilation',
    desc: 'Expand bright regions. Fills small gaps in segmentation masks.',
    params: [{ id: 'kernel_size', label: 'Kernel size', min: 3, max: 21, step: 2, default: 5 }],
  },
  {
    id: 'opening',
    icon: '🔓',
    name: 'Morphological Opening',
    desc: 'Erosion followed by dilation. Removes small bright blobs.',
    params: [{ id: 'kernel_size', label: 'Kernel size', min: 3, max: 21, step: 2, default: 5 }],
  },
  {
    id: 'closing',
    icon: '🔒',
    name: 'Morphological Closing',
    desc: 'Dilation followed by erosion. Fills small dark holes.',
    params: [{ id: 'kernel_size', label: 'Kernel size', min: 3, max: 21, step: 2, default: 5 }],
  },
  {
    id: 'connected_components',
    icon: '🔗',
    name: 'Connected Components',
    desc: 'Label each distinct region and colour it uniquely. Returns component statistics.',
    params: [{ id: 'threshold', label: 'Threshold (0–255)', min: 0, max: 255, step: 1, default: 128 }],
  },
  {
    id: 'bounding_boxes',
    icon: '🟦',
    name: 'Bounding Boxes',
    desc: 'Draw bounding rectangles and centroids around each connected region.',
    params: [{ id: 'threshold', label: 'Threshold (0–255)', min: 0, max: 255, step: 1, default: 128 }],
  },
]

/* Single tool card */
function ToolCard({ tool, imageId }) {
  const initParams = Object.fromEntries(tool.params.map(p => [p.id, p.default]))
  const [params,  setParams]  = useState(initParams)
  const [result,  setResult]  = useState(null)
  const [loading, setLoading] = useState(false)
  const [error,   setError]   = useState(null)

  async function run() {
    if (!imageId) return
    setLoading(true); setError(null)
    try {
      const r = await api.process(imageId, tool.id, params)
      setResult(r)
    } catch (e) { setError(e.message) }
    finally     { setLoading(false) }
  }

  function histData(hist) {
    if (!hist) return []
    return hist.bins.map((b, i) => ({ bin: Math.round(b), count: hist.counts[i] }))
  }

  return (
    <div className="tool-card">
      <div className="tool-card-header">
        <span className="tool-card-title">{tool.icon} {tool.name}</span>
        <button className="btn btn-primary btn-sm" onClick={run} disabled={loading || !imageId}>
          {loading ? <span className="spinner" /> : '▶ Run'}
        </button>
      </div>

      <div className="tool-card-body">
        <p style={{ fontSize: '.77rem', color: 'var(--text-secondary)', marginBottom: tool.params.length ? '.85rem' : 0, lineHeight: 1.5 }}>
          {tool.desc}
        </p>

        {tool.params.map(p => (
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

        {error && <div className="alert alert-error" style={{ marginTop: '.5rem', fontSize: '.78rem' }}>⚠️ {error}</div>}
      </div>

      {result && (
        <div className="tool-result">
          {/* Before / After */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr' }}>
            <div>
              <div style={{ fontSize: '.68rem', padding: '.3rem .75rem', background: 'var(--surface2)', borderTop: '1px solid var(--border)', borderRight: '1px solid var(--border)', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '.4px' }}>
                Original
              </div>
              {/* We don't store original image here — show histograms instead */}
            </div>
            <div>
              <div style={{ fontSize: '.68rem', padding: '.3rem .75rem', background: 'var(--surface2)', borderTop: '1px solid var(--border)', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '.4px' }}>
                Result
              </div>
            </div>
          </div>

          <img src={result.result_image} alt="result" style={{ display: 'block', width: '100%', maxHeight: 280, objectFit: 'contain', background: '#000' }} />

          {result.extra_meta?.n_components !== undefined && (
            <div className="alert alert-info" style={{ margin: '.6rem', borderRadius: 'var(--radius-sm)', fontSize: '.78rem' }}>
              🔗 Found <strong>{result.extra_meta.n_components}</strong> connected components
            </div>
          )}

          {/* Mini histograms */}
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

export default function Tools({ imageId }) {
  return (
    <div className="page">
      <div className="page-header">
        <h2 className="page-title">⚗️ Tools</h2>
        <p className="page-desc">
          Each tool runs independently on the loaded image. Pick any tool, set parameters, and click Run.
          No forced pipeline — run them in any order, any number of times.
        </p>
      </div>

      {!imageId ? (
        <div className="empty-state">
          <div className="empty-icon">⚗️</div>
          <div className="empty-title">No image loaded</div>
          <div className="empty-hint">Upload an image in Data Manager first.</div>
        </div>
      ) : (
        <div className="tools-grid">
          {TOOLS.map(tool => <ToolCard key={tool.id} tool={tool} imageId={imageId} />)}
        </div>
      )}
    </div>
  )
}
