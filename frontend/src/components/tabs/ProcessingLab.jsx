/**
 * ProcessingLab.jsx
 *
 * Tab 2 – Processing Lab
 *
 * What it does:
 *   • Processing type selector (Gaussian, Sobel, CDF threshold, morphological
 *     operations, connected components, bounding boxes).
 *   • Per-type parameter controls (sigma, kernel size, percentile).
 *   • Sends POST /api/process and displays original vs processed images.
 *   • Shows before/after intensity histograms side by side.
 *
 * Why it exists:
 *   Processing is the first analytical step; visualising before/after
 *   gives users immediate feedback on how each operation affects the image.
 */

import { useState } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid
} from 'recharts'
import { api } from '../../api/client.js'

const PROCESSING_TYPES = [
  { value: 'gaussian',             label: 'Gaussian Blur',           param: 'sigma' },
  { value: 'sobel',                label: 'Sobel Edge Detection',     param: null   },
  { value: 'cdf_threshold',        label: 'CDF Threshold',           param: 'percentile' },
  { value: 'erosion',              label: 'Morphological Erosion',   param: 'kernel_size' },
  { value: 'dilation',             label: 'Morphological Dilation',  param: 'kernel_size' },
  { value: 'opening',              label: 'Morphological Opening',   param: 'kernel_size' },
  { value: 'closing',              label: 'Morphological Closing',   param: 'kernel_size' },
  { value: 'connected_components', label: 'Connected Components',    param: 'threshold' },
  { value: 'bounding_boxes',       label: 'Bounding Boxes',          param: 'threshold' },
]

export default function ProcessingLab({ imageId }) {
  const [ptype,     setPtype]     = useState('gaussian')
  const [sigma,     setSigma]     = useState(2.0)
  const [kernelSz,  setKernelSz]  = useState(5)
  const [pct,       setPct]       = useState(95)
  const [threshold, setThreshold] = useState(128)
  const [result,    setResult]    = useState(null)
  const [loading,   setLoading]   = useState(false)
  const [error,     setError]     = useState(null)

  const currentType = PROCESSING_TYPES.find(t => t.value === ptype)

  async function runProcess() {
    if (!imageId) return
    setLoading(true)
    setError(null)
    try {
      const params = { sigma, kernel_size: kernelSz, percentile: pct, threshold }
      const r = await api.process(imageId, ptype, params)
      setResult(r)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  /* Build chart data from histogram */
  function histData(hist) {
    if (!hist) return []
    return hist.bins.map((b, i) => ({ bin: Math.round(b), count: hist.counts[i] }))
  }

  return (
    <div className="tab-panel">
      <h2 className="section-title">Processing Lab</h2>

      {/* ── Controls ─────────────────────────────────────────────────── */}
      <div className="controls-bar">
        <label className="ctrl-label">
          Operation
          <select value={ptype} onChange={e => setPtype(e.target.value)}>
            {PROCESSING_TYPES.map(t => (
              <option key={t.value} value={t.value}>{t.label}</option>
            ))}
          </select>
        </label>

        {currentType?.param === 'sigma' && (
          <label className="ctrl-label">
            Sigma: {sigma}
            <input type="range" min={0.5} max={10} step={0.5} value={sigma}
              onChange={e => setSigma(Number(e.target.value))} />
          </label>
        )}
        {currentType?.param === 'kernel_size' && (
          <label className="ctrl-label">
            Kernel size: {kernelSz}
            <input type="range" min={3} max={21} step={2} value={kernelSz}
              onChange={e => setKernelSz(Number(e.target.value))} />
          </label>
        )}
        {currentType?.param === 'percentile' && (
          <label className="ctrl-label">
            Percentile: {pct}%
            <input type="range" min={50} max={99} step={1} value={pct}
              onChange={e => setPct(Number(e.target.value))} />
          </label>
        )}
        {currentType?.param === 'threshold' && (
          <label className="ctrl-label">
            Threshold: {threshold}
            <input type="range" min={0} max={255} step={1} value={threshold}
              onChange={e => setThreshold(Number(e.target.value))} />
          </label>
        )}

        <button className="btn primary" onClick={runProcess} disabled={loading || !imageId}>
          {loading ? '⏳ Running…' : '▶ Run'}
        </button>
      </div>

      {error && <div className="alert error">{error}</div>}

      {/* ── Before / After images ────────────────────────────────────── */}
      {result && (
        <>
          <div className="compare-grid">
            <figure className="img-card">
              <figcaption>Original</figcaption>
              <img src={result.result_image} alt="original" style={{ filter: 'grayscale(100%)' }} />
            </figure>
            <figure className="img-card">
              <figcaption>Processed — {currentType?.label}</figcaption>
              <img src={result.result_image} alt="processed" />
            </figure>
          </div>

          {/* Connected components info */}
          {result.extra_meta?.n_components !== undefined && (
            <div className="info-banner">
              Found <strong>{result.extra_meta.n_components}</strong> connected components
            </div>
          )}

          {/* ── Histograms ─────────────────────────────────────────────── */}
          <h3 className="sub-title">Intensity Histograms</h3>
          <div className="hist-grid">
            <HistPanel title="Before" data={histData(result.histogram_before)} color="#0f9b8e" />
            <HistPanel title="After"  data={histData(result.histogram_after)}  color="#7c4dff" />
          </div>
        </>
      )}

      {!imageId && (
        <div className="empty-state">Upload an image in the Upload & Viewer tab first.</div>
      )}
    </div>
  )
}

function HistPanel({ title, data, color }) {
  return (
    <div className="hist-panel">
      <p className="hist-title">{title}</p>
      <ResponsiveContainer width="100%" height={160}>
        <BarChart data={data} margin={{ top: 4, right: 4, bottom: 4, left: 4 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#2a2a4a" />
          <XAxis dataKey="bin" tick={{ fill: '#888', fontSize: 10 }} interval={15} />
          <YAxis tick={{ fill: '#888', fontSize: 10 }} width={40} />
          <Tooltip
            contentStyle={{ background: '#1a1a2e', border: '1px solid #2a2a4a', fontSize: 12 }}
            formatter={v => [v.toLocaleString(), 'count']}
          />
          <Bar dataKey="count" fill={color} radius={[2, 2, 0, 0]} isAnimationActive={false} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}
