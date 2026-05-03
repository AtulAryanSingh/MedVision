/**
 * pages/QCPlots.jsx – QC analysis: window/level, histogram, CDF, feature stats
 */
import { useState } from 'react'
import {
  BarChart, Bar, AreaChart, Area,
  XAxis, YAxis, Tooltip, CartesianGrid, ResponsiveContainer,
} from 'recharts'
import { api } from '../api/client.js'

const WL_PRESETS = [
  { label: 'CT – Brain (WC40/WW80)',         wc: 40,   ww: 80   },
  { label: 'CT – Bone (WC400/WW1500)',        wc: 400,  ww: 1500 },
  { label: 'CT – Lung (WC-600/WW1500)',       wc: -600, ww: 1500 },
  { label: 'CT – Abdomen (WC60/WW400)',       wc: 60,   ww: 400  },
  { label: 'MR – T1 (WC500/WW1000)',          wc: 500,  ww: 1000 },
  { label: 'MR – T2 FLAIR (WC1000/WW2000)',  wc: 1000, ww: 2000 },
]

const TOOLTIP_STYLE = { background: '#fff', border: '1px solid #e2e8f0', fontSize: 12, borderRadius: 6 }

function buildHistData(hist) {
  if (!hist) return []
  return hist.bins.map((b, i) => ({ bin: Math.round(b), count: hist.counts[i] }))
}

function buildCdfData(hist) {
  if (!hist) return []
  const counts = hist.counts
  const total  = counts.reduce((a, b) => a + b, 0) || 1
  let cum = 0
  return hist.bins.map((b, i) => {
    cum += counts[i]
    return { bin: Math.round(b), cdf: +(cum / total).toFixed(4) }
  })
}

export default function QCPlots({ imageId }) {
  const [features, setFeatures] = useState(null)
  const [loading,  setLoading]  = useState(false)
  const [error,    setError]    = useState(null)

  async function run() {
    if (!imageId) return
    setLoading(true); setError(null)
    try {
      const f = await api.features(imageId)
      setFeatures(f)
    } catch (e) { setError(e.message) }
    finally     { setLoading(false) }
  }

  const histData = buildHistData(features?.histogram)
  const cdfData  = buildCdfData(features?.histogram)

  return (
    <div className="page">
      <div className="page-header">
        <h2 className="page-title">📊 QC & Plots</h2>
        <p className="page-desc">
          Intensity histogram, cumulative distribution function (CDF), statistical
          feature summary, and window/level presets for contrast reference.
        </p>
      </div>

      {!imageId && (
        <div className="empty-state">
          <div className="empty-icon">📊</div>
          <div className="empty-title">No image loaded</div>
          <div className="empty-hint">Upload an image in Data Manager first.</div>
        </div>
      )}

      {imageId && (
        <>
          <div className="controls-bar">
            <button className="btn btn-primary" onClick={run} disabled={loading}>
              {loading ? <><span className="spinner" /> Analysing…</> : '📊 Run QC Analysis'}
            </button>
          </div>

          {error && <div className="alert alert-error" style={{ marginBottom: '1rem' }}>⚠️ {error}</div>}

          {/* Window/Level reference table */}
          <div className="section-label">Window / Level Presets (reference)</div>
          <div className="card" style={{ marginBottom: '1.5rem' }}>
            <div className="card-body" style={{ padding: '.75rem 1.1rem' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '.8rem' }}>
                <thead>
                  <tr>
                    {['Preset', 'Window Centre', 'Window Width'].map(h => (
                      <th key={h} style={{ textAlign: 'left', padding: '.35rem .6rem', color: 'var(--text-muted)', borderBottom: '1px solid var(--border)', fontSize: '.72rem', textTransform: 'uppercase' }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {WL_PRESETS.map((p, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid var(--border)' }}>
                      <td style={{ padding: '.35rem .6rem', color: 'var(--text)' }}>{p.label}</td>
                      <td style={{ padding: '.35rem .6rem', color: 'var(--accent-text)', fontWeight: 600 }}>{p.wc}</td>
                      <td style={{ padding: '.35rem .6rem', color: 'var(--accent-text)', fontWeight: 600 }}>{p.ww}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {features && (
            <>
              {/* Stat cards */}
              <div className="section-label">Intensity Statistics</div>
              <div className="stats-grid" style={{ marginBottom: '1.5rem' }}>
                {[
                  { label: 'Mean',      value: features.mean?.toFixed(2),                          unit: 'int.' },
                  { label: 'Std Dev',   value: features.std_dev?.toFixed(2),                       unit: 'int.' },
                  { label: 'Min',       value: features.intensity_min?.toFixed(0),                 unit: 'int.' },
                  { label: 'Max',       value: features.intensity_max?.toFixed(0),                 unit: 'int.' },
                  { label: 'Median',    value: features.percentile_50?.toFixed(2),                 unit: 'p50'  },
                  { label: 'Skewness', value: features.skewness?.toFixed(3),                      unit: ''     },
                  { label: 'Kurtosis', value: features.kurtosis?.toFixed(3),                      unit: ''     },
                  { label: 'Entropy',  value: features.entropy?.toFixed(3),                       unit: 'bits' },
                  { label: 'Non-zero', value: ((features.nonzero_fraction ?? 0)*100).toFixed(1),  unit: '%'    },
                  { label: 'Coverage', value: ((features.shape_descriptors?.foreground_coverage ?? 0)*100).toFixed(1), unit: '%' },
                ].map(s => (
                  <div key={s.label} className="stat-card">
                    <div className="stat-label">{s.label}</div>
                    <div className="stat-value">{s.value ?? '—'}</div>
                    {s.unit && <div className="stat-unit">{s.unit}</div>}
                  </div>
                ))}
              </div>

              {/* Percentiles */}
              <div className="section-label">Percentiles</div>
              <div className="percentile-row" style={{ marginBottom: '1.5rem' }}>
                {[['p10', features.percentile_10], ['p25', features.percentile_25], ['p50', features.percentile_50],
                  ['p75', features.percentile_75], ['p90', features.percentile_90]].map(([l, v]) => (
                  <div key={l} className="pct-chip">
                    <span className="pct-label">{l}</span>
                    <span className="pct-value">{v?.toFixed(1)}</span>
                  </div>
                ))}
              </div>

              {/* Charts */}
              <div className="chart-grid">
                <div className="chart-card">
                  <div className="chart-title">Intensity Histogram</div>
                  <ResponsiveContainer width="100%" height={200}>
                    <BarChart data={histData} margin={{ top: 4, right: 4, bottom: 4, left: 0 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                      <XAxis dataKey="bin" tick={{ fill: '#94a3b8', fontSize: 9 }} interval={15} />
                      <YAxis tick={{ fill: '#94a3b8', fontSize: 9 }} width={38} />
                      <Tooltip contentStyle={TOOLTIP_STYLE} formatter={v => [v.toLocaleString(), 'count']} />
                      <Bar dataKey="count" fill="#2563eb" radius={[2,2,0,0]} isAnimationActive={false} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>

                <div className="chart-card">
                  <div className="chart-title">CDF (Cumulative Distribution Function)</div>
                  <ResponsiveContainer width="100%" height={200}>
                    <AreaChart data={cdfData} margin={{ top: 4, right: 4, bottom: 4, left: 0 }}>
                      <defs>
                        <linearGradient id="cdfGrad" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%"  stopColor="#0d9488" stopOpacity={0.25} />
                          <stop offset="95%" stopColor="#0d9488" stopOpacity={0.02} />
                        </linearGradient>
                      </defs>
                      <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                      <XAxis dataKey="bin" tick={{ fill: '#94a3b8', fontSize: 9 }} interval={15} />
                      <YAxis tick={{ fill: '#94a3b8', fontSize: 9 }} width={38} domain={[0, 1]} />
                      <Tooltip contentStyle={TOOLTIP_STYLE} formatter={v => [(v * 100).toFixed(1) + '%', 'CDF']} />
                      <Area type="monotone" dataKey="cdf" stroke="#0d9488" fill="url(#cdfGrad)"
                        strokeWidth={2} isAnimationActive={false} />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </>
          )}
        </>
      )}
    </div>
  )
}
