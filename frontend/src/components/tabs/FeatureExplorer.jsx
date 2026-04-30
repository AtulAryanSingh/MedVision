/**
 * FeatureExplorer.jsx
 *
 * Tab 3 – Feature Explorer
 *
 * What it does:
 *   • Calls POST /api/features to compute the statistical feature vector.
 *   • Displays feature cards (mean, std, min, max, skewness, kurtosis,
 *     entropy, non-zero fraction, foreground coverage).
 *   • Renders a 64-bin intensity histogram using Recharts.
 *   • Shows a collapsible raw JSON response.
 *
 * Why it exists:
 *   Numerical features give a quantitative fingerprint of the image that
 *   complements the visual slice view.
 */

import { useState } from 'react'
import {
  AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid
} from 'recharts'
import { api } from '../../api/client.js'

export default function FeatureExplorer({ imageId }) {
  const [features, setFeatures] = useState(null)
  const [loading,  setLoading]  = useState(false)
  const [error,    setError]    = useState(null)
  const [showJson, setShowJson] = useState(false)

  async function runExtract() {
    if (!imageId) return
    setLoading(true)
    setError(null)
    try {
      const f = await api.features(imageId)
      setFeatures(f)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  const histData = features?.histogram
    ? features.histogram.bins.map((b, i) => ({ bin: Math.round(b), count: features.histogram.counts[i] }))
    : []

  return (
    <div className="tab-panel">
      <h2 className="section-title">Feature Explorer</h2>

      <button className="btn primary" onClick={runExtract} disabled={loading || !imageId}>
        {loading ? '⏳ Extracting…' : '📊 Extract Features'}
      </button>

      {error && <div className="alert error">{error}</div>}

      {!imageId && <div className="empty-state">Upload an image first.</div>}

      {features && (
        <>
          {/* ── Stat cards ──────────────────────────────────────────── */}
          <div className="feature-grid">
            <StatCard label="Mean"       value={features.mean?.toFixed(2)}         unit="intensity" />
            <StatCard label="Std Dev"    value={features.std_dev?.toFixed(2)}       unit="intensity" />
            <StatCard label="Min"        value={features.intensity_min?.toFixed(2)} unit="intensity" />
            <StatCard label="Max"        value={features.intensity_max?.toFixed(2)} unit="intensity" />
            <StatCard label="Median"     value={features.percentile_50?.toFixed(2)} unit="p50" />
            <StatCard label="Skewness"   value={features.skewness?.toFixed(3)}      unit="" />
            <StatCard label="Kurtosis"   value={features.kurtosis?.toFixed(3)}      unit="" />
            <StatCard label="Entropy"    value={features.entropy?.toFixed(3)}       unit="bits" />
            <StatCard label="Non-zero"   value={(features.nonzero_fraction * 100)?.toFixed(1)} unit="%" />
            <StatCard label="Coverage"   value={(features.shape_descriptors?.foreground_coverage * 100)?.toFixed(1)} unit="%" />
            <StatCard label="Eff. Radius" value={features.shape_descriptors?.effective_radius_px?.toFixed(1)} unit="px" />
          </div>

          {/* ── Percentile bar ──────────────────────────────────────── */}
          <h3 className="sub-title">Percentiles</h3>
          <div className="percentile-row">
            {[
              ['p10', features.percentile_10],
              ['p25', features.percentile_25],
              ['p50', features.percentile_50],
              ['p75', features.percentile_75],
              ['p90', features.percentile_90],
            ].map(([label, val]) => (
              <div key={label} className="pct-chip">
                <span className="pct-label">{label}</span>
                <span className="pct-value">{val?.toFixed(1)}</span>
              </div>
            ))}
          </div>

          {/* ── Histogram ───────────────────────────────────────────── */}
          <h3 className="sub-title">Intensity Distribution</h3>
          <div className="chart-wrap">
            <ResponsiveContainer width="100%" height={200}>
              <AreaChart data={histData} margin={{ top: 8, right: 8, bottom: 8, left: 8 }}>
                <defs>
                  <linearGradient id="histGrad" x1="0" y1="0" x2="1" y2="0">
                    <stop offset="0%"   stopColor="#0f9b8e" stopOpacity={0.9} />
                    <stop offset="100%" stopColor="#7c4dff" stopOpacity={0.9} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#2a2a4a" />
                <XAxis dataKey="bin" tick={{ fill: '#888', fontSize: 10 }} interval={7} label={{ value: 'Intensity', position: 'insideBottom', offset: -4, fill: '#666', fontSize: 11 }} />
                <YAxis tick={{ fill: '#888', fontSize: 10 }} width={45} />
                <Tooltip
                  contentStyle={{ background: '#1a1a2e', border: '1px solid #2a2a4a', fontSize: 12 }}
                  formatter={v => [v.toLocaleString(), 'pixels']}
                />
                <Area type="monotone" dataKey="count" stroke="#0f9b8e" fill="url(#histGrad)"
                  strokeWidth={1.5} isAnimationActive={false} />
              </AreaChart>
            </ResponsiveContainer>
          </div>

          {/* ── Raw JSON ────────────────────────────────────────────── */}
          <details className="json-details" open={showJson} onToggle={e => setShowJson(e.target.open)}>
            <summary>Raw JSON response</summary>
            <pre>{JSON.stringify(features, null, 2)}</pre>
          </details>
        </>
      )}
    </div>
  )
}

function StatCard({ label, value, unit }) {
  return (
    <div className="stat-card">
      <div className="stat-label">{label}</div>
      <div className="stat-value">{value ?? '—'}</div>
      {unit && <div className="stat-unit">{unit}</div>}
    </div>
  )
}
