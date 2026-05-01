/**
 * MLLab.jsx
 *
 * Tab 4 – ML Lab
 *
 * What it does:
 *   • K control (slider 2–12) and sample-size selector.
 *   • Calls POST /api/cluster and shows:
 *     – Colour-coded KMeans segmented image.
 *     – Cluster statistics table (centre intensity, pixel count, %).
 *     – PCA 2-D scatter plot coloured by cluster label.
 *   • Explained-variance annotation on the scatter chart axes.
 *
 * Why it exists:
 *   KMeans + PCA is the standard unsupervised starting point for
 *   exploratory tissue segmentation in medical imaging.
 */

import { useState, useMemo } from 'react'
import {
  ScatterChart, Scatter, XAxis, YAxis, ZAxis,
  Tooltip, ResponsiveContainer, CartesianGrid, Legend
} from 'recharts'
import { api } from '../../api/client.js'

const CLUSTER_COLORS = [
  '#1de9b6', '#7c4dff', '#ff6d00', '#00b0ff',
  '#f50057', '#76ff03', '#ffea00', '#e040fb',
  '#00e5ff', '#ff9100', '#b2ff59', '#40c4ff',
]

export default function MLLab({ imageId }) {
  const [k,        setK]        = useState(4)
  const [nSamples, setNSamples] = useState(3000)
  const [result,   setResult]   = useState(null)
  const [loading,  setLoading]  = useState(false)
  const [error,    setError]    = useState(null)

  async function runCluster() {
    if (!imageId) return
    setLoading(true)
    setError(null)
    try {
      const r = await api.cluster(imageId, k, nSamples)
      setResult(r)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  /* ── PCA scatter data grouped by cluster label ─────────────────── */
  const scatterSeries = useMemo(() => {
    if (!result?.pca?.points) return []
    const byCluster = {}
    result.pca.points.forEach(p => {
      if (!byCluster[p.cluster]) byCluster[p.cluster] = []
      byCluster[p.cluster].push({ x: p.x, y: p.y })
    })
    return Object.entries(byCluster).map(([cl, pts]) => ({
      cluster: Number(cl),
      data: pts,
      color: CLUSTER_COLORS[Number(cl) % CLUSTER_COLORS.length],
    }))
  }, [result])

  const totalPx = result?.cluster_counts?.reduce((a, b) => a + b, 0) || 1
  const ev = result?.pca?.explained_variance || []

  return (
    <div className="tab-panel">
      <h2 className="section-title">ML Lab — KMeans + PCA</h2>

      {/* ── Controls ─────────────────────────────────────────────────── */}
      <div className="controls-bar">
        <label className="ctrl-label">
          Clusters (k): {k}
          <input type="range" min={2} max={12} step={1} value={k}
            onChange={e => setK(Number(e.target.value))} />
        </label>
        <label className="ctrl-label">
          PCA samples: {nSamples.toLocaleString()}
          <input type="range" min={500} max={10000} step={500} value={nSamples}
            onChange={e => setNSamples(Number(e.target.value))} />
        </label>
        <button className="btn primary" onClick={runCluster} disabled={loading || !imageId}>
          {loading ? '⏳ Running…' : '🤖 Run KMeans + PCA'}
        </button>
      </div>

      {error && <div className="alert error">{error}</div>}
      {!imageId && <div className="empty-state">Upload an image first.</div>}

      {result && (
        <>
          {/* ── Segmented image ──────────────────────────────────────── */}
          <div className="compare-grid">
            <figure className="img-card">
              <figcaption>KMeans Segmentation (k={result.k})</figcaption>
              <img src={result.segmented_image} alt="segmented" />
            </figure>

            {/* ── Cluster stats table ──────────────────────────────── */}
            <div className="cluster-table-wrap">
              <p className="sub-title" style={{ marginTop: 0 }}>Cluster Statistics</p>
              <table className="cluster-table">
                <thead>
                  <tr><th>#</th><th>Centre</th><th>Pixels</th><th>%</th></tr>
                </thead>
                <tbody>
                  {result.centers.map((c, i) => (
                    <tr key={i}>
                      <td>
                        <span className="cluster-dot"
                          style={{ background: CLUSTER_COLORS[i % CLUSTER_COLORS.length] }} />
                        {i + 1}
                      </td>
                      <td>{c.toFixed(1)}</td>
                      <td>{result.cluster_counts[i]?.toLocaleString()}</td>
                      <td>{((result.cluster_counts[i] / totalPx) * 100).toFixed(1)}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* ── PCA scatter ──────────────────────────────────────────── */}
          <h3 className="sub-title">
            PCA 2-D Projection
            {ev.length >= 2 && (
              <span className="ev-note">
                &nbsp;(PC1={( ev[0]*100).toFixed(1)}% · PC2={(ev[1]*100).toFixed(1)}% variance)
              </span>
            )}
          </h3>
          <div className="chart-wrap">
            <ResponsiveContainer width="100%" height={320}>
              <ScatterChart margin={{ top: 8, right: 16, bottom: 24, left: 8 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#2a2a4a" />
                <XAxis type="number" dataKey="x" name="PC1"
                  tick={{ fill: '#888', fontSize: 10 }}
                  label={{ value: ev[0] ? `PC1 (${(ev[0]*100).toFixed(1)}%)` : 'PC1', position: 'insideBottom', offset: -12, fill: '#888', fontSize: 11 }} />
                <YAxis type="number" dataKey="y" name="PC2"
                  tick={{ fill: '#888', fontSize: 10 }}
                  label={{ value: ev[1] ? `PC2 (${(ev[1]*100).toFixed(1)}%)` : 'PC2', angle: -90, position: 'insideLeft', fill: '#888', fontSize: 11 }} />
                <ZAxis range={[6, 6]} />
                <Tooltip
                  cursor={{ strokeDasharray: '3 3' }}
                  contentStyle={{ background: '#1a1a2e', border: '1px solid #2a2a4a', fontSize: 11 }}
                  formatter={(v, n) => [v.toFixed(3), n]}
                />
                <Legend wrapperStyle={{ fontSize: 12, color: '#aaa' }} />
                {scatterSeries.map(s => (
                  <Scatter
                    key={s.cluster}
                    name={`Cluster ${s.cluster + 1}`}
                    data={s.data}
                    fill={s.color}
                    opacity={0.65}
                    isAnimationActive={false}
                  />
                ))}
              </ScatterChart>
            </ResponsiveContainer>
          </div>
          <p className="caption">
            {result.pca.n_samples_used?.toLocaleString()} pixels sampled · coloured by KMeans cluster label
          </p>
        </>
      )}
    </div>
  )
}
