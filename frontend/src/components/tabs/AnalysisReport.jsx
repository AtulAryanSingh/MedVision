/**
 * AnalysisReport.jsx
 *
 * Tab 5 – Analysis Report
 *
 * What it does:
 *   • Calls GET /api/report/{image_id} to get the structured report.
 *   • Displays image info, feature summary, cluster summary (if available),
 *     processing pipeline, and interpretation hints.
 *   • Provides a "Download JSON" button.
 *   • Shows a collapsible raw JSON viewer.
 *
 * Why it exists:
 *   A dedicated report tab gives users a single exportable summary of
 *   everything computed about their image.
 */

import { useState } from 'react'
import { api } from '../../api/client.js'

export default function AnalysisReport({ imageId }) {
  const [report,   setReport]   = useState(null)
  const [loading,  setLoading]  = useState(false)
  const [error,    setError]    = useState(null)

  async function fetchReport() {
    if (!imageId) return
    setLoading(true)
    setError(null)
    try {
      const r = await api.report(imageId)
      setReport(r)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  function downloadJson() {
    if (!report) return
    const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' })
    const url  = URL.createObjectURL(blob)
    const a    = Object.assign(document.createElement('a'), {
      href: url,
      download: `medvision-report-${report.image_id?.slice(0, 8)}.json`,
    })
    a.click()
    URL.revokeObjectURL(url)
  }

  return (
    <div className="tab-panel">
      <h2 className="section-title">Analysis Report</h2>

      <div className="controls-bar">
        <button className="btn primary" onClick={fetchReport} disabled={loading || !imageId}>
          {loading ? '⏳ Generating…' : '📋 Generate Report'}
        </button>
        {report && (
          <button className="btn secondary" onClick={downloadJson}>
            ⬇ Download JSON
          </button>
        )}
      </div>

      {error && <div className="alert error">{error}</div>}
      {!imageId && <div className="empty-state">Upload an image first.</div>}

      {report && (
        <div className="report-wrap">

          {/* ── Header ───────────────────────────────────────────────── */}
          <div className="report-header">
            <h3>{report.report_id}</h3>
            <span className="chip dim">{report.generated_at?.replace('T', ' ').slice(0, 19)} UTC</span>
          </div>

          {/* ── Image info ───────────────────────────────────────────── */}
          <ReportSection title="📁 Image Information">
            <KVGrid data={{
              'Filename':    report.image_info?.filename,
              'Format':      report.image_info?.file_type,
              'Modality':    report.image_info?.modality,
              'Shape':       report.image_info?.shape?.join(' × ') + ' px',
              '3-D volume':  report.image_info?.is_3d ? 'Yes' : 'No',
              'Spacing':     report.image_info?.spacing_mm?.map(s => s.toFixed(2)).join(' × ') + ' mm',
              'Int. min':    report.image_info?.intensity_range?.min?.toFixed(2),
              'Int. max':    report.image_info?.intensity_range?.max?.toFixed(2),
            }} />
          </ReportSection>

          {/* ── Feature summary ──────────────────────────────────────── */}
          {report.feature_summary && (
            <ReportSection title="📊 Feature Summary">
              <KVGrid data={{
                'Mean intensity':    report.feature_summary.mean_intensity?.toFixed(3),
                'Std deviation':     report.feature_summary.std_deviation?.toFixed(3),
                'Min intensity':     report.feature_summary.min_intensity?.toFixed(3),
                'Max intensity':     report.feature_summary.max_intensity?.toFixed(3),
                'Skewness':          report.feature_summary.skewness?.toFixed(4),
                'Kurtosis':          report.feature_summary.kurtosis?.toFixed(4),
                'Entropy (bits)':    report.feature_summary.entropy_bits?.toFixed(4),
                'Non-zero fraction': (report.feature_summary.nonzero_fraction * 100)?.toFixed(2) + '%',
                'Fg. coverage':      (report.feature_summary.foreground_coverage * 100)?.toFixed(2) + '%',
                'Eff. radius':       report.feature_summary.effective_radius_px?.toFixed(1) + ' px',
              }} />
            </ReportSection>
          )}

          {/* ── Cluster summary ──────────────────────────────────────── */}
          {report.cluster_summary && (
            <ReportSection title="🤖 Cluster Summary">
              <KVGrid data={{
                'k (clusters)':    report.cluster_summary.k,
                'Cluster centres': report.cluster_summary.cluster_centers?.map(c => c.toFixed(1)).join(', '),
                'Pixel counts':    report.cluster_summary.cluster_counts?.join(', '),
                'Dominant cluster': `#${(report.cluster_summary.dominant_cluster ?? 0) + 1}`,
              }} />
            </ReportSection>
          )}

          {/* ── Processing pipeline ──────────────────────────────────── */}
          {report.processing_pipeline?.length > 0 && (
            <ReportSection title="⚗️ Processing Pipeline">
              {report.processing_pipeline.map((step, i) => (
                <div key={i} className="pipeline-step">
                  <span className="step-num">{i + 1}</span>
                  <code>{step.type}</code>
                  <span className="step-params">
                    {Object.entries(step.params || {}).map(([k, v]) => `${k}=${v}`).join(' · ')}
                  </span>
                </div>
              ))}
            </ReportSection>
          )}

          {/* ── Interpretation ───────────────────────────────────────── */}
          <ReportSection title="💡 Interpretation">
            <ul className="interpretation-list">
              {report.interpretation?.map((hint, i) => (
                <li key={i}>{hint}</li>
              ))}
            </ul>
          </ReportSection>

          {/* ── Raw JSON ─────────────────────────────────────────────── */}
          <details className="json-details" style={{ marginTop: '1.5rem' }}>
            <summary>Full JSON report</summary>
            <pre>{JSON.stringify(report, null, 2)}</pre>
          </details>
        </div>
      )}
    </div>
  )
}

function ReportSection({ title, children }) {
  return (
    <section className="report-section">
      <h4 className="report-section-title">{title}</h4>
      {children}
    </section>
  )
}

function KVGrid({ data }) {
  return (
    <dl className="kv-grid">
      {Object.entries(data).filter(([, v]) => v != null).map(([k, v]) => (
        <div key={k} className="kv-row">
          <dt>{k}</dt>
          <dd>{String(v)}</dd>
        </div>
      ))}
    </dl>
  )
}
