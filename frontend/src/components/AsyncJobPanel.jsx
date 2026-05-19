import { JOB_TERMINAL_STATES } from '../constants/async.js'

export default function AsyncJobPanel({ job, onRefresh, onCancel, onRetry, title = 'Async job' }) {
  if (!job) return null

  const status = job.status || 'queued'
  const terminal = JOB_TERMINAL_STATES.includes(status)
  const progress = Number.isFinite(job.progress) ? Math.max(0, Math.min(100, job.progress)) : 0

  return (
    <div className="card" style={{ marginBottom: '1rem' }}>
      <div className="card-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <span className="card-title">⏱ {title}</span>
        <div style={{ display: 'flex', gap: '.45rem' }}>
          {!terminal && (
            <button className="btn btn-outline btn-sm" onClick={onCancel}>Cancel</button>
          )}
          <button className="btn btn-outline btn-sm" onClick={onRefresh}>Refresh</button>
          {terminal && onRetry && (
            <button className="btn btn-primary btn-sm" onClick={onRetry}>Retry</button>
          )}
        </div>
      </div>
      <div className="card-body" style={{ paddingTop: '.75rem' }}>
        <div style={{ display: 'flex', gap: '.5rem', flexWrap: 'wrap', marginBottom: '.55rem' }}>
          <span className={`chip ${status === 'succeeded' ? 'green' : status === 'failed' ? 'red' : status === 'canceled' ? '' : 'blue'}`}>
            {status}
          </span>
          <span className="chip">{progress}%</span>
          <span className="chip">{job.operation}</span>
          {job.job_id && <span className="chip">{job.job_id.slice(0, 8)}</span>}
        </div>
        <div style={{ height: 8, borderRadius: 999, background: 'var(--surface2)', overflow: 'hidden' }}>
          <div style={{ width: `${progress}%`, height: '100%', background: 'var(--accent)', transition: 'width .25s ease' }} />
        </div>
        <div style={{ marginTop: '.45rem', fontSize: '.78rem', color: 'var(--text-secondary)' }}>
          {job.message || 'Processing'}
        </div>
        {job.error?.detail && (
          <div className="alert alert-error" style={{ marginTop: '.6rem', fontSize: '.78rem' }}>
            ⚠️ {job.error.detail}
          </div>
        )}
      </div>
    </div>
  )
}
