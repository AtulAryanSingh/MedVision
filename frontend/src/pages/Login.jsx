import { useState } from 'react'
import { api } from '../api/client.js'

export default function Login({ onSuccess, onSwitchToSignup }) {
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  async function handleSubmit(e) {
    e.preventDefault()
    setLoading(true)
    setError(null)
    try {
      const data = await api.login(username.trim(), password)
      onSuccess?.({ token: data.access_token, username: username.trim() })
    } catch (err) {
      setError(err.message || 'Login failed.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="card" style={{ width: '100%', maxWidth: 420 }}>
      <div className="card-header">
        <span className="card-title">🔐 Login</span>
      </div>
      <form className="card-body flex-col gap-md" onSubmit={handleSubmit}>
        <div className="form-row">
          <label className="form-label">Username</label>
          <input className="form-input" value={username} onChange={e => setUsername(e.target.value)} required />
        </div>
        <div className="form-row">
          <label className="form-label">Password</label>
          <input
            className="form-input"
            type="password"
            value={password}
            onChange={e => setPassword(e.target.value)}
            required
          />
        </div>
        {error && <div className="alert alert-error">⚠️ {error}</div>}
        <button className="btn btn-primary" type="submit" disabled={loading}>
          {loading ? <><span className="spinner" /> Signing in…</> : 'Sign in'}
        </button>
      </form>
      <div className="card-footer" style={{ justifyContent: 'space-between' }}>
        <span className="text-muted">No account yet?</span>
        <button className="btn btn-outline btn-sm" type="button" onClick={onSwitchToSignup}>
          Create account
        </button>
      </div>
    </div>
  )
}
