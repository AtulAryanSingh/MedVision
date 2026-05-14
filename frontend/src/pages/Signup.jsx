import { useState } from 'react'
import { api } from '../api/client.js'

export default function Signup({ onSuccess, onSwitchToLogin }) {
  const [username, setUsername] = useState('')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [confirmPassword, setConfirmPassword] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  async function handleSubmit(e) {
    e.preventDefault()
    if (password !== confirmPassword) {
      setError('Passwords do not match.')
      return
    }

    setLoading(true)
    setError(null)
    try {
      await api.signup(username.trim(), email.trim(), password)
      const data = await api.login(username.trim(), password)
      onSuccess?.({ token: data.access_token, username: username.trim() })
    } catch (err) {
      setError(err.message || 'Signup failed.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="card" style={{ width: '100%', maxWidth: 420 }}>
      <div className="card-header">
        <span className="card-title">📝 Sign up</span>
      </div>
      <form className="card-body flex-col gap-md" onSubmit={handleSubmit}>
        <div className="form-row">
          <label className="form-label">Username</label>
          <input className="form-input" value={username} onChange={e => setUsername(e.target.value)} minLength={3} required />
        </div>
        <div className="form-row">
          <label className="form-label">Email</label>
          <input className="form-input" type="email" value={email} onChange={e => setEmail(e.target.value)} required />
        </div>
        <div className="form-row">
          <label className="form-label">Password</label>
          <input className="form-input" type="password" value={password} onChange={e => setPassword(e.target.value)} minLength={8} required />
        </div>
        <div className="form-row">
          <label className="form-label">Confirm password</label>
          <input
            className="form-input"
            type="password"
            value={confirmPassword}
            onChange={e => setConfirmPassword(e.target.value)}
            minLength={8}
            required
          />
        </div>
        {error && <div className="alert alert-error">⚠️ {error}</div>}
        <button className="btn btn-primary" type="submit" disabled={loading}>
          {loading ? <><span className="spinner" /> Creating account…</> : 'Create account'}
        </button>
      </form>
      <div className="card-footer" style={{ justifyContent: 'space-between' }}>
        <span className="text-muted">Already have an account?</span>
        <button className="btn btn-outline btn-sm" type="button" onClick={onSwitchToLogin}>
          Back to login
        </button>
      </div>
    </div>
  )
}
