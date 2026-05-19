/* MedVision v2 – api/client.js
 * Centralised fetch wrapper for all API calls.
 */
import { getAuthToken } from '../auth.js'

const BASE = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000'

async function _req(method, path, body, isForm = false) {
  const opts = { method, headers: {} }
  const token = getAuthToken()
  if (token) {
    opts.headers.Authorization = `Bearer ${token}`
  }
  if (body) {
    if (isForm) {
      opts.body = body
    } else {
      opts.headers['Content-Type'] = 'application/json'
      opts.body = JSON.stringify(body)
    }
  }
  const res = await fetch(`${BASE}${path}`, opts)
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(err.message || err.detail || `HTTP ${res.status}`)
  }
  const payload = await res.json().catch(() => null)
  if (payload && payload.status === 'ok' && payload.data !== undefined) {
    return payload.data
  }
  return payload
}

export const api = {
  login(username, password) {
    const form = new URLSearchParams()
    form.set('username', username)
    form.set('password', password)
    return _req('POST', '/api/auth/login', form, true)
  },

  signup(username, email, password) {
    return _req('POST', '/api/auth/register', { username, email, password })
  },

  upload(file) {
    const fd = new FormData()
    fd.append('file', file)
    return _req('POST', '/api/upload', fd, true)
  },

  uploadSeries(files) {
    const fd = new FormData()
    for (const f of files) {
      fd.append('files', f)
    }
    return _req('POST', '/api/upload-series', fd, true)
  },

  preview(imageId, { axialIdx, coronalIdx, sagittalIdx } = {}) {
    const p = new URLSearchParams()
    if (axialIdx    != null) p.set('axial_idx',    axialIdx)
    if (coronalIdx  != null) p.set('coronal_idx',  coronalIdx)
    if (sagittalIdx != null) p.set('sagittal_idx', sagittalIdx)
    return _req('GET', `/api/preview/${imageId}${p.toString() ? '?' + p : ''}`)
  },

  mpr(imageId, { axialIdx, coronalIdx, sagittalIdx, windowCenter, windowWidth, maxDim } = {}) {
    const p = new URLSearchParams()
    if (axialIdx     != null) p.set('axial_idx',     axialIdx)
    if (coronalIdx   != null) p.set('coronal_idx',   coronalIdx)
    if (sagittalIdx  != null) p.set('sagittal_idx',  sagittalIdx)
    if (windowCenter != null) p.set('window_center', windowCenter)
    if (windowWidth  != null) p.set('window_width',  windowWidth)
    if (maxDim       != null) p.set('max_dim',       maxDim)
    return _req('GET', `/api/mpr/${imageId}${p.toString() ? '?' + p : ''}`)
  },

  process(imageId, processingType, params = {}) {
    return _req('POST', '/api/process', { image_id: imageId, processing_type: processingType, ...params })
  },

  features(imageId) {
    return _req('POST', '/api/features', { image_id: imageId })
  },

  cluster(imageId, k = 4, nSamples = 5000) {
    return _req('POST', '/api/cluster', { image_id: imageId, k, n_samples: nSamples })
  },

  report(imageId) {
    return _req('GET', `/api/report/${imageId}`)
  },

  patchify(imageId, patchSize = 32, stride = 16) {
    return _req('POST', '/api/patchify', { image_id: imageId, patch_size: patchSize, stride })
  },

  register(imageId, transformType, params = {}) {
    return _req('POST', '/api/register', { image_id: imageId, transform_type: transformType, ...params })
  },

  exportUrl: {
    png: (imageId) => `${BASE}/api/export/${imageId}/png`,
    csv: (imageId) => `${BASE}/api/export/${imageId}/csv`,
    npyStream: (imageId) => `${BASE}/api/export/${imageId}/npy/stream`,
  },

  exportNpy(imageId) {
    return _req('GET', `/api/export/${imageId}/npy`)
  },

  createPatchifyJob(imageId, patchSize = 32, stride = 16) {
    return _req('POST', '/api/jobs/patchify', { image_id: imageId, patch_size: patchSize, stride })
  },

  createRegisterJob(imageId, transformType, params = {}) {
    return _req('POST', '/api/jobs/register', { image_id: imageId, transform_type: transformType, ...params })
  },

  createClusterJob(imageId, k = 4, nSamples = 5000) {
    return _req('POST', '/api/jobs/cluster', { image_id: imageId, k, n_samples: nSamples })
  },

  createReportJob(imageId) {
    return _req('POST', `/api/jobs/report/${imageId}`)
  },

  getJob(jobId) {
    return _req('GET', `/api/jobs/${jobId}`)
  },

  cancelJob(jobId) {
    return _req('POST', `/api/jobs/${jobId}/cancel`)
  },

  getJobResult(jobId) {
    return _req('GET', `/api/jobs/${jobId}/result`)
  },

  jobResultStreamUrl(jobId) {
    return `${BASE}/api/jobs/${jobId}/result/stream`
  },
}
