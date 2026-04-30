/* MedVision – api/client.js
 *
 * What this module does:
 *   Provides a thin wrapper around fetch() so every tab component uses the
 *   same base URL and error-handling pattern.
 *
 * Why it exists:
 *   Centralising all API calls here means only one place needs to change if
 *   the backend URL or auth headers change.
 */

const BASE = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000'

async function _request(method, path, body, isFormData = false) {
  const opts = { method, headers: {} }
  if (body) {
    if (isFormData) {
      opts.body = body                            // FormData sets Content-Type automatically
    } else {
      opts.headers['Content-Type'] = 'application/json'
      opts.body = JSON.stringify(body)
    }
  }
  const res = await fetch(`${BASE}${path}`, opts)
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(err.detail || `HTTP ${res.status}`)
  }
  return res.json()
}

export const api = {
  /** Upload a File object; returns metadata + image_id */
  upload(file) {
    const fd = new FormData()
    fd.append('file', file)
    return _request('POST', '/api/upload', fd, true)
  },

  /** Get axial/sagittal/coronal slice base64 PNGs */
  preview(imageId, { axialIdx, coronalIdx, sagittalIdx } = {}) {
    const params = new URLSearchParams()
    if (axialIdx    != null) params.set('axial_idx',    axialIdx)
    if (coronalIdx  != null) params.set('coronal_idx',  coronalIdx)
    if (sagittalIdx != null) params.set('sagittal_idx', sagittalIdx)
    const qs = params.toString() ? `?${params}` : ''
    return _request('GET', `/api/preview/${imageId}${qs}`)
  },

  /** Run a processing operation; returns result_image + histograms */
  process(imageId, processingType, params = {}) {
    return _request('POST', '/api/process', { image_id: imageId, processing_type: processingType, ...params })
  },

  /** Extract feature vector */
  features(imageId) {
    return _request('POST', '/api/features', { image_id: imageId })
  },

  /** Run KMeans + PCA */
  cluster(imageId, k = 4, nSamples = 5000) {
    return _request('POST', '/api/cluster', { image_id: imageId, k, n_samples: nSamples })
  },

  /** Get structured analysis report */
  report(imageId) {
    return _request('GET', `/api/report/${imageId}`)
  },
}
