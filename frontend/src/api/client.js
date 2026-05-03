/* MedVision v2 – api/client.js
 * Centralised fetch wrapper for all API calls.
 */

const BASE = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000'

async function _req(method, path, body, isForm = false) {
  const opts = { method, headers: {} }
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
    throw new Error(err.detail || `HTTP ${res.status}`)
  }
  return res.json()
}

export const api = {
  upload(file) {
    const fd = new FormData()
    fd.append('file', file)
    return _req('POST', '/api/upload', fd, true)
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
  },

  exportNpy(imageId) {
    return _req('GET', `/api/export/${imageId}/npy`)
  },
}
