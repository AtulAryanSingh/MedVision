/**
 * pages/DataManager.jsx – Upload a medical image and display its metadata
 */
import { useState, useRef } from 'react'
import { api } from '../api/client.js'

const ACCEPTED = '.png,.jpg,.jpeg,.dcm,.nii,.gz'

export default function DataManager({ imageId, metadata, onUpload, onReset }) {
  const [file,      setFile]      = useState(null)
  const [dragging,  setDragging]  = useState(false)
  const [uploading, setUploading] = useState(false)
  const [error,     setError]     = useState(null)
  const fileRef = useRef()

  function pickFile(f) {
    if (!f) return
    setFile(f)
    setError(null)
  }

  async function handleUpload() {
    if (!file) return
    setUploading(true)
    setError(null)
    try {
      const meta = await api.upload(file)
      onUpload(meta.image_id, meta)
    } catch (e) {
      setError(e.message)
    } finally {
      setUploading(false)
    }
  }

  const shape = metadata?.shape || []
  const sp    = metadata?.spacing || []

  return (
    <div className="page">
      <div className="page-header">
        <h2 className="page-title">📂 Data Manager</h2>
        <p className="page-desc">
          Upload a medical image file. DICOM, NIfTI, PNG, and JPEG are supported.
          For 3-D volumes the full array is preserved — use the Workspace to scroll through slices.
        </p>
      </div>

      {/* Drop zone */}
      {!imageId && (
        <div
          className={`drop-zone${dragging ? ' drag-active' : ''}`}
          style={{ maxWidth: 560 }}
          onClick={() => fileRef.current?.click()}
          onDragOver={e => { e.preventDefault(); setDragging(true) }}
          onDragLeave={() => setDragging(false)}
          onDrop={e => { e.preventDefault(); setDragging(false); pickFile(e.dataTransfer.files[0]) }}
        >
          <input ref={fileRef} type="file" accept={ACCEPTED} hidden onChange={e => pickFile(e.target.files[0])} />
          <div className="dz-icon">🏥</div>
          <div className="dz-text">{file ? `Ready: ${file.name}` : 'Drop a file here, or click to browse'}</div>
          <div className="dz-hint">PNG · JPG · DICOM (.dcm) · NIfTI (.nii / .nii.gz)</div>
        </div>
      )}

      {error && (
        <div className="alert alert-error" style={{ maxWidth: 560, marginTop: '1rem' }}>
          ⚠️ {error}
        </div>
      )}

      {!imageId && (
        <div style={{ marginTop: '1rem' }}>
          <button className="btn btn-primary" disabled={!file || uploading} onClick={handleUpload}>
            {uploading ? <><span className="spinner" /> Uploading…</> : '📤 Upload'}
          </button>
        </div>
      )}

      {/* Image already loaded – show info */}
      {imageId && metadata && (
        <>
          <div className="alert alert-success" style={{ maxWidth: 560, marginBottom: '1.25rem' }}>
            ✅ Image loaded — use the sidebar to explore it.
          </div>

          <div className="section-label">Image Information</div>
          <div className="card" style={{ maxWidth: 560 }}>
            <div className="card-body">
              <div className="meta-table">
                <span className="meta-key">File</span>
                <span className="meta-val">{metadata.filename}</span>
                <span className="meta-key">Format</span>
                <span className="meta-val">{metadata.file_type}</span>
                <span className="meta-key">Modality</span>
                <span className="meta-val">{metadata.modality}</span>
                <span className="meta-key">Shape</span>
                <span className="meta-val">{shape.join(' × ')} px</span>
                <span className="meta-key">3-D Volume</span>
                <span className="meta-val">{metadata.is_3d ? 'Yes' : 'No'}</span>
                <span className="meta-key">Voxel spacing</span>
                <span className="meta-val">{sp.map(s => s.toFixed(3)).join(' × ')} mm</span>
                {metadata.is_3d && (
                  <>
                    <span className="meta-key">FOV (z)</span>
                    <span className="meta-val">{(shape[0] * sp[0]).toFixed(1)} mm</span>
                    <span className="meta-key">FOV (y)</span>
                    <span className="meta-val">{(shape[1] * (sp[1] ?? 1)).toFixed(1)} mm</span>
                    <span className="meta-key">FOV (x)</span>
                    <span className="meta-val">{(shape[2] * (sp[2] ?? 1)).toFixed(1)} mm</span>
                  </>
                )}
                <span className="meta-key">Intensity range</span>
                <span className="meta-val">{metadata.intensity_min?.toFixed(0)} – {metadata.intensity_max?.toFixed(0)}</span>
                <span className="meta-key">File size</span>
                <span className="meta-val">{(metadata.size_bytes / 1024).toFixed(1)} KB</span>
              </div>
            </div>
            <div className="card-footer">
              <button className="btn btn-outline btn-sm" onClick={() => { setFile(null); setError(null); onReset?.() }}>
                ↺ Upload a different file
              </button>
            </div>
          </div>

          {metadata.extra_meta && Object.keys(metadata.extra_meta).length > 0 && (
            <div style={{ marginTop: '1.25rem' }}>
              <div className="section-label">DICOM / Format Metadata</div>
              <div className="card" style={{ maxWidth: 560 }}>
                <div className="card-body">
                  <div className="meta-table">
                    {Object.entries(metadata.extra_meta)
                      .filter(([, v]) => v && String(v).trim())
                      .flatMap(([k, v]) => [
                        <span key={`meta-k-${k}`} className="meta-key">{k.replace(/_/g, ' ')}</span>,
                        <span key={`meta-v-${k}`} className="meta-val">{String(v)}</span>,
                      ])}
                  </div>
                </div>
              </div>
            </div>
          )}
        </>
      )}
    </div>
  )
}
