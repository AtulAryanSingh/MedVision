/**
 * UploadViewer.jsx
 *
 * Tab 1 – Upload & Viewer
 *
 * What it does:
 *   • Drag-and-drop / click-to-browse file selector.
 *   • Uploads the selected file to POST /api/upload.
 *   • Displays metadata (shape, modality, spacing, etc.).
 *   • Fetches and displays axial/sagittal/coronal slice previews.
 *   • For 3-D volumes: slice-index sliders let the user browse the volume.
 *
 * Why it exists:
 *   Uploading is the entry point for all downstream operations; the viewer
 *   gives immediate visual confirmation that the image loaded correctly.
 */

import { useState, useRef } from 'react'
import { api } from '../../api/client.js'

const ACCEPTED = '.png,.jpg,.jpeg,.dcm,.nii,.gz'

export default function UploadViewer({ imageId, metadata, onUpload }) {
  const [file,        setFile]        = useState(null)
  const [dragging,    setDragging]    = useState(false)
  const [uploading,   setUploading]   = useState(false)
  const [preview,     setPreview]     = useState(null)
  const [slices,      setSlices]      = useState({ axial: 0, coronal: 0, sagittal: 0 })
  const [loadingPrev, setLoadingPrev] = useState(false)
  const [error,       setError]       = useState(null)
  const fileRef = useRef()

  /* ── File selection ─────────────────────────────────────────────── */
  function pickFile(f) {
    if (!f) return
    setFile(f)
    setError(null)
  }

  /* ── Upload ─────────────────────────────────────────────────────── */
  async function handleUpload() {
    if (!file) return
    setUploading(true)
    setError(null)
    try {
      const meta = await api.upload(file)
      onUpload(meta.image_id, meta)
      await fetchPreview(meta.image_id, meta.shape, { axial: 0, coronal: 0, sagittal: 0 })
    } catch (e) {
      setError(e.message)
    } finally {
      setUploading(false)
    }
  }

  /* ── Preview ────────────────────────────────────────────────────── */
  async function fetchPreview(id, shape, sl) {
    setLoadingPrev(true)
    try {
      const is3d = shape && shape[0] > 4 && shape.length === 3 && shape[2] > 4
      const idxs = is3d
        ? { axialIdx: sl.axial, coronalIdx: sl.coronal, sagittalIdx: sl.sagittal }
        : {}
      const p = await api.preview(id, idxs)
      setPreview(p)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoadingPrev(false)
    }
  }

  /* ── Slice slider handler ────────────────────────────────────────── */
  async function onSliceChange(axis, value) {
    const newSl = { ...slices, [axis]: Number(value) }
    setSlices(newSl)
    if (imageId && metadata) {
      await fetchPreview(imageId, metadata.shape, newSl)
    }
  }

  const shape = metadata?.shape || []
  const is3d  = metadata?.is_3d || false

  return (
    <div className="tab-panel">
      <h2 className="section-title">Upload & Viewer</h2>

      {/* ── Drop zone ───────────────────────────────────────────────── */}
      {!imageId && (
        <div
          className={`drop-zone${dragging ? ' drag-over' : ''}`}
          onClick={() => fileRef.current?.click()}
          onDragOver={e => { e.preventDefault(); setDragging(true) }}
          onDragLeave={() => setDragging(false)}
          onDrop={e => { e.preventDefault(); setDragging(false); pickFile(e.dataTransfer.files[0]) }}
        >
          <input
            ref={fileRef}
            type="file"
            accept={ACCEPTED}
            hidden
            onChange={e => pickFile(e.target.files[0])}
          />
          <div className="drop-icon">🏥</div>
          <p className="drop-text">
            {file ? file.name : 'Drop a medical image here, or click to browse'}
          </p>
          <p className="drop-hint">PNG · JPG · DICOM (.dcm) · NIfTI (.nii / .nii.gz)</p>
        </div>
      )}

      {error && <div className="alert error">{error}</div>}

      {/* ── Upload button ───────────────────────────────────────────── */}
      {!imageId && (
        <button
          className="btn primary"
          disabled={!file || uploading}
          onClick={handleUpload}
        >
          {uploading ? '⏳ Uploading…' : '📤 Upload Image'}
        </button>
      )}

      {/* ── Metadata card ───────────────────────────────────────────── */}
      {metadata && (
        <div className="meta-card">
          <div className="meta-grid">
            <MetaItem label="File"      value={metadata.filename} />
            <MetaItem label="Format"    value={metadata.file_type} />
            <MetaItem label="Modality"  value={metadata.modality} />
            <MetaItem label="Shape"     value={shape.join(' × ') + ' px'} />
            <MetaItem label="3-D Vol."  value={is3d ? 'Yes' : 'No'} />
            <MetaItem label="Int. range"
              value={`${metadata.intensity_min?.toFixed(0)} – ${metadata.intensity_max?.toFixed(0)}`} />
            <MetaItem label="Spacing"   value={metadata.spacing?.map(s => s.toFixed(2)).join(' × ') + ' mm'} />
            <MetaItem label="Size"      value={`${(metadata.size_bytes / 1024).toFixed(1)} KB`} />
          </div>
        </div>
      )}

      {/* ── Slice viewer ────────────────────────────────────────────── */}
      {imageId && (
        <>
          <h3 className="sub-title">Slice Preview {loadingPrev && <span className="spinner-inline">⏳</span>}</h3>

          {/* Sliders for 3-D volumes */}
          {is3d && preview && (
            <div className="slider-row">
              {[
                { axis: 'axial',    label: 'Axial (Z)',    max: shape[0] - 1 },
                { axis: 'coronal',  label: 'Coronal (Y)',  max: shape[1] - 1 },
                { axis: 'sagittal', label: 'Sagittal (X)', max: shape[2] - 1 },
              ].map(s => (
                <label key={s.axis} className="slider-label">
                  {s.label} [{slices[s.axis]}]
                  <input
                    type="range" min={0} max={s.max} value={slices[s.axis]}
                    onChange={e => onSliceChange(s.axis, e.target.value)}
                  />
                </label>
              ))}
            </div>
          )}

          {preview && (
            <div className="slice-grid">
              {[
                { key: 'axial',    label: is3d ? 'Axial'    : 'Image' },
                { key: 'coronal',  label: is3d ? 'Coronal'  : 'Image' },
                { key: 'sagittal', label: is3d ? 'Sagittal' : 'Image' },
              ].map(o => (
                <figure key={o.key} className="slice-figure">
                  <figcaption>{o.label}</figcaption>
                  <img src={preview[o.key]} alt={o.label} className="slice-img" />
                </figure>
              ))}
            </div>
          )}

          {!preview && !loadingPrev && (
            <button className="btn secondary" onClick={() => fetchPreview(imageId, shape, slices)}>
              🔍 Load Preview
            </button>
          )}
        </>
      )}
    </div>
  )
}

function MetaItem({ label, value }) {
  return (
    <div className="meta-item">
      <span className="meta-label">{label}</span>
      <span className="meta-value">{value}</span>
    </div>
  )
}
