/**
 * pages/Home.jsx – Landing page with feature cards
 */
export default function Home({ onNavigate }) {
  const FEATURES = [
    {
      id: 'data',
      icon: '📂',
      name: 'Data Manager',
      desc: 'Upload DICOM, NIfTI, PNG/JPEG files. Supports 2-D slices and full 3-D volumes.',
    },
    {
      id: 'workspace',
      icon: '🔬',
      name: 'Workspace / MPR',
      desc: 'Multi-Planar Reconstruction with spacing-correct aspect ratios. Axial · Coronal · Sagittal.',
    },
    {
      id: 'qc',
      icon: '📊',
      name: 'QC & Plots',
      desc: 'Window/Level contrast presets, intensity histogram and CDF, FOV display, intensity statistics.',
    },
    {
      id: 'tools',
      icon: '⚗️',
      name: 'Modular Tools',
      desc: 'Gaussian, Median, Sobel, morphology (erosion/dilation/open/close), CDF threshold, connected components. Each tool runs independently.',
    },
    {
      id: 'patchify',
      icon: '🧩',
      name: 'Patchify 3D',
      desc: 'Slice a 3-D volume into cubic patches with configurable size and stride. Download as NPZ.',
    },
    {
      id: 'register',
      icon: '📐',
      name: 'Registration & Resampling',
      desc: 'Apply geometric transforms (translate, rotate, zoom, affine) with bicubic or nearest-neighbour interpolation.',
    },
    {
      id: 'dl',
      icon: '🧬',
      name: 'Deep Learning',
      desc: 'Export dataset for training. Download a ready-to-run Google Colab notebook for 3-D U-Net / CNN.',
    },
    {
      id: 'exports',
      icon: '📥',
      name: 'Downloads',
      desc: 'Export results as PNG, NumPy arrays (.npy), volume patches (.npz), or component metrics (.csv).',
    },
  ]

  return (
    <div className="page">
      {/* Hero */}
      <div style={{ marginBottom: '2.5rem' }}>
        <h1 style={{ fontSize: '1.9rem', fontWeight: 800, letterSpacing: '-.5px', color: 'var(--text)', lineHeight: 1.15 }}>
          MedVision
        </h1>
        <p style={{ marginTop: '.6rem', fontSize: '1rem', color: 'var(--text-secondary)', maxWidth: 580, lineHeight: 1.7 }}>
          A modular medical-imaging workbench. Upload any clinical image, explore it with
          clinically-accurate Multi-Planar Reconstruction, run independent tools in any order,
          and export results for downstream analysis or deep-learning pipelines.
        </p>
        <div style={{ marginTop: '1.2rem', display: 'flex', gap: '.75rem', flexWrap: 'wrap' }}>
          <button className="btn btn-primary btn-lg" onClick={() => onNavigate('data')}>
            📂 Upload Image
          </button>
          <button className="btn btn-outline btn-lg" onClick={() => onNavigate('workspace')}>
            🔬 Open Workspace
          </button>
        </div>
      </div>

      {/* Feature cards */}
      <div className="section-label">Capabilities</div>
      <div className="features-grid">
        {FEATURES.map(f => (
          <div key={f.id} className="feat-card" onClick={() => onNavigate(f.id)}>
            <div className="feat-card-icon">{f.icon}</div>
            <div className="feat-card-name">{f.name}</div>
            <div className="feat-card-desc">{f.desc}</div>
          </div>
        ))}
      </div>

      {/* Supported formats */}
      <div style={{ marginTop: '2.5rem' }}>
        <div className="section-label">Supported formats</div>
        <div style={{ display: 'flex', gap: '.5rem', flexWrap: 'wrap', marginTop: '.5rem' }}>
          {['DICOM (.dcm)', 'NIfTI (.nii / .nii.gz)', 'PNG', 'JPEG'].map(f => (
            <span key={f} className="chip">{f}</span>
          ))}
        </div>
      </div>
    </div>
  )
}
