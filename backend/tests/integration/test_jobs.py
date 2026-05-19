import time

import numpy as np


def _upload_png(client) -> str:
    import cv2

    arr = (np.random.default_rng(11).random((64, 64)) * 255).astype(np.uint8)
    ok, buf = cv2.imencode('.png', arr)
    assert ok
    resp = client.post('/api/upload', files={'file': ('job.png', buf.tobytes(), 'image/png')})
    assert resp.status_code == 200, resp.text
    return resp.json()['image_id']


def _upload_nifti(client, tmp_path) -> str:
    import nibabel as nib

    vol = (np.random.default_rng(12).random((24, 48, 48)) * 2000).astype(np.float32)
    # nibabel expects (X, Y, Z)
    vol_nib = np.transpose(vol, (2, 1, 0))
    affine = np.diag([1.2, 1.2, 2.5, 1.0])
    img = nib.Nifti1Image(vol_nib, affine)
    path = tmp_path / 'job_volume.nii'
    nib.save(img, path)

    with open(path, 'rb') as fh:
        raw = fh.read()
    resp = client.post('/api/upload', files={'file': ('job_volume.nii', raw, 'application/octet-stream')})
    assert resp.status_code == 200, resp.text
    return resp.json()['image_id']


def _wait_for_terminal(client, job_id: str, timeout_s: float = 15.0):
    started = time.time()
    last = None
    while time.time() - started < timeout_s:
        resp = client.get(f'/api/jobs/{job_id}')
        assert resp.status_code == 200, resp.text
        last = resp.json()['data']
        if last['status'] in {'succeeded', 'failed', 'canceled'}:
            return last
        time.sleep(0.1)
    raise AssertionError(f'Job {job_id} did not finish in time, last={last}')


def test_patchify_job_success_and_stream(client, tmp_path):
    image_id = _upload_nifti(client, tmp_path)

    create = client.post('/api/jobs/patchify', json={'image_id': image_id, 'patch_size': 16, 'stride': 8})
    assert create.status_code == 202, create.text
    job = create.json()['data']

    done = _wait_for_terminal(client, job['job_id'])
    assert done['status'] == 'succeeded', done

    result = client.get(f"/api/jobs/{job['job_id']}/result")
    assert result.status_code == 200, result.text
    payload = result.json()['data']
    assert payload['n_patches'] > 0

    streamed = client.get(f"/api/jobs/{job['job_id']}/result/stream")
    assert streamed.status_code == 200, streamed.text
    assert streamed.headers['content-type'].startswith('application/octet-stream')
    assert len(streamed.content) > 0


def test_patchify_job_failure_on_2d_input(client):
    image_id = _upload_png(client)

    create = client.post('/api/jobs/patchify', json={'image_id': image_id, 'patch_size': 16, 'stride': 8})
    assert create.status_code == 202, create.text
    job = create.json()['data']

    done = _wait_for_terminal(client, job['job_id'])
    assert done['status'] == 'failed'
    assert '3-D volume' in done['error']['detail']


def test_patchify_job_cancel_request(client, tmp_path):
    image_id = _upload_nifti(client, tmp_path)

    create = client.post('/api/jobs/patchify', json={'image_id': image_id, 'patch_size': 8, 'stride': 2})
    assert create.status_code == 202, create.text
    job_id = create.json()['data']['job_id']

    cancel = client.post(f'/api/jobs/{job_id}/cancel')
    assert cancel.status_code == 200, cancel.text

    status = client.get(f'/api/jobs/{job_id}')
    assert status.status_code == 200
    state = status.json()['data']
    assert state['cancel_requested'] is True or state['status'] in {'succeeded', 'canceled'}


def test_export_npy_stream(client):
    image_id = _upload_png(client)
    resp = client.get(f'/api/export/{image_id}/npy/stream')
    assert resp.status_code == 200, resp.text
    assert resp.headers['content-type'].startswith('application/octet-stream')
    assert len(resp.content) > 0
