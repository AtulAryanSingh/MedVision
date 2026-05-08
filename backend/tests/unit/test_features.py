import numpy as np

from features.extractor import extract_features


def test_extract_features_returns_expected_shape():
    arr = np.zeros((10, 10), dtype=np.float32)
    arr[2:8, 2:8] = 50.0

    feat = extract_features(arr)

    assert "mean" in feat
    assert "std_dev" in feat
    assert "histogram" in feat
    assert "shape_descriptors" in feat
    assert len(feat["histogram"]["bins"]) == 64
    assert len(feat["histogram"]["counts"]) == 64
    assert feat["shape_descriptors"]["total_pixels"] == 100
