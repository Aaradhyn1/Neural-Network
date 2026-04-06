import numpy as np

from neural_network.preprocessing import impute_missing_with_mean, minmax_normalize, train_val_test_split


def test_impute_and_normalize_pipeline() -> None:
    x = np.array([[1.0, np.nan], [3.0, 5.0], [2.0, 7.0]], dtype=np.float64)
    x_imputed = impute_missing_with_mean(x)
    assert not np.isnan(x_imputed).any()

    x_norm = minmax_normalize(x_imputed)
    assert np.all(x_norm >= 0.0)
    assert np.all(x_norm <= 1.0)


def test_train_val_test_split_sizes() -> None:
    x = np.random.default_rng(0).normal(size=(100, 4))
    y = np.random.default_rng(1).integers(0, 2, size=(100, 1)).astype(np.float64)
    (x_train, _), (x_val, _), (x_test, _) = train_val_test_split(x, y, train_ratio=0.7, val_ratio=0.15)
    assert x_train.shape[0] == 70
    assert x_val.shape[0] == 15
    assert x_test.shape[0] == 15
