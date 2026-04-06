from neural_network.model_library import get_model_spec
from neural_network.train import accuracy, train_model


def test_model_library_wide_preset_exists() -> None:
    spec = get_model_spec("llm_wide")
    assert spec.hidden_sizes[0] >= 128


def test_train_reaches_high_parity_accuracy() -> None:
    model = train_model(epochs=500, model_preset="parity", lr=0.01, optimizer="adam", log_every=250)
    acc = accuracy(model, bits=4)
    assert acc >= 0.95
