from neural_network.train import accuracy, train_model


def test_train_and_accuracy_range() -> None:
    model = train_model(epochs=2)
    acc = accuracy(model)
    assert 0.0 <= acc <= 1.0
