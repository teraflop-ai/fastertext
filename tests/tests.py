from fastertext import load_model


def test_batch_predictions():
    model = load_model("model_v3.bin")
    labels, probs = model.batch(["hello world", "bonjour le monde"], k=2)
    id2label = model.get_labels()

    assert labels.shape == probs.shape == (2, 2)
    assert ((probs >= 0) & (probs <= 1)).all()
    assert (probs[:, 0] >= probs[:, 1]).all()
    assert all(
        isinstance(id2label[int(label)], str)
        for row in labels
        for label in row
    )