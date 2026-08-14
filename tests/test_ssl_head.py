import torch

from muffle.models.ssl_head import AttentivePooling, SSLHeadClassifier


def test_attentive_pooling_output_shape():
    pooling = AttentivePooling(hidden_size=32)
    hidden_states = torch.randn(4, 50, 32)  # batch=4, time=50, hidden=32

    pooled = pooling(hidden_states)

    assert pooled.shape == (4, 32)
