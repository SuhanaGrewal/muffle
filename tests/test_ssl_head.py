import torch

from muffle.models.ssl_head import AttentivePooling, SSLHeadClassifier


def test_attentive_pooling_output_shape():
    pooling = AttentivePooling(hidden_size=32)
    hidden_states = torch.randn(4, 50, 32)  # batch=4, time=50, hidden=32

    pooled = pooling(hidden_states)

    assert pooled.shape == (4, 32)


def test_attentive_pooling_weights_sum_to_one_per_frame_axis():
    pooling = AttentivePooling(hidden_size=16)
    hidden_states = torch.randn(2, 10, 16)

    weights = torch.softmax(pooling.attn(hidden_states), dim=1)

    assert torch.allclose(weights.sum(dim=1), torch.ones(2, 1), atol=1e-6)
