import pytest
import torch
from loom_policy import LoomPolicy, ModelConfig


def model(**kwargs):
    return LoomPolicy(ModelConfig(d_model=16, heads=2, layers=1, patch_size=8, **kwargs)).eval()


def test_vit_preserves_rectangular_grid_positions_and_mixes_patches():
    torch.manual_seed(3)
    encoder = model().image_encoder
    inputs = []
    hook = encoder.transformer.register_forward_pre_hook(lambda module, args: inputs.append(args[0]))
    image = torch.full((3, 16, 24), .4)
    with torch.no_grad():
        tokens, xy = encoder(image)
        changed = image.clone()
        changed[:, 8:, 16:] = .9
        other, _ = encoder(changed)
    hook.remove()
    assert tokens.shape == (6, 16) and xy.shape == (6, 2)
    torch.testing.assert_close(xy, torch.tensor([[1/6, .25], [.5, .25], [5/6, .25],
                                                [1/6, .75], [.5, .75], [5/6, .75]]))
    # Identical input patches get different locations before attention.
    assert not torch.allclose(inputs[0][:, 0], inputs[0][:, 1])
    assert not torch.allclose(inputs[0][:, 0], inputs[0][:, 3])
    # The first patch's pixels are unchanged, but its encoded context changes.
    assert not torch.allclose(tokens[0], other[0])


def test_bert_preserves_utf8_tokens_and_reads_right_context():
    torch.manual_seed(5)
    encoder = model().text_encoder
    with torch.no_grad():
        assert encoder("ab方").shape == (len("ab方".encode("utf-8")), 16)
        first = encoder("abc")
        second = encoder("abd")
        repeated = encoder("aa")
    # A later byte changes the first token: attention is bidirectional.
    assert not torch.allclose(first[0], second[0])
    assert not torch.allclose(repeated[0], repeated[1])


def test_limits_fail_before_modality_attention_without_silent_truncation():
    m = model(max_text_tokens=5)
    calls = []
    hooks = [module.register_forward_pre_hook(lambda *args: calls.append(1))
             for module in (m.image_encoder.transformer, m.text_encoder.transformer)]
    with pytest.raises(ValueError, match="max_text_tokens"):
        m.text_encoder("abc方")  # Six UTF-8 bytes; the limit is not characters.
    with pytest.raises(ValueError, match="max_tokens"):
        m.text_encoder("ab方", token_budget=4)
    with pytest.raises(ValueError, match="max_tokens"):
        m.image_encoder(torch.zeros(3, 16, 24), token_budget=5)
    with pytest.raises(ValueError, match="image must"):
        m.image_encoder(torch.empty(3, 0, 16))
    assert not calls
    for hook in hooks:
        hook.remove()
    assert m.text_encoder("ab方").shape == (5, 16)


def test_random_initialization_is_reproducible_and_layer_weights_are_independent():
    torch.manual_seed(7)
    a = model()
    torch.manual_seed(7)
    b = model()
    for key, value in a.state_dict().items():
        torch.testing.assert_close(value, b.state_dict()[key], rtol=0, atol=0)
    torch.manual_seed(8)
    c = model()
    assert not torch.equal(a.image_encoder.patch.weight, c.image_encoder.patch.weight)
    assert not torch.equal(a.text_encoder.embedding.weight, c.text_encoder.embedding.weight)
    assert all(p.requires_grad for p in a.parameters())
    for encoder in (a.image_encoder, a.text_encoder):
        assert not torch.equal(encoder.transformer.layers[0].self_attn.in_proj_weight,
                               encoder.transformer.layers[1].self_attn.in_proj_weight)
    for settings in ({"vision_layers": 0}, {"text_layers": 0}, {"max_text_tokens": 0}):
        with pytest.raises(ValueError, match="positive integers"):
            ModelConfig(**settings)
