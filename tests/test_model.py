"""Tests for multihop.model on a real (small) model.

Model is chosen by env var MULTIHOP_TEST_MODEL (default Qwen/Qwen2.5-0.5B-Instruct), e.g.
    MULTIHOP_TEST_MODEL=allenai/Olmo-3-7B-Instruct .venv/bin/python -m pytest tests/test_model.py -s
"""

import os

import pytest
import torch

from multihop.model import (
    ModelConfig,
    ablate_direction,
    hidden_size,
    add_direction,
    get_layers,
    greedy_generate,
    format_prompts,
    layer_at_fraction,
    layers_in_band,
    load_model,
    record,
    tokenize,
)

MODEL_ID = os.environ.get("MULTIHOP_TEST_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
K = 8
# bf16 tolerance for "same computation, different batch shape" comparisons (max |diff| / max |ref|).
# Measured on Qwen2.5-0.5B: padded-vs-single differs by up to 0.020 in bf16, which is the same size as
# the bf16-vs-float32 error of a single prompt (up to 0.021); in float32 the padding difference is ~3e-6
# (see test_left_padding_matches_single_fp32, the real correctness check).
BF16_REL_TOL = 5e-2
QUESTIONS = [
    "What is the first letter of the capital of the country where the Eiffel Tower is located? "
    "Answer immediately with only the letter.",
    "Name a fruit.",
    "What is the first letter of the name of the author of the novel whose main character is a whale-hunting "
    "captain named Ahab? Answer immediately with only the letter.",
]


@pytest.fixture(scope="module")
def mt():
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    model, tok = load_model(ModelConfig(model_id=MODEL_ID))
    return model, tok


@pytest.fixture(scope="module")
def prompts(mt):
    return format_prompts(mt[1], QUESTIONS)


def rel_err(a, b):
    """max |a - b| / max |b|, in float32."""
    a, b = a.float(), b.float()
    return ((a - b).abs().max() / b.abs().max()).item()


# ----------------------------------------------------------------------------- pure helpers


def test_fraction_to_layer():
    assert layer_at_fraction(0.0, 32) == 0
    assert layer_at_fraction(0.5, 32) == 16
    assert layer_at_fraction(1.0, 32) == 31  # clipped to last layer
    assert layer_at_fraction(0.25, 24) == 6
    assert layers_in_band([0.25, 0.75], 32) == list(range(8, 24))
    assert layers_in_band([0.0, 1.0], 32) == list(range(32))
    assert layers_in_band([0.25, 0.75], 80) == list(range(20, 60))
    assert layers_in_band([0.5, 0.51], 32) == [16]  # never empty
    with pytest.raises(ValueError):
        layers_in_band([0.75, 0.25], 32)
    with pytest.raises(ValueError):
        layer_at_fraction(1.5, 32)


# ----------------------------------------------------------------------------- recording


def test_record_matches_hidden_states(mt, prompts):
    model, tok = mt
    n = len(get_layers(model))
    rec = record(model, tok, prompts, k=K)
    assert rec.acts.shape == (len(prompts), n, K, hidden_size(model))
    assert rec.acts.device.type == "cpu"

    # Reference: unpadded single-prompt forward passes (record batches by equal length, without padding;
    # padded batches are not exact for every architecture, e.g. linear attention in Qwen3.5/3.8).
    inner = getattr(model.model, "language_model", model.model)
    for j, p in enumerate(prompts):
        with torch.no_grad():
            hs = model(**tokenize(tok, [p], model.device), output_hidden_states=True, use_cache=False).hidden_states
        # hidden_states[i + 1] is the output of layer i, except the last entry, which HF returns
        # after the final norm; so compare all but the last layer exactly.
        for i in range(n - 1):
            assert torch.equal(rec.acts[j, i], hs[i + 1][0, -K:].cpu()), f"prompt {j} layer {i}"
        with torch.no_grad():
            normed = inner.norm(rec.acts[j, n - 1].to(model.device)).cpu()
        assert torch.equal(normed, hs[n][0, -K:].cpu())

    # Read tokens are the last K real tokens of each prompt.
    for row, p in zip(rec.read_tokens, prompts):
        ids = tok(p, add_special_tokens=False)["input_ids"][-K:]
        assert row == tok.convert_ids_to_tokens(ids)


def test_layer_subset(mt, prompts):
    model, tok = mt
    full = record(model, tok, prompts, k=K)
    sub = record(model, tok, prompts, k=K, layers=[0, 5, 3])
    assert sub.layers == [0, 5, 3]
    assert torch.equal(sub.acts, full.acts[:, [0, 5, 3]])


def test_greedy_generate_matches_record_argmax(mt, prompts):
    model, tok = mt
    rec = record(model, tok, prompts, k=K)
    gens = greedy_generate(model, tok, prompts, max_new_tokens=1)
    assert gens == [tok.decode([i], skip_special_tokens=True) for i in rec.next_token_ids.tolist()]


def test_left_padding_matches_single(mt, prompts):
    model, tok = mt
    lengths = [len(tok(p, add_special_tokens=False)["input_ids"]) for p in prompts]
    assert len(set(lengths)) > 1, "test needs prompts of different lengths"
    batched = record(model, tok, prompts, k=K)
    worst = 0.0
    for j, p in enumerate(prompts):
        single = record(model, tok, [p], k=K)
        err = rel_err(batched.acts[j], single.acts[0])
        abs_err = (batched.acts[j].float() - single.acts[0].float()).abs().max().item()
        print(f"\n[padding] prompt {j} (len {lengths[j]}): max abs diff {abs_err:.4g}, rel {err:.3g}")
        worst = max(worst, err)
        assert batched.next_token_ids[j] == single.next_token_ids[0]
    assert worst < BF16_REL_TOL


def test_left_padding_matches_single_fp32(mt, prompts):
    """In float32, batching with left padding must match single-prompt runs almost exactly."""
    if sum(p.numel() for p in mt[0].parameters()) > 2e9:  # a second, fp32 copy would not fit
        pytest.skip("fp32 check only for small models")
    model, tok = load_model(ModelConfig(model_id=MODEL_ID, dtype="float32"))
    batched = record(model, tok, prompts, k=K)
    for j, p in enumerate(prompts):
        single = record(model, tok, [p], k=K)
        err = rel_err(batched.acts[j], single.acts[0])
        print(f"\n[padding fp32] prompt {j}: rel {err:.3g}")
        assert err < 1e-4
    del model
    torch.cuda.empty_cache()


# ----------------------------------------------------------------------------- interventions


def test_ablation_projects_out_direction(mt, prompts):
    model, tok = mt
    n = len(get_layers(model))
    g = torch.Generator().manual_seed(0)
    d = torch.randn(hidden_size(model), generator=g)
    d_unit = d / d.norm()
    band = layers_in_band([0.25, 0.75], n)

    p = prompts[0]
    seq_len = len(tok(p, add_special_tokens=False)["input_ids"])
    clean = record(model, tok, [p], k=seq_len)  # record every position
    with ablate_direction(model, d, band, k=K):
        abl = record(model, tok, [p], k=seq_len)

    proj = abl.acts[0, :, -K:].float() @ d_unit  # [layers, K]
    norms = abl.acts[0, :, -K:].float().norm(dim=-1)
    rel_proj = (proj.abs() / norms)[band]
    print(f"\n[ablation] max |proj|/|h| on ablated layers/positions: {rel_proj.max():.3g}")
    assert rel_proj.max() < 1e-2  # zero up to bf16 rounding

    # Positions before the last K are untouched at every layer (causal attention).
    assert torch.equal(abl.acts[0, :, :-K], clean.acts[0, :, :-K])
    # Layers before the band are untouched everywhere.
    assert torch.equal(abl.acts[0, : band[0]], clean.acts[0, : band[0]])
    # And the ablation did change something.
    assert not torch.equal(abl.acts[0, band[0], -K:], clean.acts[0, band[0], -K:])
    # The modified output propagates to later layers (hook return value is used downstream).
    assert not torch.equal(abl.acts[0, -1, -K:], clean.acts[0, -1, -K:])


def test_ablation_per_layer_dict(mt, prompts):
    model, tok = mt
    g = torch.Generator().manual_seed(1)
    layers = [2, 4]
    dirs = {l: torch.randn(hidden_size(model), generator=g) for l in layers}
    with ablate_direction(model, dirs, layers, k=K):
        rec = record(model, tok, prompts, k=K, layers=layers)
    for j, l in enumerate(layers):
        u = dirs[l] / dirs[l].norm()
        rel = (rec.acts[:, j].float() @ u).abs() / rec.acts[:, j].float().norm(dim=-1)
        assert rel.max() < 1e-2


def test_ablation_with_left_padding(mt, prompts):
    model, tok = mt
    n = len(get_layers(model))
    d = torch.randn(hidden_size(model), generator=torch.Generator().manual_seed(2))
    band = layers_in_band([0.25, 0.75], n)
    with ablate_direction(model, d, band, k=K):
        batched = record(model, tok, prompts, k=K)
        singles = [record(model, tok, [p], k=K) for p in prompts]
    u = d / d.norm()
    for j, s in enumerate(singles):
        assert rel_err(batched.acts[j], s.acts[0]) < BF16_REL_TOL
        rel = (batched.acts[j][band].float() @ u).abs() / batched.acts[j][band].float().norm(dim=-1)
        assert rel.max() < 1e-2


def test_steering_adds_vector(mt, prompts):
    model, tok = mt
    layer = 3
    d = torch.randn(hidden_size(model), generator=torch.Generator().manual_seed(3))
    clean = record(model, tok, prompts, k=K, layers=[layer])
    with add_direction(model, d, layer, scale=4.0):
        steered = record(model, tok, prompts, k=K, layers=[layer, layer + 1])
    diff = steered.acts[:, 0].float() - clean.acts[:, 0].float()
    expected = (4.0 * d).expand_as(diff)
    assert rel_err(diff, expected) < 2e-2
    # The steered output is what the next layer sees.
    clean_next = record(model, tok, prompts, k=K, layers=[layer + 1])
    assert not torch.equal(steered.acts[:, 1], clean_next.acts[:, 0])


def _n_hooks(model):
    return sum(len(l._forward_hooks) for l in get_layers(model))


def test_hooks_removed_even_on_exception(mt, prompts):
    model, tok = mt
    d = torch.randn(hidden_size(model))
    # transformers 5 permanently installs its own output-capturing hook per layer the first time
    # output_hidden_states=True is used, so count relative to what is present now.
    base = _n_hooks(model)
    with ablate_direction(model, d, [0, 1], k=K):
        assert _n_hooks(model) == base + 2
    assert _n_hooks(model) == base
    with pytest.raises(RuntimeError):
        with ablate_direction(model, d, [0, 1], k=K), add_direction(model, d, 2, 1.0):
            assert _n_hooks(model) == base + 3
            raise RuntimeError("boom")
    assert _n_hooks(model) == base
    with pytest.raises(ValueError):  # record() fails (k too large) -> its hooks must not leak
        record(model, tok, prompts, k=10_000)
    assert _n_hooks(model) == base
    # A failure inside the forward pass of record() must also clean up.
    with pytest.raises(RuntimeError):
        with add_direction(model, torch.randn(7), 0, 1.0):  # wrong size -> error inside forward
            record(model, tok, prompts, k=K)
    assert _n_hooks(model) == base
