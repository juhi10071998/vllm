# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Parity: generic ModelOptLinearMethod == ModelOptNvFp4LinearMethod (W4A4).

The generic QuantKey-driven linear method (``modelopt_experimental.py``) must be
byte-for-byte equivalent to today's ``ModelOptNvFp4LinearMethod`` for an NVFP4
W4A4 layer — same registered params after ``create_weights`` and same values
after ``process_weights_after_loading``. The quirks fail silently (wrong scale
=> garbage output, no error), so this is the fast deterministic gate that backs
the GSM8K shadow run.

Run: ``pytest tests/quantization/test_modelopt_generic_nvfp4_parity.py``.
"""

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(),
    reason="NVFP4 W4A4 kernel requires a CUDA (Blackwell-class) GPU",
)

GROUP_SIZE = 16
IN_FEATURES = 512
OUT_PARTS = [256, 256]  # a fused qkv-style layer, exercises nparts > 1
PARAM_NAMES = (
    "weight",
    "weight_scale",
    "weight_global_scale",
    "input_global_scale",
    "input_global_scale_inv",
    "alpha",
)


def _make_config(algo="NVFP4"):
    from vllm.model_executor.layers.quantization.modelopt import (
        ModelOptNvFp4Config,
    )

    return ModelOptNvFp4Config(
        quant_method=algo,
        is_checkpoint_nvfp4_serialized=True,
        kv_cache_quant_algo=None,
        exclude_modules=[],
        group_size=GROUP_SIZE,
    )


def _run_create_weights(method):
    layer = torch.nn.Module()
    # A real vLLM LinearBase carries params_dtype; the Marlin (W4A16) repack
    # reads it. Bare nn.Module doesn't, so set it as the layer would.
    layer.params_dtype = torch.bfloat16
    method.create_weights(
        layer,
        input_size_per_partition=IN_FEATURES,
        output_partition_sizes=list(OUT_PARTS),
        input_size=IN_FEATURES,
        output_size=sum(OUT_PARTS),
        params_dtype=torch.bfloat16,
        weight_loader=lambda *a, **k: None,
    )
    return layer, method


def _build_old_layer(config):
    from vllm.model_executor.layers.quantization.modelopt import (
        ModelOptNvFp4LinearMethod,
    )

    return _run_create_weights(ModelOptNvFp4LinearMethod(config))


def _build_old_w4a16_layer(config):
    from vllm.model_executor.layers.quantization.modelopt import (
        ModelOptNvFp4W4A16LinearMethod,
    )

    return _run_create_weights(ModelOptNvFp4W4A16LinearMethod(config))


def _build_new_layer(config, algo="NVFP4"):
    from vllm.model_executor.layers.quantization.modelopt_experimental import (
        ModelOptLinearMethod,
        resolve,
    )

    spec, ctx, format_scheme = resolve(algo, config, prefix="model.layers.0.qkv")
    return _run_create_weights(ModelOptLinearMethod(spec, ctx, format_scheme))


def _param_signature(layer):
    return {
        name: (tuple(p.shape), p.dtype)
        for name, p in layer.named_parameters()
    }


def test_create_weights_param_signature_parity(dist_init):
    """After create_weights both layers register the same params (name,
    shape, dtype) — allocation drift would be a silent load bug."""
    config = _make_config()
    old_layer, _ = _build_old_layer(config)
    new_layer, _ = _build_new_layer(config)
    assert _param_signature(old_layer) == _param_signature(new_layer)


def _fill_identical(old_layer, new_layer):
    """Load the same deterministic 'checkpoint' data into both layers, for the
    params they share. (W4A16's old method carries a placeholder input_scale the
    generic path drops — C4 — so it is skipped here and deleted in its process.)
    """
    gen = torch.Generator().manual_seed(0)
    old = dict(old_layer.named_parameters())
    new = dict(new_layer.named_parameters())
    for name in old:
        if name not in new:
            continue
        p = old[name]
        if p.dtype == torch.uint8:
            data = torch.randint(0, 256, p.shape, generator=gen, dtype=torch.uint8)
        elif p.dtype == torch.float8_e4m3fn:
            data = (torch.rand(p.shape, generator=gen) * 0.1 + 0.05).to(
                torch.float8_e4m3fn
            )
        else:
            data = torch.rand(p.shape, generator=gen) * 0.1 + 0.05
        p.data.copy_(data)
        new[name].data.copy_(data)


def test_processed_param_value_parity(dist_init):
    """After process_weights_after_loading (incl. the shared cutlass repack)
    every param matches byte-for-byte."""
    config = _make_config()
    old_layer, old_method = _build_old_layer(config)
    new_layer, new_method = _build_new_layer(config)

    assert _param_signature(old_layer) == _param_signature(new_layer)
    _fill_identical(old_layer, new_layer)

    old_layer, new_layer = old_layer.cuda(), new_layer.cuda()
    old_method.process_weights_after_loading(old_layer)
    new_method.process_weights_after_loading(new_layer)

    old = dict(old_layer.named_parameters())
    new = dict(new_layer.named_parameters())
    assert set(old) == set(new), (set(old), set(new))
    for name in PARAM_NAMES:
        assert name in old, f"expected param {name!r} missing from baseline"
        a, b = old[name], new[name]
        assert a.shape == b.shape and a.dtype == b.dtype, name
        if a.dtype in (torch.uint8, torch.float8_e4m3fn):
            assert torch.equal(a.to(torch.float32), b.to(torch.float32)), name
        else:
            assert torch.allclose(a, b, rtol=0, atol=0), name


# --- W4A16 (weight-only) ---------------------------------------------------
# QuantSpec(kNvfp4Static, None): same fp4 weight scheme, no activation. The old
# ModelOptNvFp4W4A16LinearMethod registers a placeholder input_scale (C4) the
# generic path drops, and stores weight_scale as GroupQuantScaleParameter vs the
# generic's ModelWeightParameter (equivalent per A3 — identical bases). After
# process both reduce to {weight, weight_scale, weight_global_scale}.

W4A16_PARAM_NAMES = ("weight", "weight_scale", "weight_global_scale")


def test_w4a16_create_weights_param_signature_parity(dist_init):
    """create_weights parity minus the intentionally-dropped placeholder
    input_scale (C4)."""
    config = _make_config("W4A16_NVFP4")
    old_layer, _ = _build_old_w4a16_layer(config)
    new_layer, _ = _build_new_layer(config, algo="W4A16_NVFP4")
    old_sig = _param_signature(old_layer)
    new_sig = _param_signature(new_layer)
    assert "input_scale" in old_sig, "expected old W4A16 placeholder input_scale"
    old_sig.pop("input_scale")  # C4: generic drops it
    assert old_sig == new_sig


def test_w4a16_processed_param_value_parity(dist_init):
    """After process (incl. the shared Marlin repack) the weight-only params
    match byte-for-byte, and neither side has input_scale/alpha."""
    config = _make_config("W4A16_NVFP4")
    old_layer, old_method = _build_old_w4a16_layer(config)
    new_layer, new_method = _build_new_layer(config, algo="W4A16_NVFP4")

    _fill_identical(old_layer, new_layer)

    old_layer, new_layer = old_layer.cuda(), new_layer.cuda()
    old_method.process_weights_after_loading(old_layer)
    new_method.process_weights_after_loading(new_layer)

    old = dict(old_layer.named_parameters())
    new = dict(new_layer.named_parameters())
    assert set(old) == set(new), (set(old), set(new))
    assert "input_scale" not in new and "alpha" not in new
    for name in W4A16_PARAM_NAMES:
        assert name in old, f"expected param {name!r} missing from baseline"
        a, b = old[name], new[name]
        assert a.shape == b.shape and a.dtype == b.dtype, name
        if a.dtype in (torch.uint8, torch.float8_e4m3fn):
            assert torch.equal(a.to(torch.float32), b.to(torch.float32)), name
        else:
            assert torch.allclose(a, b, rtol=0, atol=0), name


# --- FP8 per-tensor static (W8A8) ------------------------------------------
# KFp8StaticTensor is bivalent: kFp8StaticTensorSym in BOTH slots. After process
# both reduce to {weight (transposed), weight_scale, input_scale}.

FP8_PARAM_NAMES = ("weight", "weight_scale", "input_scale")


@pytest.fixture
def fp8_vllm_config(monkeypatch):
    """The old ModelOptFp8LinearMethod.__init__ reads
    get_current_vllm_config().model_config.dtype; the bare test config has no
    model_config. Stub it (both modules) so old and new build identically."""
    import types

    from vllm.model_executor.layers.quantization import modelopt as _mo
    from vllm.model_executor.layers.quantization import (
        modelopt_experimental as _me,
    )

    fake = types.SimpleNamespace(
        model_config=types.SimpleNamespace(dtype=torch.bfloat16)
    )
    monkeypatch.setattr(_mo, "get_current_vllm_config", lambda: fake)
    monkeypatch.setattr(_me, "get_current_vllm_config", lambda: fake)
    yield


def _make_fp8_config(algo="FP8"):
    from vllm.model_executor.layers.quantization.modelopt import ModelOptFp8Config

    return ModelOptFp8Config(
        quant_method=algo,
        is_checkpoint_fp8_serialized=True,
        kv_cache_quant_method=None,
        exclude_modules=[],
    )


def _build_old_fp8_layer(config):
    from vllm.model_executor.layers.quantization.modelopt import (
        ModelOptFp8LinearMethod,
    )

    return _run_create_weights(ModelOptFp8LinearMethod(config))


def _build_old_pcpt_layer(config):
    from vllm.model_executor.layers.quantization.modelopt import (
        ModelOptFp8PcPtLinearMethod,
    )

    return _run_create_weights(ModelOptFp8PcPtLinearMethod(config))


def _build_old_pbwo_layer(config):
    from vllm.model_executor.layers.quantization.modelopt import (
        ModelOptFp8PbWoLinearMethod,
    )

    return _run_create_weights(ModelOptFp8PbWoLinearMethod(config))


def test_fp8_create_weights_param_signature_parity(dist_init, fp8_vllm_config):
    config = _make_fp8_config()
    old_layer, _ = _build_old_fp8_layer(config)
    new_layer, _ = _build_new_layer(config, algo="FP8")
    assert _param_signature(old_layer) == _param_signature(new_layer)


def test_fp8_processed_param_value_parity(dist_init, fp8_vllm_config):
    config = _make_fp8_config()
    old_layer, old_method = _build_old_fp8_layer(config)
    new_layer, new_method = _build_new_layer(config, algo="FP8")

    assert _param_signature(old_layer) == _param_signature(new_layer)
    _fill_identical(old_layer, new_layer)

    old_layer, new_layer = old_layer.cuda(), new_layer.cuda()
    old_method.process_weights_after_loading(old_layer)
    new_method.process_weights_after_loading(new_layer)

    old = dict(old_layer.named_parameters())
    new = dict(new_layer.named_parameters())
    assert set(old) == set(new), (set(old), set(new))
    for name in FP8_PARAM_NAMES:
        assert name in old, f"expected param {name!r} missing from baseline"
        a, b = old[name], new[name]
        assert a.shape == b.shape and a.dtype == b.dtype, name
        if a.dtype in (torch.uint8, torch.float8_e4m3fn):
            assert torch.equal(a.to(torch.float32), b.to(torch.float32)), name
        else:
            assert torch.allclose(a, b, rtol=0, atol=0), name


# --- FP8 PcPt (per-channel static weight, dynamic per-token activation) -----
# KFp8StaticChannel (weight) x KDynamicNoParam (activation). No input_scale (the
# activation is dynamically quantized in-kernel). After process: {weight, weight_scale}.

PCPT_PARAM_NAMES = ("weight", "weight_scale")


def test_pcpt_create_weights_param_signature_parity(dist_init, fp8_vllm_config):
    config = _make_fp8_config("FP8_PER_CHANNEL_PER_TOKEN")
    old_layer, _ = _build_old_pcpt_layer(config)
    new_layer, _ = _build_new_layer(config, algo="FP8_PER_CHANNEL_PER_TOKEN")
    assert _param_signature(old_layer) == _param_signature(new_layer)


def test_pcpt_processed_param_value_parity(dist_init, fp8_vllm_config):
    config = _make_fp8_config("FP8_PER_CHANNEL_PER_TOKEN")
    old_layer, old_method = _build_old_pcpt_layer(config)
    new_layer, new_method = _build_new_layer(
        config, algo="FP8_PER_CHANNEL_PER_TOKEN"
    )

    assert _param_signature(old_layer) == _param_signature(new_layer)
    _fill_identical(old_layer, new_layer)

    old_layer, new_layer = old_layer.cuda(), new_layer.cuda()
    old_method.process_weights_after_loading(old_layer)
    new_method.process_weights_after_loading(new_layer)

    old = dict(old_layer.named_parameters())
    new = dict(new_layer.named_parameters())
    assert set(old) == set(new), (set(old), set(new))
    assert "input_scale" not in new
    for name in PCPT_PARAM_NAMES:
        assert name in old, f"expected param {name!r} missing from baseline"
        a, b = old[name], new[name]
        assert a.shape == b.shape and a.dtype == b.dtype, name
        if a.dtype in (torch.uint8, torch.float8_e4m3fn):
            assert torch.equal(a.to(torch.float32), b.to(torch.float32)), name
        else:
            assert torch.allclose(a, b, rtol=0, atol=0), name


# --- FP8 PbWo (128x128 block-static weight, dynamic per-block act) -----------
# KFp8Block128 x KDynamicNoParam. C12: the OLD method never runs the block
# kernel's post-load (guard checks self.fp8_linear but the kernel is stored as
# self.w8a8_block_fp8_linear). The generic base DOES run it. So we can't compare
# against old-as-is; we apply the skipped kernel step to the old layer too and
# assert generic == (old squeeze + block kernel post-load) — i.e. CT-aligned.

PBWO_PARAM_NAMES = ("weight", "weight_scale")


def test_pbwo_create_weights_param_signature_parity(dist_init, fp8_vllm_config):
    config = _make_fp8_config("FP8_PB_WO")
    old_layer, _ = _build_old_pbwo_layer(config)
    new_layer, _ = _build_new_layer(config, algo="FP8_PB_WO")
    assert _param_signature(old_layer) == _param_signature(new_layer)


def test_pbwo_processed_param_value_parity(dist_init, fp8_vllm_config):
    config = _make_fp8_config("FP8_PB_WO")
    old_layer, old_method = _build_old_pbwo_layer(config)
    new_layer, new_method = _build_new_layer(config, algo="FP8_PB_WO")

    assert _param_signature(old_layer) == _param_signature(new_layer)
    _fill_identical(old_layer, new_layer)

    old_layer, new_layer = old_layer.cuda(), new_layer.cuda()
    old_method.process_weights_after_loading(old_layer)  # squeeze; kernel SKIPPED
    # C12: apply the block kernel post-load the old method's misnamed guard skips.
    old_method.w8a8_block_fp8_linear.process_weights_after_loading(old_layer)
    new_method.process_weights_after_loading(new_layer)  # squeeze + kernel

    old = dict(old_layer.named_parameters())
    new = dict(new_layer.named_parameters())
    assert set(old) == set(new), (set(old), set(new))
    for name in PBWO_PARAM_NAMES:
        assert name in old, f"expected param {name!r} missing from baseline"
        a, b = old[name], new[name]
        assert a.shape == b.shape and a.dtype == b.dtype, name
        if a.dtype in (torch.uint8, torch.float8_e4m3fn):
            assert torch.equal(a.to(torch.float32), b.to(torch.float32)), name
        else:
            assert torch.allclose(a, b, rtol=0, atol=0), name


# --- MXFP8 (block-32 e4m3 weight + e8m0 scale, dynamic activation) -----------
# KMxfp8Static x KDynamicNoParam. Both old and new run the same
# init_mxfp8_linear_kernel post-load unconditionally -> byte-identical.

MXFP8_PARAM_NAMES = ("weight", "weight_scale")


def _make_mxfp8_config():
    from vllm.model_executor.layers.quantization.modelopt import (
        ModelOptMxFp8Config,
    )

    return ModelOptMxFp8Config(
        is_checkpoint_mxfp8_serialized=True,
        kv_cache_quant_algo=None,
        exclude_modules=[],
    )


def _build_old_mxfp8_layer(config):
    from vllm.model_executor.layers.quantization.modelopt import (
        ModelOptMxFp8LinearMethod,
    )

    return _run_create_weights(ModelOptMxFp8LinearMethod(config))


def test_mxfp8_create_weights_param_signature_parity(dist_init):
    config = _make_mxfp8_config()
    old_layer, _ = _build_old_mxfp8_layer(config)
    new_layer, _ = _build_new_layer(config, algo="MXFP8")
    assert _param_signature(old_layer) == _param_signature(new_layer)


def test_mxfp8_processed_param_value_parity(dist_init):
    config = _make_mxfp8_config()
    old_layer, old_method = _build_old_mxfp8_layer(config)
    new_layer, new_method = _build_new_layer(config, algo="MXFP8")

    assert _param_signature(old_layer) == _param_signature(new_layer)
    _fill_identical(old_layer, new_layer)

    old_layer, new_layer = old_layer.cuda(), new_layer.cuda()
    old_method.process_weights_after_loading(old_layer)
    new_method.process_weights_after_loading(new_layer)

    old = dict(old_layer.named_parameters())
    new = dict(new_layer.named_parameters())
    assert set(old) == set(new), (set(old), set(new))
    for name in MXFP8_PARAM_NAMES:
        assert name in old, f"expected param {name!r} missing from baseline"
        a, b = old[name], new[name]
        assert a.shape == b.shape and a.dtype == b.dtype, name
        if a.dtype in (torch.uint8, torch.float8_e4m3fn):
            assert torch.equal(a.to(torch.float32), b.to(torch.float32)), name
        else:
            assert torch.allclose(a, b, rtol=0, atol=0), name


# --- Runtime parity: kernel class + input_quant_key exposure (all formats) ---
# The param-parity tests above prove the *registered params* are byte-identical,
# but they cannot see the RUNTIME surface: which concrete kernel is selected and
# whether the layer advertises input_quant_key (activation-quant fusion). FP8
# per-tensor diverged there (C2 — generic exposed the key, old did not) despite
# byte-identical params. This test guards that surface for every format.


def _old_kernel(method):
    for attr in ("fp8_linear", "w8a8_block_fp8_linear", "kernel"):
        k = getattr(method, attr, None)
        if k is not None:
            return k
    raise AssertionError(f"no kernel attr on {type(method).__name__}")


def _runtime_pair(name):
    if name == "nvfp4_w4a4":
        c = _make_config("NVFP4")
        return _build_old_layer(c), _build_new_layer(c, "NVFP4")
    if name == "w4a16":
        c = _make_config("W4A16_NVFP4")
        return _build_old_w4a16_layer(c), _build_new_layer(c, "W4A16_NVFP4")
    if name == "fp8":
        c = _make_fp8_config("FP8")
        return _build_old_fp8_layer(c), _build_new_layer(c, "FP8")
    if name == "pcpt":
        c = _make_fp8_config("FP8_PER_CHANNEL_PER_TOKEN")
        return (
            _build_old_pcpt_layer(c),
            _build_new_layer(c, "FP8_PER_CHANNEL_PER_TOKEN"),
        )
    if name == "pbwo":
        c = _make_fp8_config("FP8_PB_WO")
        return _build_old_pbwo_layer(c), _build_new_layer(c, "FP8_PB_WO")
    if name == "mxfp8":
        c = _make_mxfp8_config()
        return _build_old_mxfp8_layer(c), _build_new_layer(c, "MXFP8")
    raise ValueError(name)


@pytest.mark.parametrize(
    "name", ["nvfp4_w4a4", "w4a16", "fp8", "pcpt", "pbwo", "mxfp8"]
)
def test_runtime_kernel_and_expose_parity(dist_init, fp8_vllm_config, name):
    (old_layer, old_method), (new_layer, new_method) = _runtime_pair(name)

    # Same concrete kernel class (invisible to param-parity; a mismatch shifts
    # numerics and can add run-to-run variance).
    old_k, new_k = _old_kernel(old_method), new_method.kernel
    assert type(old_k) is type(new_k), (
        name, type(old_k).__name__, type(new_k).__name__
    )

    # Same input_quant_key exposure — the C2 surface that bit FP8 per-tensor.
    old_exp = hasattr(old_layer, "input_quant_key")
    new_exp = hasattr(new_layer, "input_quant_key")
    assert old_exp == new_exp, (name, "old_exposes", old_exp, "new_exposes", new_exp)
    if old_exp:
        assert old_layer.input_quant_key == new_layer.input_quant_key, name
