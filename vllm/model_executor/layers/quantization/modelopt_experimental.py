# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experimental QuantKey-driven ModelOpt linear method (flag-gated).

Opt-in via ``VLLM_MODELOPT_GENERIC``. Routes ModelOpt **linear** layers through a
single generic ``ModelOptLinearMethod`` that composes per-``QuantKey`` schemes,
instead of the per-format method classes in ``modelopt.py``. MoE and the
front-end configs are **imported unchanged** from ``modelopt.py`` (this preserves
the ``ModelOpt*`` class-name coupling that ``routed_experts.py`` gates on).

This first slice implements **NVFP4 W4A4 linear only**; every other role/scheme
rejects loudly. See ``linear_design_concrete.md`` for the full design.
"""

from dataclasses import dataclass
from enum import Enum

import torch
from torch.nn.parameter import Parameter

import vllm.envs as envs
from vllm.config import get_current_vllm_config
from vllm.config.quantization import QuantSpec
from vllm.logger import init_logger
from vllm.model_executor.kernels.linear import (
    MarlinNvFp4LinearKernel,
    NvFp4LinearLayerConfig,
    init_fp8_linear_kernel,
    init_mxfp8_linear_kernel,
    init_nvfp4_linear_kernel,
)
from vllm.model_executor.layers.attention import Attention, MLAAttention
from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.fusion.quant_activation import (
    expose_input_quant_key,
)
from vllm.model_executor.layers.linear import (
    LinearBase,
    LinearMethodBase,
    UnquantizedLinearMethod,
    register_weight_loader_v2_supported_method,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    get_marlin_input_dtype,
)
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    process_fp8_weight_channel_strategy,
)
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    MXFP8_BLOCK_SIZE,
    MXFP8_SCALE_DTYPE,
    MXFP8_VALUE_DTYPE,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    FP4_DTYPE,
    QuantKey,
    kFp8Dynamic128Sym,
    kFp8DynamicTokenSym,
    kFp8Static128BlockSym,
    kFp8StaticTensorSym,
    kFp8StaticTokenSym,
    kMxfp8Dynamic,
    kMxfp8Static,
    kNvfp4Dynamic,
    kNvfp4Static,
)
from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
    requantize_with_max_scale,
)
from vllm.model_executor.parameter import (
    BlockQuantScaleParameter,
    ChannelQuantScaleParameter,
    ModelWeightParameter,
    PerTensorScaleParameter,
)
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead

# Imported unchanged from modelopt.py — same class objects, so the ModelOpt*
# name coupling (routed_experts.py:703,744) and sub-config identity are intact.
from vllm.model_executor.layers.quantization.modelopt import (
    ModelOptFp8Config,
    ModelOptMxFp8Config,
    ModelOptNvFp4Config,
)

logger = init_logger(__name__)


class Role(Enum):
    WEIGHT = "weight"
    ACT = "activation"


WEIGHT = Role.WEIGHT
ACT = Role.ACT

# Weight-loader "unloaded shard" marker — FP8 family fills scales with it; the
# NVFP4/MXFP8 families deliberately do not (C3, load-bearing asymmetry).
SENTINEL = torch.finfo(torch.float32).min


# ---------------------------------------------------------------------------
# 1. Helpers (linear_design_concrete.md §1)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CkptCtx:
    """Per-checkpoint facts a QuantKey cannot carry."""

    serialized: bool
    group_size: int | None = None


@dataclass(frozen=True)
class RuntimeDtypes:
    """Runtime/model dtypes kernels need — format-agnostic."""

    input_dtype: torch.dtype
    out_dtype: torch.dtype
    marlin_input_dtype: torch.dtype | None = None


@dataclass(frozen=True)
class Shapes:
    """Layer geometry from create_weights args."""

    out_parts: list[int]
    in_: int
    params_dtype: torch.dtype

    @property
    def out(self) -> int:
        return sum(self.out_parts)

    @property
    def nparts(self) -> int:
        return len(self.out_parts)


# ---------------------------------------------------------------------------
# 3. QuantKey schemes (linear_design_concrete.md §3-4)
# ---------------------------------------------------------------------------


class QuantKeyScheme:
    """One scheme per QuantKey. Selected by key *content*; the base supplies the
    ``role`` from the slot it fills the key into (wkey->WEIGHT, akey->ACT).

    Every scheme branches on ``role`` explicitly and ``reject``s any role it has
    not validated — never falling through to the wrong role's registration
    (silent-garbage trap, C13).
    """

    key: QuantKey
    requires_serialized: bool = True
    # Whether the base advertises the kernel's input_quant_key on the layer
    # (enables upstream activation-quant fusion). Behavior-preserving per-format:
    # the old NVFP4 W4A4 method exposed it (modelopt.py:1205), the old FP8 method
    # did NOT — and the FP8 kernel *does* return a static key, so exposing there
    # flips activation quant into a fused path and diverges (C2). Read off the
    # weight scheme; default True (matches NVFP4 + the mgoin direction), False on
    # the FP8 scheme to preserve today's behavior. Adopting FP8 fusion is a
    # separate deliberate change, not part of this behavior-preserving migration.
    exposes_input_quant_key: bool = True

    def create_weights(self, layer, role, ctx, shapes, wl) -> None:
        raise NotImplementedError

    def process(self, layer, role) -> None:
        pass

    @staticmethod
    def reject(role) -> None:
        raise NotImplementedError(
            f"role {role!r} not validated for this QuantKey scheme"
        )

    @staticmethod
    def register_params(layer, name, shape, dtype, cls, wl, *, init=None, **dims):
        p = cls(data=torch.empty(shape, dtype=dtype), weight_loader=wl, **dims)
        if init is not None:
            p.data.fill_(init)
        layer.register_parameter(name, p)


class KNvfp4Static(QuantKeyScheme):
    """NVFP4 weight scheme (W4A4 and W4A16 share it). Weight-role only today."""

    key = kNvfp4Static

    def create_weights(self, layer, role, ctx, shapes, wl) -> None:
        if role is not WEIGHT:
            self.reject(role)
        if shapes.in_ % 16 != 0:
            raise ValueError(
                "Unsupported model when in features size is not multiple of 16"
            )
        weight_dtype = (
            torch.float8_e4m3fn if ctx.serialized else shapes.params_dtype
        )
        # Packed NVFP4 weight: 2 fp4 items per byte along the input dim.
        self.register_params(
            layer,
            "weight",
            (shapes.out, shapes.in_ // 2),
            torch.uint8,
            ModelWeightParameter,
            wl,
            input_dim=1,
            output_dim=0,
        )
        # Per-tensor global weight scale.
        self.register_params(
            layer,
            "weight_scale_2",
            (shapes.nparts,),
            torch.float32,
            PerTensorScaleParameter,
            wl,
        )
        # Per-block (group_size) weight scale.
        self.register_params(
            layer,
            "weight_scale",
            (shapes.out, shapes.in_ // ctx.group_size),
            weight_dtype,
            ModelWeightParameter,
            wl,
            input_dim=1,
            output_dim=0,
        )

    def process(self, layer, role) -> None:
        if role is not WEIGHT:
            self.reject(role)
        if torch.unique(layer.weight_scale_2).numel() != 1:
            logger.warning_once(
                "In NVFP4 linear, the global weight scale differs across "
                "parallel layers (e.g. q_proj, k_proj, v_proj). This will "
                "likely reduce accuracy. Consider a checkpoint with a shared "
                "global NVFP4 scale for parallel layers."
            )
        # Raw max, no reciprocation — Marlin/cutlass want ModelOpt's amax/2688.
        weight_global_scale = layer.weight_scale_2.max().to(torch.float32)
        layer.weight_global_scale = Parameter(
            weight_global_scale, requires_grad=False
        )
        del layer.weight_scale_2


class KNvfp4Dynamic(QuantKeyScheme):
    """NVFP4 activation scheme (W4A4). Has a static global input scale on disk;
    the per-group scale is computed at runtime inside the kernel."""

    key = kNvfp4Dynamic

    def create_weights(self, layer, role, ctx, shapes, wl) -> None:
        if role is not ACT:
            self.reject(role)
        self.register_params(
            layer,
            "input_scale",
            (shapes.nparts,),
            torch.float32,
            PerTensorScaleParameter,
            wl,
        )

    def process(self, layer, role) -> None:
        if role is not ACT:
            self.reject(role)
        if torch.unique(layer.input_scale).numel() != 1:
            logger.warning_once(
                "In NVFP4 linear, the global input scale differs across "
                "parallel layers (e.g. q_proj, k_proj, v_proj). This will "
                "likely reduce accuracy. Consider a checkpoint with a shared "
                "global NVFP4 scale for parallel layers."
            )
        input_global_scale = layer.input_scale.max().to(torch.float32)
        layer.input_global_scale = Parameter(
            input_global_scale, requires_grad=False
        )
        layer.input_global_scale_inv = Parameter(
            (1.0 / layer.input_global_scale).to(torch.float32),
            requires_grad=False,
        )
        del layer.input_scale


class KFp8StaticTensor(QuantKeyScheme):
    """Plain per-tensor static FP8 — bivalent: serves BOTH the weight slot and
    the activation slot (W8A8). One key in both QuantSpec slots."""

    key = kFp8StaticTensorSym
    requires_serialized = False  # FP8 alone allows a non-serialized checkpoint
    exposes_input_quant_key = False  # old FP8 method did not expose it (C2)

    def create_weights(self, layer, role, ctx, shapes, wl) -> None:
        if role is WEIGHT:
            weight_dtype = (
                torch.float8_e4m3fn if ctx.serialized else shapes.params_dtype
            )
            self.register_params(
                layer, "weight", (shapes.out, shapes.in_), weight_dtype,
                ModelWeightParameter, wl, input_dim=1, output_dim=0,
            )
            layer.orig_dtype = shapes.params_dtype
            if ctx.serialized:
                self.register_params(
                    layer, "weight_scale", (shapes.nparts,), torch.float32,
                    PerTensorScaleParameter, wl, init=SENTINEL,
                )
        elif role is ACT:
            if ctx.serialized:
                self.register_params(
                    layer, "input_scale", (shapes.nparts,), torch.float32,
                    PerTensorScaleParameter, wl, init=SENTINEL,
                )
        else:
            self.reject(role)

    def process(self, layer, role) -> None:
        if role is WEIGHT:
            weight = layer.weight
            max_w_scale = layer.weight_scale.max()
            if not (layer.weight_scale == layer.weight_scale[0]).all():
                max_w_scale, weight = requantize_with_max_scale(
                    layer.weight, layer.weight_scale, layer.logical_widths
                )
            # Transpose lives here (Scope A; belongs to the kernel — C1).
            layer.weight = Parameter(weight.t(), requires_grad=False)
            layer.weight_scale = Parameter(max_w_scale, requires_grad=False)
        elif role is ACT:
            layer.input_scale = Parameter(
                layer.input_scale.max(), requires_grad=False
            )
        else:
            self.reject(role)


class KFp8StaticChannel(QuantKeyScheme):
    """Per-channel static FP8 weight (the 'PcPt' weight). Weight-role only —
    there is no static per-channel *activation* today."""

    key = kFp8StaticTokenSym
    exposes_input_quant_key = False  # old PcPt method did not expose it

    def create_weights(self, layer, role, ctx, shapes, wl) -> None:
        if role is not WEIGHT:
            self.reject(role)
        self.register_params(
            layer, "weight", (shapes.out, shapes.in_), torch.float8_e4m3fn,
            ModelWeightParameter, wl, input_dim=1, output_dim=0,
        )
        self.register_params(
            layer, "weight_scale", (shapes.out,), torch.float32,
            ChannelQuantScaleParameter, wl, output_dim=0, init=SENTINEL,
        )

    def process(self, layer, role) -> None:
        if role is not WEIGHT:
            self.reject(role)
        weight, weight_scale, _ = process_fp8_weight_channel_strategy(
            layer.weight, layer.weight_scale.data
        )
        layer.weight = Parameter(weight.t(), requires_grad=False)  # C1 (Scope A)
        layer.weight_scale = Parameter(weight_scale, requires_grad=False)


class KFp8Block128(QuantKeyScheme):
    """128x128 block-static FP8 weight ('PbWo'). Weight-role only. ModelOpt
    exports the scale 4-D [out_blk,1,in_blk,1] (C6); process squeezes to 2-D.
    No transpose (block kernel keeps [out,in])."""

    key = kFp8Static128BlockSym
    exposes_input_quant_key = False

    def create_weights(self, layer, role, ctx, shapes, wl) -> None:
        if role is not WEIGHT:
            self.reject(role)
        if shapes.out % 128 != 0 or shapes.in_ % 128 != 0:
            raise ValueError(
                f"FP8_PB_WO requires out/in divisible by 128, got "
                f"{shapes.out}x{shapes.in_}"
            )
        self.register_params(
            layer, "weight", (shapes.out, shapes.in_), torch.float8_e4m3fn,
            ModelWeightParameter, wl, input_dim=1, output_dim=0,
        )
        ob, ib = shapes.out // 128, shapes.in_ // 128
        self.register_params(
            layer, "weight_scale", (ob, 1, ib, 1), torch.float32,
            BlockQuantScaleParameter, wl, input_dim=2, output_dim=0,
            init=SENTINEL,
        )
        layer.weight_block_size = [128, 128]

    def process(self, layer, role) -> None:
        if role is not WEIGHT:
            self.reject(role)
        layer.weight = Parameter(layer.weight.data, requires_grad=False)
        s = layer.weight_scale
        if s.dim() == 4:
            s = s.squeeze(1).squeeze(-1)  # [ob,1,ib,1] -> [ob,ib]
        elif s.dim() != 2:
            raise ValueError(
                f"Unexpected FP8_PB_WO weight_scale shape {tuple(s.shape)}"
            )
        layer.weight_scale = Parameter(s.contiguous(), requires_grad=False)


class KMxfp8Static(QuantKeyScheme):
    """MXFP8 weight: fp8-e4m3 values + per-32-block e8m0 (uint8) scale.
    Weight-role only. process is validate-only + idempotency guard (C13)."""

    key = kMxfp8Static
    exposes_input_quant_key = False

    def create_weights(self, layer, role, ctx, shapes, wl) -> None:
        if role is not WEIGHT:
            self.reject(role)
        if shapes.in_ % MXFP8_BLOCK_SIZE != 0:
            raise ValueError(
                f"MXFP8 requires in divisible by {MXFP8_BLOCK_SIZE}, "
                f"got {shapes.in_}"
            )
        self.register_params(
            layer, "weight", (shapes.out, shapes.in_), MXFP8_VALUE_DTYPE,
            ModelWeightParameter, wl, input_dim=1, output_dim=0,
        )
        self.register_params(
            layer, "weight_scale",
            (shapes.out, shapes.in_ // MXFP8_BLOCK_SIZE), MXFP8_SCALE_DTYPE,
            ModelWeightParameter, wl, input_dim=1, output_dim=0,
        )

    def process(self, layer, role) -> None:
        if role is not WEIGHT:
            self.reject(role)
        # Idempotency: emulation kernel may dequant weight to >=2-byte at load.
        if layer.weight.element_size() >= 2:
            return
        assert layer.weight.ndim == 2 and layer.weight.dtype == MXFP8_VALUE_DTYPE
        assert layer.weight_scale.ndim == 2
        assert layer.weight_scale.dtype == MXFP8_SCALE_DTYPE


class KDynamicNoParam(QuantKeyScheme):
    """Dynamic activation with no stored scale (W8A8): quantized at runtime in
    the kernel. NOT the same as activation=None (weight-only) — init_fp8 needs a
    non-None activation key. Activation-role only. Serves the fp8 per-token, fp8
    per-block, and mxfp8 dynamic activation keys."""

    requires_serialized = False

    def create_weights(self, layer, role, ctx, shapes, wl) -> None:
        if role is not ACT:
            self.reject(role)
        # dynamic -> nothing stored

    def process(self, layer, role) -> None:
        if role is not ACT:
            self.reject(role)


SCHEME_FOR: dict[QuantKey | None, QuantKeyScheme] = {
    kNvfp4Static: KNvfp4Static(),
    kNvfp4Dynamic: KNvfp4Dynamic(),
    kFp8StaticTensorSym: KFp8StaticTensor(),
    kFp8StaticTokenSym: KFp8StaticChannel(),
    kFp8Static128BlockSym: KFp8Block128(),
    kMxfp8Static: KMxfp8Static(),
    kFp8DynamicTokenSym: KDynamicNoParam(),
    kFp8Dynamic128Sym: KDynamicNoParam(),
    kMxfp8Dynamic: KDynamicNoParam(),
}


# ---------------------------------------------------------------------------
# 5. The one generic cross-key rule (linear_design_concrete.md §5)
# ---------------------------------------------------------------------------


def maybe_fuse_global_scales(layer) -> None:
    """alpha = input_global_scale * weight_global_scale, presence-gated.

    W4A4 has both -> computed; W4A16 has no input_global_scale -> skipped.
    """
    if hasattr(layer, "weight_global_scale") and hasattr(
        layer, "input_global_scale"
    ):
        layer.alpha = Parameter(
            layer.input_global_scale * layer.weight_global_scale,
            requires_grad=False,
        )


# ---------------------------------------------------------------------------
# 7. Kernel selection (linear_design_concrete.md §7)
# ---------------------------------------------------------------------------


def select_linear_kernel(spec: QuantSpec, layer, rt: RuntimeDtypes):
    """Thin family dispatcher on the weight key: nvfp4 / mxfp8 / fp8."""
    w = spec.weight
    if w.dtype == FP4_DTYPE:
        if spec.activation is None:
            # W4A16: pin Marlin exactly like ModelOptNvFp4W4A16LinearMethod. We
            # can't route through init_nvfp4_linear_kernel(use_a16=True): under
            # VLLM_BATCH_INVARIANT its first branch force-selects Cutlass (W4A4),
            # whose apply reads layer.input_global_scale_inv — absent for W4A16,
            # so it AttributeErrors. Pinning matches old and is BI-safe.
            return MarlinNvFp4LinearKernel(NvFp4LinearLayerConfig())
        return init_nvfp4_linear_kernel(use_a16=False)  # W4A4 (Cutlass/etc.)
    if w.scale.dtype == MXFP8_SCALE_DTYPE:
        return init_mxfp8_linear_kernel()
    # fp8 family: init_fp8 routes block-vs-plain itself off the activation key.
    return init_fp8_linear_kernel(
        activation_quant_key=spec.activation,
        weight_quant_key=w,
        input_dtype=rt.input_dtype,
        out_dtype=rt.out_dtype,
        weight_shape=layer.weight.shape,
        module_name=type(layer).__name__,
    )


# ---------------------------------------------------------------------------
# 2. The generic base method (linear_design_concrete.md §2)
# ---------------------------------------------------------------------------


@register_weight_loader_v2_supported_method
class ModelOptLinearMethod(LinearMethodBase):
    """Generic, format-agnostic linear method. Holds a weight scheme + an
    activation scheme (from the QuantSpec pair), runs a fixed lifecycle, selects
    the kernel from the pair, and applies.

    Registered for weight_loader_v2 (like every ModelOpt linear method) so the
    ``BasevLLMParameter`` params route through the v2 fused loader, not the
    legacy shape-assert path.
    """

    def __init__(
        self,
        spec: QuantSpec,
        ctx: CkptCtx,
        format_scheme=None,
    ) -> None:
        self.spec = spec
        self.ctx = ctx
        self.wkey = SCHEME_FOR[spec.weight]
        self.akey = None if spec.activation is None else SCHEME_FOR[spec.activation]
        self.out_dtype = torch.get_default_dtype()
        # Only the fp8/mxfp8 kernels read input_dtype; nvfp4 ignores it. During
        # real serving model_config is always set (matches ModelOptFp8LinearMethod);
        # fall back defensively when it is absent (bare unit-test config).
        model_config = getattr(get_current_vllm_config(), "model_config", None)
        self.input_dtype = (
            model_config.dtype
            if model_config is not None
            else torch.get_default_dtype()
        )
        self.marlin_input_dtype = None
        # Kernel/backend are chosen in create_weights (after get_quant_method),
        # so the front-end marlin poke stays dormant for this NVFP4 slice — same
        # as today's ModelOptNvFp4LinearMethod.
        self.kernel = None

    def create_weights(
        self,
        layer,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        del input_size, output_size
        if self.wkey.requires_serialized and not self.ctx.serialized:
            raise ValueError(
                f"{self.spec.weight} requires a serialized checkpoint"
            )
        weight_loader = extra_weight_attrs.get("weight_loader")
        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = sum(output_partition_sizes)
        shapes = Shapes(output_partition_sizes, input_size_per_partition, params_dtype)

        self.wkey.create_weights(layer, WEIGHT, self.ctx, shapes, weight_loader)
        if self.akey:
            self.akey.create_weights(layer, ACT, self.ctx, shapes, weight_loader)

        rt = RuntimeDtypes(self.input_dtype, self.out_dtype, self.marlin_input_dtype)
        self.kernel = select_linear_kernel(self.spec, layer, rt)
        if self.wkey.exposes_input_quant_key:
            expose_input_quant_key(layer, self.kernel)

    def process_weights_after_loading(self, layer) -> None:
        self.wkey.process(layer, WEIGHT)
        if self.akey:
            self.akey.process(layer, ACT)
        maybe_fuse_global_scales(layer)
        self.kernel.process_weights_after_loading(layer)

    def apply(self, layer, x, bias=None):
        return self.kernel.apply_weights(layer=layer, x=x, bias=bias)


# ---------------------------------------------------------------------------
# 6. Front-end: resolve() + the flag-gated config (experimental_dispatch.md)
# ---------------------------------------------------------------------------


def resolve(algo: str, subcfg, prefix: str):
    """Turn an existing (untouched) sub-config into (QuantSpec, CkptCtx,
    format_scheme). Strictly read-only over ``subcfg`` — the one real hazard in
    mixed mode is writing back to a config shared with the imported MoE method.
    """
    if algo == "FP8":
        # Plain per-tensor static FP8 (W8A8): same key in both slots (bivalent).
        ctx = CkptCtx(
            serialized=subcfg.is_checkpoint_fp8_serialized, group_size=None
        )
        return (
            QuantSpec(weight=kFp8StaticTensorSym, activation=kFp8StaticTensorSym),
            ctx,
            None,
        )
    if algo == "FP8_PER_CHANNEL_PER_TOKEN":
        # PcPt: per-channel static weight, dynamic per-token activation (W8A8).
        ctx = CkptCtx(
            serialized=subcfg.is_checkpoint_fp8_serialized, group_size=None
        )
        return (
            QuantSpec(
                weight=kFp8StaticTokenSym, activation=kFp8DynamicTokenSym
            ),
            ctx,
            None,
        )
    if algo == "FP8_PB_WO":
        # PbWo: 128x128 block-static weight, dynamic per-block activation (W8A8).
        # C12: the generic base runs the block kernel's post-load, which the old
        # method skipped via a misnamed guard — validate vs CT block-FP8.
        ctx = CkptCtx(
            serialized=subcfg.is_checkpoint_fp8_serialized, group_size=None
        )
        return (
            QuantSpec(
                weight=kFp8Static128BlockSym, activation=kFp8Dynamic128Sym
            ),
            ctx,
            None,
        )
    if algo == "MXFP8":
        # MXFP8: block(32) e4m3 weight + e8m0 scale, dynamic activation.
        ctx = CkptCtx(
            serialized=subcfg.is_checkpoint_mxfp8_serialized, group_size=None
        )
        return QuantSpec(weight=kMxfp8Static, activation=kMxfp8Dynamic), ctx, None

    # NVFP4 family (W4A4 / W4A16).
    ctx = CkptCtx(
        serialized=subcfg.is_checkpoint_nvfp4_serialized,
        group_size=subcfg.group_size,
    )
    if algo == "NVFP4":
        # W4A4: static fp4 weight + dynamic fp4 activation (has a static global
        # input scale). alpha = weight_gs * input_gs.
        return QuantSpec(weight=kNvfp4Static, activation=kNvfp4Dynamic), ctx, None
    if algo == "W4A16_NVFP4":
        # W4A16: same fp4 weight, no activation quant. activation=None drives
        # use_a16=True in select_linear_kernel (-> Marlin) and skips alpha. The
        # old method's placeholder input_scale is intentionally dropped (C4).
        return QuantSpec(weight=kNvfp4Static, activation=None), ctx, None
    raise NotImplementedError(
        f"resolve: algo {algo!r} not supported under VLLM_MODELOPT_GENERIC yet "
        "(supported: FP8 / NVFP4 / W4A16_NVFP4)"
    )


def _generic_get_quant_method(config, layer, prefix, algo):
    """Shared get_quant_method for the flag-gated configs: mirror the built-in
    preamble, rewire ONLY the LinearBase arm to resolve() + ModelOptLinearMethod.
    The MoE / KV-cache / exclude arms stay the inherited (imported) behavior.
    ``algo`` is passed by each config (MXFP8's config carries no quant_method)."""
    if isinstance(layer, (Attention, MLAAttention)):
        return config.KVCacheMethodCls(config)

    if config.is_layer_excluded(prefix):
        if isinstance(layer, (LinearBase, ParallelLMHead)):
            return UnquantizedLinearMethod()
        return None

    if (
        "vision_tower" in prefix
        or "vision_model" in prefix
        or "vit_large_projector" in prefix
    ):
        return UnquantizedLinearMethod()

    if isinstance(layer, (LinearBase, ParallelLMHead)):
        spec, ctx, format_scheme = resolve(algo, config, prefix)
        method = ModelOptLinearMethod(spec, ctx, format_scheme)
        if getattr(method, "backend", "") == "marlin":
            method.marlin_input_dtype = get_marlin_input_dtype(prefix)
        return method
    elif isinstance(layer, RoutedExperts):
        quant_method = config.FusedMoEMethodCls(
            quant_config=config, moe_config=layer.moe_config
        )
        if getattr(quant_method, "backend", "") == "marlin":
            quant_method.marlin_input_dtype = get_marlin_input_dtype(prefix)
        return quant_method

    return None


class ModelOptGenericNvFp4Config(ModelOptNvFp4Config):
    """Flag-gated front-end for the generic NVFP4 (W4A4 / W4A16) linear path.
    Subclasses the built-in config; only the LinearBase arm of get_quant_method
    is rewired. MoE stays the inherited (imported) class."""

    def get_name(self):
        return "modelopt_generic_fp4"

    @classmethod
    def override_quantization_method(cls, hf_quant_cfg, user_quant, hf_config=None):
        if not envs.VLLM_MODELOPT_GENERIC:
            return None
        algo = cls._extract_modelopt_quant_algo(hf_quant_cfg)
        # Claims both NVFP4 (W4A4) and W4A16_NVFP4 — both contain "NVFP4".
        if algo is not None and "NVFP4" in algo:
            return "modelopt_generic_fp4"
        return None

    def get_quant_method(self, layer, prefix):
        return _generic_get_quant_method(self, layer, prefix, self.quant_method)


class ModelOptGenericFp8Config(ModelOptFp8Config):
    """Flag-gated front-end for the generic FP8 linear path. Same shape as the
    NVFP4 config but subclasses the FP8 front-end (different sub-config fields,
    get_name, override). Claims plain per-tensor FP8 only this slice."""

    def get_name(self):
        return "modelopt_generic"

    @classmethod
    def override_quantization_method(cls, hf_quant_cfg, user_quant, hf_config=None):
        if not envs.VLLM_MODELOPT_GENERIC:
            return None
        algo = cls._extract_modelopt_quant_algo(hf_quant_cfg)
        if algo in ("FP8", "FP8_PER_CHANNEL_PER_TOKEN", "FP8_PB_WO"):
            return "modelopt_generic"
        return None

    def get_quant_method(self, layer, prefix):
        return _generic_get_quant_method(self, layer, prefix, self.quant_method)


class ModelOptGenericMxFp8Config(ModelOptMxFp8Config):
    """Flag-gated front-end for the generic MXFP8 linear path."""

    def get_name(self):
        return "modelopt_generic_mxfp8"

    @classmethod
    def override_quantization_method(cls, hf_quant_cfg, user_quant, hf_config=None):
        if not envs.VLLM_MODELOPT_GENERIC:
            return None
        algo = cls._extract_modelopt_quant_algo(hf_quant_cfg)
        if algo == "MXFP8":
            return "modelopt_generic_mxfp8"
        return None

    def get_quant_method(self, layer, prefix):
        return _generic_get_quant_method(self, layer, prefix, "MXFP8")

