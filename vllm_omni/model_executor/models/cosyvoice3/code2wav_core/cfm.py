# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adopted from https://github.com/FunAudioLLM/CosyVoice/tree/main/cosyvoice/flow
"""Conditional Flow Matching (CFM) classes for audio generation."""

import os
import time
from abc import ABC
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.nn import functional as F
from torch.profiler import record_function
from vllm.logger import init_logger
from vllm.platforms import current_platform

from vllm_omni.model_executor.models.cosyvoice3.utils import make_pad_mask

logger = init_logger(__name__)


def _cosyvoice3_cfm_cuda_graph_enabled() -> bool:
    value = os.environ.get(
        "COSYVOICE3_CFM_CUDA_GRAPH",
        os.environ.get("COSYVOICE3_CFM_CUDAGRAPH", "0"),
    ).strip().lower()
    return value in ("1", "true", "yes", "on")


def _cosyvoice3_cfm_cuda_graph_max_graphs() -> int:
    value = os.environ.get("COSYVOICE3_CFM_CUDA_GRAPH_MAX_GRAPHS", "4")
    try:
        return max(1, int(value))
    except ValueError:
        logger.warning(
            "CosyVoice3 CFM CUDA Graph: unsupported COSYVOICE3_CFM_CUDA_GRAPH_MAX_GRAPHS=%r; using 4",
            value,
        )
        return 4


def _cosyvoice3_prompt_prefix_cache_enabled() -> bool:
    value = os.environ.get("COSYVOICE3_PROMPT_PREFIX_CACHE", "0").strip().lower()
    return value in ("1", "true", "yes", "on")


def _cosyvoice3_prompt_prefix_cache_max_size() -> int:
    value = os.environ.get("COSYVOICE3_PROMPT_PREFIX_CACHE_MAX_SIZE", "16")
    try:
        return max(1, int(value))
    except ValueError:
        logger.warning(
            "CosyVoice3 prompt prefix cache: unsupported COSYVOICE3_PROMPT_PREFIX_CACHE_MAX_SIZE=%r; using 16",
            value,
        )
        return 16


@dataclass(frozen=True)
class FlowAttentionBatchMetadata:
    """Shape/length metadata for a CosyVoice3 flow batch.

    PR2 keeps dense Attention execution, but centralizes the right-padding
    lengths so PR3 can reuse the same object for varlen Attention metadata.
    """

    token_lens: torch.Tensor
    prompt_token_lens: torch.Tensor
    prompt_feat_lens: torch.Tensor
    total_token_lens: torch.Tensor
    total_mel_lens: torch.Tensor
    generated_mel_lens: torch.Tensor
    max_seqlen: int

    @property
    def batch_size(self) -> int:
        return int(self.token_lens.numel())

    def __getitem__(self, key: str):
        return getattr(self, key)


@dataclass
class FlowPromptPrefixCacheEntry:
    prompt_token_embed: torch.Tensor
    stable_h_prefix: torch.Tensor
    prompt_feat: torch.Tensor
    spk_embedding: torch.Tensor
    prompt_token_len: int
    prompt_feat_len: int
    dtype: torch.dtype
    device: torch.device
    input_size: int
    output_size: int
    pre_lookahead_len: int
    build_ms: float


class FlowPromptPrefixCache:
    """LRU cache for immutable CosyVoice3 prompt/reference flow prefixes."""

    def __init__(self, *, enabled: bool, max_size: int) -> None:
        self.enabled = bool(enabled)
        self.max_size = int(max_size)
        self._cache: OrderedDict[str, FlowPromptPrefixCacheEntry] = OrderedDict()
        self._stats = {
            "hits": 0,
            "misses": 0,
            "bypasses": 0,
            "evictions": 0,
            "shape_mismatches": 0,
            "build_preprocess_ms": 0.0,
            "saved_preprocess_ms": 0.0,
        }

    def stats(self) -> dict[str, int | float]:
        stats = dict(self._stats)
        stats["cache_size"] = len(self._cache)
        lookups = int(stats["hits"]) + int(stats["misses"])
        stats["hit_rate"] = float(stats["hits"] / lookups) if lookups > 0 else 0.0
        return stats

    def record_bypass(self, count: int = 1) -> None:
        self._stats["bypasses"] += int(count)

    def get_or_build(
        self,
        flow: "CausalMaskedDiffWithDiT",
        *,
        spk_id: str | None,
        prompt_token: torch.Tensor,
        prompt_token_len: int,
        prompt_feat: torch.Tensor,
        prompt_feat_len: int,
        embedding: torch.Tensor,
    ) -> FlowPromptPrefixCacheEntry | None:
        if not self.enabled:
            self.record_bypass()
            return None
        if not spk_id:
            self.record_bypass()
            with record_function("cosyvoice3_prompt_prefix_cache_bypass"):
                pass
            return None

        entry = self._cache.get(spk_id)
        if entry is not None:
            if self._compatible(
                entry,
                prompt_token_len=prompt_token_len,
                prompt_feat_len=prompt_feat_len,
                dtype=prompt_feat.dtype,
                device=prompt_feat.device,
                input_size=flow.input_size,
                output_size=flow.output_size,
                pre_lookahead_len=flow.pre_lookahead_len,
            ):
                self._cache.move_to_end(spk_id)
                self._stats["hits"] += 1
                self._stats["saved_preprocess_ms"] += entry.build_ms
                with record_function("cosyvoice3_prompt_prefix_cache_hit"):
                    pass
                return entry
            self._stats["shape_mismatches"] += 1
            self.record_bypass()
            with record_function("cosyvoice3_prompt_prefix_cache_shape_mismatch"):
                pass
            return None

        with record_function("cosyvoice3_prompt_prefix_cache_miss"):
            started = time.perf_counter()
            entry = self._build_entry(
                flow,
                prompt_token=prompt_token,
                prompt_token_len=prompt_token_len,
                prompt_feat=prompt_feat,
                prompt_feat_len=prompt_feat_len,
                embedding=embedding,
            )
            entry.build_ms = (time.perf_counter() - started) * 1000.0
        self._stats["misses"] += 1
        self._stats["build_preprocess_ms"] += entry.build_ms
        self._cache[spk_id] = entry
        self._cache.move_to_end(spk_id)
        while len(self._cache) > self.max_size:
            self._cache.popitem(last=False)
            self._stats["evictions"] += 1
        return entry

    def _compatible(
        self,
        entry: FlowPromptPrefixCacheEntry,
        *,
        prompt_token_len: int,
        prompt_feat_len: int,
        dtype: torch.dtype,
        device: torch.device,
        input_size: int,
        output_size: int,
        pre_lookahead_len: int,
    ) -> bool:
        return (
            entry.prompt_token_len == prompt_token_len
            and entry.prompt_feat_len == prompt_feat_len
            and entry.dtype == dtype
            and entry.device == device
            and entry.input_size == input_size
            and entry.output_size == output_size
            and entry.pre_lookahead_len == pre_lookahead_len
        )

    def _build_entry(
        self,
        flow: "CausalMaskedDiffWithDiT",
        *,
        prompt_token: torch.Tensor,
        prompt_token_len: int,
        prompt_feat: torch.Tensor,
        prompt_feat_len: int,
        embedding: torch.Tensor,
    ) -> FlowPromptPrefixCacheEntry:
        valid_prompt_token = prompt_token[:, :prompt_token_len]
        prompt_token_embed = flow.input_embedding(torch.clamp(valid_prompt_token, min=0)).detach().contiguous()
        stable_h_len = max(0, prompt_token_len - int(flow.pre_lookahead_len))
        if stable_h_len > 0:
            context = prompt_token_embed[:, stable_h_len:prompt_token_len]
            stable_h_prefix = flow.pre_lookahead_layer(prompt_token_embed[:, :stable_h_len], context=context)
            stable_h_prefix = stable_h_prefix[:, :stable_h_len].detach().contiguous()
        else:
            stable_h_prefix = prompt_token_embed.new_zeros((1, 0, flow.input_size))
        prompt_feat_prefix = prompt_feat[:, :prompt_feat_len].detach().contiguous()
        spk_embedding = flow.spk_embed_affine_layer(F.normalize(embedding, dim=1)).detach().contiguous()
        return FlowPromptPrefixCacheEntry(
            prompt_token_embed=prompt_token_embed,
            stable_h_prefix=stable_h_prefix,
            prompt_feat=prompt_feat_prefix,
            spk_embedding=spk_embedding,
            prompt_token_len=prompt_token_len,
            prompt_feat_len=prompt_feat_len,
            dtype=prompt_feat.dtype,
            device=prompt_feat.device,
            input_size=flow.input_size,
            output_size=flow.output_size,
            pre_lookahead_len=int(flow.pre_lookahead_len),
            build_ms=0.0,
        )


@dataclass
class _CFMEulerGraphEntry:
    graph: Any
    static_x: torch.Tensor
    static_t_span: torch.Tensor
    static_mu: torch.Tensor
    static_mask: torch.Tensor
    static_spks: torch.Tensor
    static_cond: torch.Tensor
    static_x_in: torch.Tensor
    static_mask_in: torch.Tensor
    static_mu_in: torch.Tensor
    static_t_in: torch.Tensor
    static_spks_in: torch.Tensor
    static_cond_in: torch.Tensor
    static_out: torch.Tensor


class CUDAGraphCFMEulerRunner:
    """Per-shape CUDA Graph replay for the CosyVoice3 CFM Euler loop."""

    def __init__(self, *, enabled: bool, max_graphs: int = 4) -> None:
        self.enabled = bool(enabled)
        self.max_graphs = int(max_graphs)
        self._cache: OrderedDict[tuple[object, ...], _CFMEulerGraphEntry] = OrderedDict()
        self._stats = {
            "calls": 0,
            "captures": 0,
            "replays": 0,
            "fallbacks": 0,
            "failures": 0,
            "shape_misses": 0,
            "cache_full_fallbacks": 0,
            "cache_size": 0,
        }
        self.last_call_info: dict[str, object] = {}

    def stats(self) -> dict[str, int | float]:
        stats = dict(self._stats)
        stats["cache_size"] = len(self._cache)
        stats["unique_graphs"] = len(self._cache)
        stats["total_euler_calls"] = stats["calls"]
        calls = int(stats["calls"])
        stats["replay_hit_rate"] = float(stats["replays"] / calls) if calls > 0 else 0.0
        return stats

    def try_replay(
        self,
        cfm: "ConditionalCFM",
        *,
        x: torch.Tensor,
        t_span: torch.Tensor,
        mu: torch.Tensor,
        mask: torch.Tensor,
        spks: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor | None:
        self._stats["calls"] += 1
        ineligible_reason = self._ineligible_reason(cfm, x=x)
        if ineligible_reason is not None:
            self._stats["fallbacks"] += 1
            self.last_call_info = {
                "mode": "eager",
                "reason": ineligible_reason,
                "shape": tuple(x.shape),
                "cache_size": len(self._cache),
            }
            with record_function(f"cosyvoice3_cfm_cudagraph_fallback_b{x.size(0)}_t{x.size(2)}"):
                pass
            return None

        key = self._key(cfm, x=x, t_span=t_span, mu=mu, mask=mask, spks=spks, cond=cond)
        entry = self._cache.get(key)
        if entry is None:
            self._stats["shape_misses"] += 1
            if len(self._cache) >= self.max_graphs:
                self._stats["fallbacks"] += 1
                self._stats["cache_full_fallbacks"] += 1
                self.last_call_info = {
                    "mode": "eager",
                    "reason": "cache_full",
                    "shape": tuple(x.shape),
                    "cache_size": len(self._cache),
                }
                with record_function(f"cosyvoice3_cfm_cudagraph_fallback_b{x.size(0)}_t{x.size(2)}"):
                    pass
                return None
            entry = self._capture(cfm, x=x, t_span=t_span, mu=mu, mask=mask, spks=spks, cond=cond)
            if entry is None:
                self._stats["fallbacks"] += 1
                with record_function(f"cosyvoice3_cfm_cudagraph_fallback_b{x.size(0)}_t{x.size(2)}"):
                    pass
                return None
            self._cache[key] = entry
            mode_reason = "capture"
        else:
            self._cache.move_to_end(key)
            entry.static_x.copy_(x)
            entry.static_t_span.copy_(t_span)
            entry.static_mu.copy_(mu)
            entry.static_mask.copy_(mask)
            entry.static_spks.copy_(spks)
            entry.static_cond.copy_(cond)
            mode_reason = "hit"

        with record_function(f"cosyvoice3_cfm_cudagraph_replay_b{x.size(0)}_t{x.size(2)}"):
            entry.graph.replay()
        self._stats["replays"] += 1
        self.last_call_info = {
            "mode": "graph",
            "reason": mode_reason,
            "shape": tuple(x.shape),
            "cache_size": len(self._cache),
        }
        return entry.static_out.clone()

    def _ineligible_reason(self, cfm: "ConditionalCFM", *, x: torch.Tensor) -> str | None:
        if not self.enabled:
            return "disabled"
        if x.device.type != "cuda":
            return "non_cuda"
        if torch.cuda.is_current_stream_capturing():
            return "nested_capture"
        if not isinstance(cfm.estimator, torch.nn.Module):
            return "non_torch_estimator"
        return None

    def _key(
        self,
        cfm: "ConditionalCFM",
        *,
        x: torch.Tensor,
        t_span: torch.Tensor,
        mu: torch.Tensor,
        mask: torch.Tensor,
        spks: torch.Tensor,
        cond: torch.Tensor,
    ) -> tuple[object, ...]:
        return (
            x.device.type,
            x.device.index,
            tuple(x.shape),
            x.dtype,
            tuple(t_span.shape),
            t_span.dtype,
            tuple(mu.shape),
            tuple(mask.shape),
            tuple(spks.shape),
            tuple(cond.shape),
            cfm.t_scheduler,
            float(cfm.inference_cfg_rate),
        )

    def _capture(
        self,
        cfm: "ConditionalCFM",
        *,
        x: torch.Tensor,
        t_span: torch.Tensor,
        mu: torch.Tensor,
        mask: torch.Tensor,
        spks: torch.Tensor,
        cond: torch.Tensor,
    ) -> _CFMEulerGraphEntry | None:
        batch, channels, timesteps = x.shape
        cfg_batch = 2 * batch
        entry = _CFMEulerGraphEntry(
            graph=None,
            static_x=x.contiguous().clone(),
            static_t_span=t_span.contiguous().clone(),
            static_mu=mu.contiguous().clone(),
            static_mask=mask.contiguous().clone(),
            static_spks=spks.contiguous().clone(),
            static_cond=cond.contiguous().clone(),
            static_x_in=torch.zeros((cfg_batch, channels, timesteps), device=x.device, dtype=x.dtype),
            static_mask_in=torch.zeros((cfg_batch, 1, timesteps), device=x.device, dtype=mask.dtype),
            static_mu_in=torch.zeros((cfg_batch, channels, timesteps), device=x.device, dtype=mu.dtype),
            static_t_in=torch.zeros((cfg_batch,), device=x.device, dtype=t_span.dtype),
            static_spks_in=torch.zeros((cfg_batch, spks.size(1)), device=x.device, dtype=spks.dtype),
            static_cond_in=torch.zeros((cfg_batch, channels, timesteps), device=x.device, dtype=cond.dtype),
            static_out=torch.empty_like(x),
        )
        try:
            with torch.no_grad():
                _ = self._run_static(cfm, entry)
            torch.accelerator.synchronize(x.device)

            graph = torch.cuda.CUDAGraph()
            with record_function(f"cosyvoice3_cfm_cudagraph_capture_b{batch}_t{timesteps}"):
                with torch.no_grad(), torch.cuda.graph(graph, pool=current_platform.get_global_graph_pool()):
                    entry.static_out = self._run_static(cfm, entry)
            entry.graph = graph
        except Exception:
            logger.warning(
                "Disabling CosyVoice3 CFM CUDA Graph after capture failure for shape=%s",
                tuple(x.shape),
                exc_info=True,
            )
            self.enabled = False
            self._stats["failures"] += 1
            self.last_call_info = {
                "mode": "eager",
                "reason": "capture_failed",
                "shape": tuple(x.shape),
                "cache_size": len(self._cache),
            }
            return None
        self._stats["captures"] += 1
        return entry

    def _run_static(self, cfm: "ConditionalCFM", entry: _CFMEulerGraphEntry) -> torch.Tensor:
        batch = entry.static_x.size(0)
        x = entry.static_x
        t_span = entry.static_t_span
        for step in range(1, len(t_span)):
            dt = t_span[step] - t_span[step - 1]
            entry.static_x_in[:batch].copy_(x)
            entry.static_x_in[batch:].copy_(x)
            entry.static_mask_in[:batch].copy_(entry.static_mask)
            entry.static_mask_in[batch:].copy_(entry.static_mask)
            entry.static_mu_in.zero_()
            entry.static_mu_in[:batch].copy_(entry.static_mu)
            entry.static_t_in[:] = t_span[step - 1]
            entry.static_spks_in.zero_()
            entry.static_spks_in[:batch].copy_(entry.static_spks)
            entry.static_cond_in.zero_()
            entry.static_cond_in[:batch].copy_(entry.static_cond)
            dphi_dt = cfm.forward_estimator(
                entry.static_x_in,
                entry.static_mask_in,
                entry.static_mu_in,
                entry.static_t_in,
                entry.static_spks_in,
                entry.static_cond_in,
            )
            dphi_dt, cfg_dphi_dt = torch.split(dphi_dt, [batch, batch], dim=0)
            dphi_dt = (1.0 + cfm.inference_cfg_rate) * dphi_dt - cfm.inference_cfg_rate * cfg_dphi_dt
            x = x + dt * dphi_dt
        return x


class BASECFM(torch.nn.Module, ABC):
    def __init__(
        self,
        n_feats,
        cfm_params,
        n_spks=1,
        spk_emb_dim=128,
    ):
        super().__init__()
        self.n_feats = n_feats
        self.n_spks = n_spks
        self.spk_emb_dim = spk_emb_dim
        self.solver = cfm_params.solver
        if hasattr(cfm_params, "sigma_min"):
            self.sigma_min = cfm_params.sigma_min
        else:
            self.sigma_min = 1e-4

        self.estimator = None


class ConditionalCFM(BASECFM):
    def __init__(self, in_channels, cfm_params, n_spks=1, spk_emb_dim=64, estimator: torch.nn.Module = None):
        super().__init__(
            n_feats=in_channels,
            cfm_params=cfm_params,
            n_spks=n_spks,
            spk_emb_dim=spk_emb_dim,
        )
        self.t_scheduler = cfm_params.t_scheduler
        self.training_cfg_rate = cfm_params.training_cfg_rate
        self.inference_cfg_rate = cfm_params.inference_cfg_rate
        in_channels = in_channels + (spk_emb_dim if n_spks > 0 else 0)
        # Just change the architecture of the estimator here
        self.estimator = estimator
        self._cfg_workspace_cache: dict[tuple[int, int, int, torch.dtype, torch.device], tuple[torch.Tensor, ...]] = {}
        self._cuda_graph_runner = CUDAGraphCFMEulerRunner(
            enabled=_cosyvoice3_cfm_cuda_graph_enabled(),
            max_graphs=_cosyvoice3_cfm_cuda_graph_max_graphs(),
        )

    def get_cuda_graph_stats(self) -> dict[str, int | float]:
        return self._cuda_graph_runner.stats()

    @torch.inference_mode()
    def forward(
        self, mu, mask, n_timesteps, temperature=1.0, spks=None, cond=None, prompt_len=0, cache=torch.zeros(1, 80, 0, 2)
    ):
        """Forward diffusion

        Args:
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): output_mask
                shape: (batch_size, 1, mel_timesteps)
            n_timesteps (int): number of diffusion steps
            temperature (float, optional): temperature for scaling noise. Defaults to 1.0.
            spks (torch.Tensor, optional): speaker ids. Defaults to None.
                shape: (batch_size, spk_emb_dim)
            cond (Optional[Any], optional): Not used but kept for future purposes

        Returns:
            sample (torch.Tensor): generated mel-spectrogram
                shape: (batch_size, n_feats, mel_timesteps)
        """

        z = self._sample_masked_noise(mu, mask, temperature)
        cache_size = cache.shape[2]
        # fix prompt and overlap part mu and z
        if cache_size != 0:
            z[:, :, :cache_size] = cache[:, :, :, 0]
            mu[:, :, :cache_size] = cache[:, :, :, 1]
        z_cache = torch.concat([z[:, :, :prompt_len], z[:, :, -34:]], dim=2)
        mu_cache = torch.concat([mu[:, :, :prompt_len], mu[:, :, -34:]], dim=2)
        cache = torch.stack([z_cache, mu_cache], dim=-1)

        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device, dtype=mu.dtype)
        if self.t_scheduler == "cosine":
            t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)
        return self.solve_euler(z, t_span=t_span, mu=mu, mask=mask, spks=spks, cond=cond), cache

    @staticmethod
    def _sample_masked_noise(mu: torch.Tensor, mask: torch.Tensor, temperature: float) -> torch.Tensor:
        """Sample row-wise noise so right-padding length does not perturb rows.

        This preserves deterministic per-request comparisons between singleton
        flow execution and padded batched flow execution when the same RNG seed
        and request order are used.
        """
        z = torch.zeros_like(mu)
        valid_lens = mask.squeeze(1).to(torch.bool).sum(dim=1)
        for row, valid_len_t in enumerate(valid_lens):
            valid_len = int(valid_len_t.item())
            if valid_len > 0:
                z[row, :, :valid_len] = torch.randn(
                    (mu.size(1), valid_len),
                    device=mu.device,
                    dtype=mu.dtype,
                )
        return z * temperature

    def solve_euler(self, x, t_span, mu, mask, spks, cond):
        """
        Fixed euler solver for ODEs.
        Args:
            x (torch.Tensor): random noise
            t_span (torch.Tensor): n_timesteps interpolated
                shape: (n_timesteps + 1,)
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): output_mask
                shape: (batch_size, 1, mel_timesteps)
            spks (torch.Tensor, optional): speaker ids. Defaults to None.
                shape: (batch_size, spk_emb_dim)
            cond (Optional[Any], optional): Not used but kept for future purposes
        """
        t, _, dt = t_span[0], t_span[-1], t_span[1] - t_span[0]
        t = t.unsqueeze(dim=0)

        sol = []

        batch = x.size(0)
        # TensorRT estimator profiles are currently exported for CFG batch=2.
        if not isinstance(self.estimator, torch.nn.Module) and batch != 1:
            raise RuntimeError("CosyVoice3 TensorRT flow estimator only supports singleton flow batches")

        graph_out = self._cuda_graph_runner.try_replay(
            self,
            x=x,
            t_span=t_span,
            mu=mu,
            mask=mask,
            spks=spks,
            cond=cond,
        )
        if graph_out is not None:
            return graph_out

        x_in, mask_in, mu_in, t_in, spks_in, cond_in = self._get_cfg_workspace(
            batch=batch,
            channels=x.size(1),
            timesteps=x.size(2),
            spk_dim=spks.size(1),
            dtype=spks.dtype,
            device=x.device,
        )
        with record_function(f"cosyvoice3_flow:euler_loop_b{batch}_t{x.size(2)}"):
            for step in range(1, len(t_span)):
                # Classifier-Free Guidance inference introduced in VoiceBox
                x_in[:batch] = x
                x_in[batch:] = x
                mask_in[:batch] = mask
                mask_in[batch:] = mask
                mu_in.zero_()
                mu_in[:batch] = mu
                t_in[:] = t
                spks_in.zero_()
                spks_in[:batch] = spks
                cond_in.zero_()
                cond_in[:batch] = cond
                dphi_dt = self.forward_estimator(x_in, mask_in, mu_in, t_in, spks_in, cond_in)
                dphi_dt, cfg_dphi_dt = torch.split(dphi_dt, [batch, batch], dim=0)
                dphi_dt = (1.0 + self.inference_cfg_rate) * dphi_dt - self.inference_cfg_rate * cfg_dphi_dt
                x = x + dt * dphi_dt
                t = t + dt
                sol.append(x)
                if step < len(t_span) - 1:
                    dt = t_span[step + 1] - t

        return sol[-1]

    def _get_cfg_workspace(
        self,
        *,
        batch: int,
        channels: int,
        timesteps: int,
        spk_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor, ...]:
        key = (batch, channels, timesteps, spk_dim, dtype, device)
        workspace = self._cfg_workspace_cache.get(key)
        if workspace is None:
            cfg_batch = 2 * batch
            workspace = (
                torch.zeros([cfg_batch, channels, timesteps], device=device, dtype=dtype),
                torch.zeros([cfg_batch, 1, timesteps], device=device, dtype=dtype),
                torch.zeros([cfg_batch, channels, timesteps], device=device, dtype=dtype),
                torch.zeros([cfg_batch], device=device, dtype=dtype),
                torch.zeros([cfg_batch, spk_dim], device=device, dtype=dtype),
                torch.zeros([cfg_batch, channels, timesteps], device=device, dtype=dtype),
            )
            self._cfg_workspace_cache[key] = workspace
        return workspace

    def forward_estimator(self, x, mask, mu, t, spks, cond):
        if isinstance(self.estimator, torch.nn.Module):
            with record_function(f"cosyvoice3_flow:estimator_torch_b{x.size(0)}_t{x.size(2)}"):
                return self.estimator(x, mask, mu, t, spks, cond)
        else:
            # TensorRT estimator: bind raw device pointers. The flow runs in
            # fp32 but the engine may have fp16 I/O (strongly-typed fp16 engine),
            # so cast inputs/output to the engine's dtype at the boundary. Keep
            # references to the cast buffers alive until execute completes (a bare
            # ``.contiguous().data_ptr()`` could free the temp -> dangling ptr).
            io_dtype = getattr(self.estimator, "io_dtype", x.dtype)
            [estimator, stream], trt_engine = self.estimator.acquire_estimator()
            # NOTE need to synchronize when switching stream
            torch.cuda.current_stream().synchronize()
            with record_function(f"cosyvoice3_flow:estimator_trt_b{x.size(0)}_t{x.size(2)}"), stream:
                x_e = x.to(io_dtype).contiguous()
                mask_e = mask.to(io_dtype).contiguous()
                mu_e = mu.to(io_dtype).contiguous()
                t_e = t.to(io_dtype).contiguous()
                spks_e = spks.to(io_dtype).contiguous()
                cond_e = cond.to(io_dtype).contiguous()
                out_e = torch.empty_like(x_e)
                estimator.set_input_shape("x", (2, 80, x_e.size(2)))
                estimator.set_input_shape("mask", (2, 1, x_e.size(2)))
                estimator.set_input_shape("mu", (2, 80, x_e.size(2)))
                estimator.set_input_shape("t", (2,))
                estimator.set_input_shape("spks", (2, 80))
                estimator.set_input_shape("cond", (2, 80, x_e.size(2)))
                data_ptrs = [
                    x_e.data_ptr(),
                    mask_e.data_ptr(),
                    mu_e.data_ptr(),
                    t_e.data_ptr(),
                    spks_e.data_ptr(),
                    cond_e.data_ptr(),
                    out_e.data_ptr(),
                ]
                for i, j in enumerate(data_ptrs):
                    estimator.set_tensor_address(trt_engine.get_tensor_name(i), j)
                # run trt engine
                assert estimator.execute_async_v3(torch.cuda.current_stream().cuda_stream) is True
                torch.cuda.current_stream().synchronize()
            self.estimator.release_estimator(estimator, stream)
            return out_e.to(x.dtype)


class CausalConditionalCFM(ConditionalCFM):
    def __init__(self, in_channels, cfm_params, n_spks=1, spk_emb_dim=64, estimator: torch.nn.Module = None):
        super().__init__(in_channels, cfm_params, n_spks, spk_emb_dim, estimator)

    @torch.inference_mode()
    def forward(self, mu, mask, n_timesteps, temperature=1.0, spks=None, cond=None, streaming: bool = False):
        """Forward diffusion

        Args:
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): output_mask
                shape: (batch_size, 1, mel_timesteps)
            n_timesteps (int): number of diffusion steps
            temperature (float, optional): temperature for scaling noise. Defaults to 1.0.
            spks (torch.Tensor, optional): speaker ids. Defaults to None.
                shape: (batch_size, spk_emb_dim)
            cond (Optional[Any], optional): Not used but kept for future purposes

        Returns:
            sample (torch.Tensor): generated mel-spectrogram
                shape: (batch_size, n_feats, mel_timesteps)
        """

        z = self._sample_masked_noise(mu, mask, temperature)

        # fix prompt and overlap part mu and z
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device, dtype=mu.dtype)

        if self.t_scheduler == "cosine":
            t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)

        return self.solve_euler(z, t_span=t_span, mu=mu, mask=mask, spks=spks, cond=cond), None


class CausalMaskedDiffWithDiT(torch.nn.Module):
    def __init__(
        self,
        input_size: int = 512,
        output_size: int = 80,
        spk_embed_dim: int = 192,
        output_type: str = "mel",
        vocab_size: int = 4096,
        input_frame_rate: int = 50,
        only_mask_loss: bool = True,
        token_mel_ratio: int = 2,
        pre_lookahead_len: int = 3,
        pre_lookahead_layer: torch.nn.Module = None,
        decoder: torch.nn.Module = None,
        decoder_conf: dict = {
            "in_channels": 240,
            "out_channel": 80,
            "spk_emb_dim": 80,
            "n_spks": 1,
            "cfm_params": DictConfig(
                {
                    "sigma_min": 1e-06,
                    "solver": "euler",
                    "t_scheduler": "cosine",
                    "training_cfg_rate": 0.2,
                    "inference_cfg_rate": 0.7,
                    "reg_loss_type": "l1",
                }
            ),
            "decoder_params": {
                "channels": [256, 256],
                "dropout": 0.0,
                "attention_head_dim": 64,
                "n_blocks": 4,
                "num_mid_blocks": 12,
                "num_heads": 8,
                "act_fn": "gelu",
            },
        },
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.decoder_conf = decoder_conf
        self.vocab_size = vocab_size
        self.output_type = output_type
        self.input_frame_rate = input_frame_rate
        logger.info(f"input frame rate={self.input_frame_rate}")
        self.input_embedding = nn.Embedding(vocab_size, input_size)
        self.spk_embed_affine_layer = torch.nn.Linear(spk_embed_dim, output_size)
        self.pre_lookahead_len = pre_lookahead_len
        self.pre_lookahead_layer = pre_lookahead_layer
        self.decoder = decoder
        self.only_mask_loss = only_mask_loss
        self.token_mel_ratio = token_mel_ratio
        self._prompt_prefix_cache = FlowPromptPrefixCache(
            enabled=_cosyvoice3_prompt_prefix_cache_enabled(),
            max_size=_cosyvoice3_prompt_prefix_cache_max_size(),
        )

    def get_prompt_prefix_cache_stats(self) -> dict[str, int | float]:
        return self._prompt_prefix_cache.stats()

    @torch.inference_mode()
    def inference(
        self,
        token,
        token_len,
        prompt_token,
        prompt_token_len,
        prompt_feat,
        prompt_feat_len,
        embedding,
        streaming: bool = True,
        finalize: bool = False,
        n_timesteps: int = 10,
        prompt_prefix_cache_keys: list[str | None] | None = None,
    ):
        if prompt_prefix_cache_keys is not None and self._prompt_prefix_cache.enabled:
            if not any(prompt_prefix_cache_keys):
                self._prompt_prefix_cache.record_bypass(token.shape[0])
                with record_function("cosyvoice3_prompt_prefix_cache_bypass"):
                    pass
                return self.inference(
                    token=token,
                    token_len=token_len,
                    prompt_token=prompt_token,
                    prompt_token_len=prompt_token_len,
                    prompt_feat=prompt_feat,
                    prompt_feat_len=prompt_feat_len,
                    embedding=embedding,
                    streaming=streaming,
                    finalize=finalize,
                    n_timesteps=n_timesteps,
                    prompt_prefix_cache_keys=None,
                )
            return self._inference_with_prompt_prefix_cache(
                token=token,
                token_len=token_len,
                prompt_token=prompt_token,
                prompt_token_len=prompt_token_len,
                prompt_feat=prompt_feat,
                prompt_feat_len=prompt_feat_len,
                embedding=embedding,
                streaming=streaming,
                finalize=finalize,
                n_timesteps=n_timesteps,
                prompt_prefix_cache_keys=prompt_prefix_cache_keys,
            )

        batch = token.shape[0]
        # xvec projection

        embedding = F.normalize(embedding, dim=1)

        embedding = self.spk_embed_affine_layer(embedding)

        # concat text and prompt_text by true row lengths. A dense
        # ``torch.concat([prompt_token, token], dim=1)`` would insert prompt
        # padding before codec tokens for rows with shorter prompt lengths.
        speech_token_len = token_len
        total_token_len = prompt_token_len + speech_token_len
        max_total_token_len = int(total_token_len.max().item()) if batch > 0 else 0
        combined_token = token.new_zeros((batch, max_total_token_len))
        for row in range(batch):
            prompt_len = int(prompt_token_len[row].item())
            speech_len = int(speech_token_len[row].item())
            if prompt_len > 0:
                combined_token[row, :prompt_len] = prompt_token[row, :prompt_len]
            if speech_len > 0:
                combined_token[row, prompt_len : prompt_len + speech_len] = token[row, :speech_len]
        token, token_len = combined_token, total_token_len
        mask = (~make_pad_mask(token_len, max_len=max_total_token_len)).unsqueeze(-1).to(embedding)
        token = self.input_embedding(torch.clamp(token, min=0)) * mask
        # text encode. For heterogeneous right-padded batches, the streaming
        # lookahead context must be sliced at each row's true valid tail rather
        # than at the padded tensor's global tail.
        h_lens = token_len if finalize is True else torch.clamp(token_len - self.pre_lookahead_len, min=0)
        max_h_len = int(h_lens.max().item()) if batch > 0 else 0
        h = token.new_zeros((batch, max_h_len, self.input_size))
        for row in range(batch):
            valid_tokens = int(token_len[row].item())
            out_tokens = int(h_lens[row].item())
            if out_tokens == 0:
                continue
            row_token = token[row : row + 1, :valid_tokens]
            if finalize is True:
                row_h = self.pre_lookahead_layer(row_token)
            else:
                row_h = self.pre_lookahead_layer(
                    row_token[:, :out_tokens],
                    context=row_token[:, out_tokens:valid_tokens],
                )
            h[row : row + 1, :out_tokens] = row_h[:, :out_tokens]
        h = h.repeat_interleave(self.token_mel_ratio, dim=1)
        total_mel_lens = h_lens * int(self.token_mel_ratio)
        generated_mel_lens = torch.clamp(total_mel_lens - prompt_feat_len, min=0)
        max_generated_mel_len = int(generated_mel_lens.max().item()) if batch > 0 else 0

        # get conditions
        conds = torch.zeros([batch, h.shape[1], self.output_size], device=token.device).to(h.dtype)
        for row in range(batch):
            prompt_len = int(prompt_feat_len[row].item())
            if prompt_len > 0:
                conds[row, :prompt_len] = prompt_feat[row, :prompt_len]

        conds = conds.transpose(1, 2)

        mask = (~make_pad_mask(total_mel_lens, max_len=h.shape[1])).to(h)
        feat, _ = self.decoder(
            mu=h.transpose(1, 2).contiguous(),
            mask=mask.unsqueeze(1),
            spks=embedding,
            cond=conds,
            n_timesteps=max(1, int(n_timesteps)),
            streaming=streaming,
        )

        generated = feat.new_zeros((batch, self.output_size, max_generated_mel_len))
        for row in range(batch):
            start = int(prompt_feat_len[row].item())
            length = int(generated_mel_lens[row].item())
            if length > 0:
                generated[row, :, :length] = feat[row, :, start : start + length]
        metadata = FlowAttentionBatchMetadata(
            token_lens=token_len - prompt_token_len,
            prompt_token_lens=prompt_token_len,
            prompt_feat_lens=prompt_feat_len,
            total_token_lens=total_token_len,
            total_mel_lens=total_mel_lens,
            generated_mel_lens=generated_mel_lens,
            max_seqlen=int(h.shape[1]),
        )
        return generated, metadata

    def _inference_with_prompt_prefix_cache(
        self,
        *,
        token,
        token_len,
        prompt_token,
        prompt_token_len,
        prompt_feat,
        prompt_feat_len,
        embedding,
        streaming: bool,
        finalize: bool,
        n_timesteps: int,
        prompt_prefix_cache_keys: list[str | None],
    ):
        batch = token.shape[0]
        if len(prompt_prefix_cache_keys) != batch:
            self._prompt_prefix_cache.record_bypass(batch)
            return self.inference(
                token=token,
                token_len=token_len,
                prompt_token=prompt_token,
                prompt_token_len=prompt_token_len,
                prompt_feat=prompt_feat,
                prompt_feat_len=prompt_feat_len,
                embedding=embedding,
                streaming=streaming,
                finalize=finalize,
                n_timesteps=n_timesteps,
                prompt_prefix_cache_keys=None,
            )

        speech_token_len = token_len
        total_token_len = prompt_token_len + speech_token_len
        h_lens = total_token_len if finalize is True else torch.clamp(total_token_len - self.pre_lookahead_len, min=0)
        max_h_len = int(h_lens.max().item()) if batch > 0 else 0
        h = prompt_feat.new_zeros((batch, max_h_len, self.input_size))
        spk_rows: list[torch.Tensor] = []
        entries: list[FlowPromptPrefixCacheEntry | None] = []

        for row in range(batch):
            p_tok_len = int(prompt_token_len[row].item())
            p_feat_len = int(prompt_feat_len[row].item())
            key = prompt_prefix_cache_keys[row]
            entry = self._prompt_prefix_cache.get_or_build(
                self,
                spk_id=key,
                prompt_token=prompt_token[row : row + 1],
                prompt_token_len=p_tok_len,
                prompt_feat=prompt_feat[row : row + 1],
                prompt_feat_len=p_feat_len,
                embedding=embedding[row : row + 1],
            )
            entries.append(entry)
            if entry is None:
                spk_rows.append(self.spk_embed_affine_layer(F.normalize(embedding[row : row + 1], dim=1)))
            else:
                spk_rows.append(entry.spk_embedding)

        projected_embedding = torch.cat(spk_rows, dim=0) if spk_rows else embedding.new_zeros((0, self.output_size))

        for row in range(batch):
            p_tok_len = int(prompt_token_len[row].item())
            speech_len = int(speech_token_len[row].item())
            valid_tokens = int(total_token_len[row].item())
            out_tokens = int(h_lens[row].item())
            if out_tokens == 0:
                continue

            entry = entries[row]
            if entry is None:
                prompt_embed = self.input_embedding(torch.clamp(prompt_token[row : row + 1, :p_tok_len], min=0))
                stable_h = None
            else:
                prompt_embed = entry.prompt_token_embed
                stable_h = entry.stable_h_prefix
            speech_embed = self.input_embedding(torch.clamp(token[row : row + 1, :speech_len], min=0))
            row_token = torch.cat([prompt_embed, speech_embed], dim=1)

            stable_to_use = 0
            if stable_h is not None and stable_h.size(1) > 0:
                stable_to_use = min(int(stable_h.size(1)), out_tokens)
                h[row : row + 1, :stable_to_use] = stable_h[:, :stable_to_use]
            if out_tokens <= stable_to_use:
                continue

            # Conv2 in PreLookaheadLayer uses two left conv1 outputs, so include
            # a two-token overlap before the uncached suffix and discard it.
            suffix_start = max(0, stable_to_use - 2)
            suffix_inputs = row_token[:, suffix_start:valid_tokens]
            local_out_tokens = out_tokens - suffix_start
            if finalize is True:
                suffix_h = self.pre_lookahead_layer(suffix_inputs)
            else:
                suffix_h = self.pre_lookahead_layer(
                    suffix_inputs[:, :local_out_tokens],
                    context=suffix_inputs[:, local_out_tokens:],
                )
            copy_start = stable_to_use - suffix_start
            h[row : row + 1, stable_to_use:out_tokens] = suffix_h[:, copy_start:local_out_tokens]

        h = h.repeat_interleave(self.token_mel_ratio, dim=1)
        total_mel_lens = h_lens * int(self.token_mel_ratio)
        generated_mel_lens = torch.clamp(total_mel_lens - prompt_feat_len, min=0)
        max_generated_mel_len = int(generated_mel_lens.max().item()) if batch > 0 else 0

        conds = torch.zeros([batch, h.shape[1], self.output_size], device=token.device).to(h.dtype)
        for row in range(batch):
            p_feat_len = int(prompt_feat_len[row].item())
            if p_feat_len <= 0:
                continue
            entry = entries[row]
            prompt_feat_row = prompt_feat[row : row + 1, :p_feat_len] if entry is None else entry.prompt_feat
            conds[row : row + 1, :p_feat_len] = prompt_feat_row

        conds = conds.transpose(1, 2)
        mask = (~make_pad_mask(total_mel_lens, max_len=h.shape[1])).to(h)
        feat, _ = self.decoder(
            mu=h.transpose(1, 2).contiguous(),
            mask=mask.unsqueeze(1),
            spks=projected_embedding,
            cond=conds,
            n_timesteps=max(1, int(n_timesteps)),
            streaming=streaming,
        )

        generated = feat.new_zeros((batch, self.output_size, max_generated_mel_len))
        for row in range(batch):
            start = int(prompt_feat_len[row].item())
            length = int(generated_mel_lens[row].item())
            if length > 0:
                generated[row, :, :length] = feat[row, :, start : start + length]
        metadata = FlowAttentionBatchMetadata(
            token_lens=total_token_len - prompt_token_len,
            prompt_token_lens=prompt_token_len,
            prompt_feat_lens=prompt_feat_len,
            total_token_lens=total_token_len,
            total_mel_lens=total_mel_lens,
            generated_mel_lens=generated_mel_lens,
            max_seqlen=int(h.shape[1]),
        )
        return generated, metadata
