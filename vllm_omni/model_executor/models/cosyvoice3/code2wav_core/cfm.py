# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adopted from https://github.com/FunAudioLLM/CosyVoice/tree/main/cosyvoice/flow
"""Conditional Flow Matching (CFM) classes for audio generation."""

import os
from abc import ABC
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.nn import functional as F
from vllm.logger import init_logger
from vllm.platforms import current_platform

from vllm_omni.model_executor.models.cosyvoice3.utils import make_pad_mask

logger = init_logger(__name__)


def _cosyvoice3_cfm_cuda_graph_enabled() -> bool:
    value = (
        os.environ.get(
            "COSYVOICE3_CFM_CUDA_GRAPH",
            os.environ.get("COSYVOICE3_CFM_CUDAGRAPH", "0"),
        )
        .strip()
        .lower()
    )
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
    """Capture and replay the complete CosyVoice3 CFM Euler loop per shape."""

    def __init__(self, *, enabled: bool, max_graphs: int = 4) -> None:
        self.enabled = bool(enabled)
        self.max_graphs = max(1, int(max_graphs))
        self._cache: OrderedDict[tuple[object, ...], _CFMEulerGraphEntry] = OrderedDict()
        self._stats = {
            "calls": 0,
            "captures": 0,
            "replays": 0,
            "fallbacks": 0,
            "failures": 0,
            "shape_misses": 0,
            "cache_full_fallbacks": 0,
        }
        self.last_call_info: dict[str, object] = {}

    def stats(self) -> dict[str, int | float]:
        stats = dict(self._stats)
        stats["cache_size"] = len(self._cache)
        stats["unique_graphs"] = len(self._cache)
        stats["total_euler_calls"] = stats["calls"]
        calls = int(stats["calls"])
        stats["replay_hit_rate"] = float(stats["replays"] / calls) if calls else 0.0
        return stats

    def try_replay(
        self,
        cfm: "ConditionalCFM",
        *,
        x: torch.Tensor,
        t_span: torch.Tensor,
        mu: torch.Tensor,
        mask: torch.Tensor,
        spks: torch.Tensor | None,
        cond: torch.Tensor | None,
    ) -> torch.Tensor | None:
        self._stats["calls"] += 1
        ineligible_reason = self._ineligible_reason(cfm, x=x, spks=spks, cond=cond)
        if ineligible_reason is not None:
            self._record_fallback(ineligible_reason, x)
            return None

        assert spks is not None and cond is not None
        key = self._key(cfm, x=x, t_span=t_span, mu=mu, mask=mask, spks=spks, cond=cond)
        entry = self._cache.get(key)
        if entry is None:
            self._stats["shape_misses"] += 1
            if len(self._cache) >= self.max_graphs:
                self._stats["cache_full_fallbacks"] += 1
                self._record_fallback("cache_full", x)
                return None
            entry = self._capture(
                cfm,
                x=x,
                t_span=t_span,
                mu=mu,
                mask=mask,
                spks=spks,
                cond=cond,
            )
            if entry is None:
                self._record_fallback("capture_failed", x)
                return None
            self._cache[key] = entry
            reason = "capture"
        else:
            self._cache.move_to_end(key)
            entry.static_x.copy_(x)
            entry.static_t_span.copy_(t_span)
            entry.static_mu.copy_(mu)
            entry.static_mask.copy_(mask)
            entry.static_spks.copy_(spks)
            entry.static_cond.copy_(cond)
            reason = "hit"

        with torch.profiler.record_function(f"cosyvoice3_cfm_cudagraph_replay_b{x.size(0)}_t{x.size(2)}"):
            entry.graph.replay()
        self._stats["replays"] += 1
        self.last_call_info = {
            "mode": "graph",
            "reason": reason,
            "shape": tuple(x.shape),
            "cache_size": len(self._cache),
        }
        return entry.static_out.clone()

    def _record_fallback(self, reason: str, x: torch.Tensor) -> None:
        self._stats["fallbacks"] += 1
        self.last_call_info = {
            "mode": "eager",
            "reason": reason,
            "shape": tuple(x.shape),
            "cache_size": len(self._cache),
        }
        with torch.profiler.record_function(f"cosyvoice3_cfm_cudagraph_fallback_b{x.size(0)}_t{x.size(2)}"):
            pass

    def _ineligible_reason(
        self,
        cfm: "ConditionalCFM",
        *,
        x: torch.Tensor,
        spks: torch.Tensor | None,
        cond: torch.Tensor | None,
    ) -> str | None:
        if not self.enabled:
            return "disabled"
        if x.device.type != "cuda":
            return "non_cuda"
        if spks is None or cond is None:
            return "missing_conditioning"
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
            mu.dtype,
            tuple(mask.shape),
            mask.dtype,
            tuple(spks.shape),
            spks.dtype,
            tuple(cond.shape),
            cond.dtype,
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
        estimator_dtype = spks.dtype
        entry = _CFMEulerGraphEntry(
            graph=None,
            static_x=x.contiguous().clone(),
            static_t_span=t_span.contiguous().clone(),
            static_mu=mu.contiguous().clone(),
            static_mask=mask.contiguous().clone(),
            static_spks=spks.contiguous().clone(),
            static_cond=cond.contiguous().clone(),
            static_x_in=torch.zeros(
                (cfg_batch, channels, timesteps),
                device=x.device,
                dtype=estimator_dtype,
            ),
            static_mask_in=torch.zeros(
                (cfg_batch, 1, timesteps),
                device=x.device,
                dtype=estimator_dtype,
            ),
            static_mu_in=torch.zeros(
                (cfg_batch, channels, timesteps),
                device=x.device,
                dtype=estimator_dtype,
            ),
            static_t_in=torch.zeros((cfg_batch,), device=x.device, dtype=estimator_dtype),
            static_spks_in=torch.zeros(
                (cfg_batch, spks.size(1)),
                device=x.device,
                dtype=estimator_dtype,
            ),
            static_cond_in=torch.zeros(
                (cfg_batch, channels, timesteps),
                device=x.device,
                dtype=estimator_dtype,
            ),
            static_out=torch.empty_like(x),
        )
        try:
            with torch.no_grad():
                self._run_static(cfm, entry)
            torch.accelerator.synchronize(x.device)

            graph = torch.cuda.CUDAGraph()
            with torch.profiler.record_function(f"cosyvoice3_cfm_cudagraph_capture_b{batch}_t{timesteps}"):
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

        with torch.profiler.record_function("cosyvoice3_cfm_noise_cache"):
            z = torch.randn_like(mu).to(mu.device).to(mu.dtype) * temperature
            cache_size = cache.shape[2]
            # fix prompt and overlap part mu and z
            if cache_size != 0:
                z[:, :, :cache_size] = cache[:, :, :, 0]
                mu[:, :, :cache_size] = cache[:, :, :, 1]
            z_cache = torch.concat([z[:, :, :prompt_len], z[:, :, -34:]], dim=2)
            mu_cache = torch.concat([mu[:, :, :prompt_len], mu[:, :, -34:]], dim=2)
            cache = torch.stack([z_cache, mu_cache], dim=-1)

        with torch.profiler.record_function("cosyvoice3_cfm_t_span"):
            t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device, dtype=mu.dtype)
            if self.t_scheduler == "cosine":
                t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)
        with torch.profiler.record_function(f"cosyvoice3_cfm_euler_{max(1, int(n_timesteps))}_steps"):
            return self.solve_euler(z, t_span=t_span, mu=mu, mask=mask, spks=spks, cond=cond), cache

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
            return graph_out.float()

        # Do not use concat, it may cause memory format changed and trt infer with wrong results!
        # NOTE when flow run in amp mode, x.dtype is float32, which cause nan in trt fp16
        # inference, so set dtype=spks.dtype.  The batch is doubled for CFG:
        # first B rows are conditioned, second B rows are unconditional.
        batch_size = int(x.size(0))
        estimator_batch = 2 * batch_size
        estimator_dtype = spks.dtype if spks is not None else x.dtype
        channels = int(x.size(1))
        spk_dim = int(spks.size(1)) if spks is not None else int(self.spk_emb_dim)
        with torch.profiler.record_function("cosyvoice3_cfm_cfg_allocate_2b"):
            x_in = torch.zeros(
                [estimator_batch, channels, x.size(2)],
                device=x.device,
                dtype=estimator_dtype,
            )
            mask_in = torch.zeros([estimator_batch, 1, x.size(2)], device=x.device, dtype=estimator_dtype)
            mu_in = torch.zeros(
                [estimator_batch, channels, x.size(2)],
                device=x.device,
                dtype=estimator_dtype,
            )
            t_in = torch.zeros([estimator_batch], device=x.device, dtype=estimator_dtype)
            spks_in = torch.zeros([estimator_batch, spk_dim], device=x.device, dtype=estimator_dtype)
            cond_in = torch.zeros(
                [estimator_batch, channels, x.size(2)],
                device=x.device,
                dtype=estimator_dtype,
            )
        for step in range(1, len(t_span)):
            # Classifier-Free Guidance inference introduced in VoiceBox
            with torch.profiler.record_function("cosyvoice3_cfm_cfg_prepare_2b"):
                x_in[:batch_size] = x
                x_in[batch_size:] = x
                mask_in[:batch_size] = mask
                mask_in[batch_size:] = mask
                mu_in[:batch_size] = mu
                t_in[:] = t
                if spks is not None:
                    spks_in[:batch_size] = spks
                if cond is not None:
                    cond_in[:batch_size] = cond
            with torch.profiler.record_function("cosyvoice3_cfm_forward_estimator"):
                dphi_dt = self.forward_estimator(x_in, mask_in, mu_in, t_in, spks_in, cond_in)
            with torch.profiler.record_function("cosyvoice3_cfm_cfg_combine"):
                dphi_dt, cfg_dphi_dt = torch.split(dphi_dt, [batch_size, batch_size], dim=0)
                dphi_dt = (1.0 + self.inference_cfg_rate) * dphi_dt - self.inference_cfg_rate * cfg_dphi_dt
            with torch.profiler.record_function("cosyvoice3_cfm_euler_update"):
                x = x + dt * dphi_dt
                t = t + dt
            sol.append(x)
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t

        return sol[-1].float()

    def forward_estimator(self, x, mask, mu, t, spks, cond):
        if isinstance(self.estimator, torch.nn.Module):
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
            with stream:
                x_e = x.to(io_dtype).contiguous()
                mask_e = mask.to(io_dtype).contiguous()
                mu_e = mu.to(io_dtype).contiguous()
                t_e = t.to(io_dtype).contiguous()
                spks_e = spks.to(io_dtype).contiguous()
                cond_e = cond.to(io_dtype).contiguous()
                out_e = torch.empty_like(x_e)
                estimator.set_input_shape("x", tuple(x_e.shape))
                estimator.set_input_shape("mask", tuple(mask_e.shape))
                estimator.set_input_shape("mu", tuple(mu_e.shape))
                estimator.set_input_shape("t", tuple(t_e.shape))
                estimator.set_input_shape("spks", tuple(spks_e.shape))
                estimator.set_input_shape("cond", tuple(cond_e.shape))
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

        with torch.profiler.record_function("cosyvoice3_cfm_noise_cache"):
            z = (
                torch.randn(
                    (mu.size(0), mu.size(1), mu.size(2)),
                    device=mu.device,
                    dtype=mu.dtype,
                )
                * temperature
            )

        # fix prompt and overlap part mu and z
        with torch.profiler.record_function("cosyvoice3_cfm_t_span"):
            t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device, dtype=mu.dtype)

            if self.t_scheduler == "cosine":
                t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)

        with torch.profiler.record_function(f"cosyvoice3_cfm_euler_{max(1, int(n_timesteps))}_steps"):
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
    ):
        with torch.profiler.record_function("cosyvoice3_cfm_speaker_embedding"):
            embedding = F.normalize(embedding, dim=1)
            embedding = self.spk_embed_affine_layer(embedding)

        with torch.profiler.record_function("cosyvoice3_cfm_token_embedding_lookahead"):
            # concat text and prompt_text
            codec_token_len = token_len
            token, total_token_len = torch.concat([prompt_token, token], dim=1), prompt_token_len + codec_token_len
            mask = (~make_pad_mask(total_token_len, max_len=token.shape[1])).unsqueeze(-1).to(embedding)
            token = self.input_embedding(torch.clamp(token, min=0)) * mask
            # text encode
            if finalize is True:
                h = self.pre_lookahead_layer(token)
            else:
                h = self.pre_lookahead_layer(
                    token[:, : -self.pre_lookahead_len], context=token[:, -self.pre_lookahead_len :]
                )

        with torch.profiler.record_function("cosyvoice3_cfm_repeat_to_mel_axis"):
            h = h.repeat_interleave(self.token_mel_ratio, dim=1)

        batch_size = int(token.shape[0])
        with torch.profiler.record_function("cosyvoice3_cfm_cond_prompt_mel"):
            mel_len1, mel_len2 = prompt_feat.shape[1], h.shape[1] - prompt_feat.shape[1]

            # get conditions
            conds = torch.zeros([batch_size, mel_len1 + mel_len2, self.output_size], device=token.device).to(h.dtype)
            conds[:, :mel_len1] = prompt_feat
            conds = conds.transpose(1, 2)

            lookahead = 0 if finalize else int(self.pre_lookahead_len)
            valid_h_lens = torch.clamp(total_token_len.to(torch.long) - lookahead, min=0)
            mel_lens = torch.clamp(valid_h_lens * int(self.token_mel_ratio), max=mel_len1 + mel_len2)
            mask = (~make_pad_mask(mel_lens, max_len=mel_len1 + mel_len2)).to(h)

        feat, _ = self.decoder(
            mu=h.transpose(1, 2).contiguous(),
            mask=mask.unsqueeze(1),
            spks=embedding,
            cond=conds,
            n_timesteps=max(1, int(n_timesteps)),
            streaming=streaming,
        )

        with torch.profiler.record_function("cosyvoice3_cfm_crop_prompt_mel"):
            feat = feat[:, :, mel_len1:]
            assert feat.shape[2] == mel_len2
        return feat.float(), None
