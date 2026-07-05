# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
CosyVoice3 Code2Wav Stage - Converts speech tokens to audio waveforms.

This module contains the code2wav (token-to-waveform) stage which uses:
1. DiT (Diffusion Transformer) with optimized attention backends
2. CFM (Conditional Flow Matching) for mel spectrogram generation
3. HiFiGAN vocoder for waveform synthesis
"""

from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.profiler import record_function
from vllm.logger import init_logger

from vllm_omni.diffusion.models.cosyvoice3_audio.cosyvoice3_dit import DiT
from vllm_omni.model_executor.models.cosyvoice3.code2wav_core.cfm import (
    CausalConditionalCFM,
    CausalMaskedDiffWithDiT,
)
from vllm_omni.model_executor.models.cosyvoice3.code2wav_core.hifigan import (
    CausalConvRNNF0Predictor,
    CausalHiFTGenerator,
)
from vllm_omni.model_executor.models.cosyvoice3.code2wav_core.layers import PreLookaheadLayer
from vllm_omni.transformers_utils.configs.cosyvoice3 import CosyVoice3Config

logger = init_logger(__name__)


def _cosyvoice3_flow_dtype() -> torch.dtype:
    """Requested dtype for the CosyVoice3 flow model.

    Default is fp32 for compatibility. ``fp16`` is an opt-in path used for
    flow-matching experiments and profiling.
    """
    value = os.environ.get("COSYVOICE3_FLOW_DTYPE", "fp32").strip().lower()
    if value in ("", "fp32", "float32"):
        return torch.float32
    if value in ("fp16", "float16"):
        return torch.float16
    logger.warning(
        "CosyVoice3 code2wav: unsupported COSYVOICE3_FLOW_DTYPE=%r; using fp32",
        value,
    )
    return torch.float32


def _cosyvoice3_flow_batch_enabled() -> bool:
    return os.environ.get("COSYVOICE3_FLOW_BATCH", "0") in ("1", "true", "True")


def _cosyvoice3_flow_batch_wait_ms() -> int:
    value = os.environ.get("COSYVOICE3_FLOW_BATCH_WAIT_MS", "10")
    try:
        return max(0, int(value))
    except ValueError:
        logger.warning(
            "CosyVoice3 code2wav: unsupported COSYVOICE3_FLOW_BATCH_WAIT_MS=%r; using 10",
            value,
        )
        return 10


def _cosyvoice3_flow_batch_max_size() -> int:
    value = os.environ.get("COSYVOICE3_FLOW_BATCH_MAX_SIZE", "4")
    try:
        return max(1, int(value))
    except ValueError:
        logger.warning(
            "CosyVoice3 code2wav: unsupported COSYVOICE3_FLOW_BATCH_MAX_SIZE=%r; using 4",
            value,
        )
        return 4


class CosyVoice3Code2Wav(nn.Module):
    """CosyVoice3 Code2Wav stage for token-to-waveform conversion.

    This class encapsulates:
    - Flow matching decoder with DiT backbone (using diffusion attention)
    - HiFiGAN vocoder for mel-to-waveform conversion
    """

    def __init__(self, config: CosyVoice3Config):
        super().__init__()
        self.config = config

        # Build flow matching components
        pre_lookahead_layer = PreLookaheadLayer(**config.flow["pre_lookahead_layer"])

        decoder_cfg = config.flow["decoder"]
        cfm_params = DictConfig(decoder_cfg["cfm_params"])

        # DiT estimator using diffusion attention (Flash/Sage/SDPA backends)
        estimator = DiT(**decoder_cfg["estimator"])

        decoder = CausalConditionalCFM(
            in_channels=decoder_cfg["in_channels"],
            estimator=estimator,
            cfm_params=cfm_params,
            n_spks=decoder_cfg["n_spks"],
            spk_emb_dim=decoder_cfg["spk_emb_dim"],
        )

        self.flow_model = CausalMaskedDiffWithDiT(
            input_size=config.flow["input_size"],
            output_size=config.flow["output_size"],
            spk_embed_dim=config.flow["spk_embed_dim"],
            output_type=config.flow["output_type"],
            vocab_size=config.flow["vocab_size"],
            input_frame_rate=config.flow["input_frame_rate"],
            only_mask_loss=config.flow["only_mask_loss"],
            token_mel_ratio=config.flow["token_mel_ratio"],
            pre_lookahead_len=config.flow["pre_lookahead_len"],
            pre_lookahead_layer=pre_lookahead_layer,
            decoder=decoder,
        )

        # Build HiFiGAN vocoder
        f0_predictor = CausalConvRNNF0Predictor(
            num_class=config.hift["f0_predictor"]["num_class"],
            in_channels=config.hift["f0_predictor"]["in_channels"],
            cond_channels=config.hift["f0_predictor"]["cond_channels"],
        )

        self.hift = CausalHiFTGenerator(
            in_channels=config.hift["in_channels"],
            base_channels=config.hift["base_channels"],
            nb_harmonics=config.hift["nb_harmonics"],
            sampling_rate=config.hift["sampling_rate"],
            nsf_alpha=config.hift["nsf_alpha"],
            nsf_sigma=config.hift["nsf_sigma"],
            nsf_voiced_threshold=config.hift["nsf_voiced_threshold"],
            upsample_rates=config.hift["upsample_rates"],
            upsample_kernel_sizes=config.hift["upsample_kernel_sizes"],
            istft_params=config.hift["istft_params"],
            resblock_kernel_sizes=config.hift["resblock_kernel_sizes"],
            resblock_dilation_sizes=config.hift["resblock_dilation_sizes"],
            source_resblock_kernel_sizes=config.hift["source_resblock_kernel_sizes"],
            source_resblock_dilation_sizes=config.hift["source_resblock_dilation_sizes"],
            lrelu_slope=config.hift["lrelu_slope"],
            audio_limit=config.hift["audio_limit"],
            conv_pre_look_right=config.hift["conv_pre_look_right"],
            f0_predictor=f0_predictor,
        )
        # Run hift in float32 to avoid dtype mismatches in internal ops
        self.hift = self.hift.float()

        # Streaming/chunking parameters
        self.token_overlap_len = 20
        self.mel_overlap_len = int(self.token_overlap_len / self.flow_model.input_frame_rate * 22050 / 256)
        self.mel_window = np.hamming(2 * self.mel_overlap_len)
        self.mel_cache_len = 20
        self.source_cache_len = int(self.mel_cache_len * 256)
        self.speech_window = np.hamming(2 * self.source_cache_len)
        self._flow_batch_metadata_cache: dict[tuple[object, ...], object] = {}
        self._flow_batch_stats = {
            "groups": 0,
            "requests": 0,
            "metadata_hits": 0,
            "metadata_misses": 0,
            "fallbacks": 0,
        }

    @property
    def input_frame_rate(self) -> int:
        """Input frame rate from flow model."""
        return self.flow_model.input_frame_rate

    @property
    def token_mel_ratio(self) -> int:
        """Token to mel ratio."""
        return self.flow_model.token_mel_ratio

    @property
    def output_size(self) -> int:
        """Output mel dimension."""
        return self.flow_model.output_size

    @property
    def input_embedding(self) -> nn.Embedding:
        """Token embedding layer."""
        return self.flow_model.input_embedding

    @property
    def pre_lookahead_layer(self) -> nn.Module:
        """Pre-lookahead layer."""
        return self.flow_model.pre_lookahead_layer

    @property
    def decoder(self) -> nn.Module:
        """Flow matching decoder."""
        return self.flow_model.decoder

    @property
    def spk_embed_affine_layer(self) -> nn.Linear:
        """Speaker embedding affine layer."""
        return self.flow_model.spk_embed_affine_layer

    @torch.inference_mode()
    def _forward_mel(
        self,
        token: torch.Tensor,
        prompt_token: torch.Tensor,
        prompt_feat: torch.Tensor,
        embedding: torch.Tensor,
        n_timesteps: int = 10,
        token_offset_tokens: int = 0,
        streaming: bool = True,
        finalize: bool = False,
    ) -> torch.Tensor:
        """Generate mel features via the upstream flow-model inference path."""
        flow_weight = next(self.flow_model.parameters())
        device = flow_weight.device
        dtype = flow_weight.dtype

        token = token.to(device=device, dtype=torch.int32)
        prompt_token = prompt_token.to(device=device, dtype=torch.int32)
        prompt_feat = prompt_feat.to(device=device, dtype=dtype)
        embedding = embedding.to(device=device, dtype=dtype)
        token_len = torch.tensor([token.shape[1]], device=device, dtype=torch.int32)
        prompt_token_len = torch.tensor([prompt_token.shape[1]], device=device, dtype=torch.int32)
        prompt_feat_len = torch.tensor([prompt_feat.shape[1]], device=device, dtype=torch.int32)

        feat, _ = self.flow_model.inference(
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
        )

        trim_mel = max(0, int(token_offset_tokens)) * int(self.token_mel_ratio)
        if trim_mel > 0:
            feat = feat[:, :, trim_mel:]

        return feat

    @torch.inference_mode()
    def _forward_mel_batch(
        self,
        requests: list[dict[str, torch.Tensor | int | bool]],
        *,
        n_timesteps: int = 10,
    ) -> list[torch.Tensor]:
        """Run the flow model once for a padded group and split per request.

        This deliberately keeps dense padded tensors in PR2. The returned
        metadata/cache state gives PR3 a stable place to attach varlen Attention
        indices and workspace without changing the public code2wav contract.
        """
        if not requests:
            return []

        flow_weight = next(self.flow_model.parameters())
        device = flow_weight.device
        dtype = flow_weight.dtype
        batch = len(requests)

        token_lens = [int(req["token"].shape[1]) for req in requests]  # type: ignore[index]
        prompt_token_lens = [int(req["prompt_token"].shape[1]) for req in requests]  # type: ignore[index]
        prompt_feat_lens = [int(req["prompt_feat"].shape[1]) for req in requests]  # type: ignore[index]
        max_token_len = max(token_lens)
        max_prompt_token_len = max(prompt_token_lens)
        max_prompt_feat_len = max(prompt_feat_lens)

        token = torch.zeros((batch, max_token_len), device=device, dtype=torch.int32)
        prompt_token = torch.zeros((batch, max_prompt_token_len), device=device, dtype=torch.int32)
        feat_dim = int(requests[0]["prompt_feat"].shape[-1])  # type: ignore[index]
        prompt_feat = torch.zeros((batch, max_prompt_feat_len, feat_dim), device=device, dtype=dtype)
        embedding_rows: list[torch.Tensor] = []
        for row, req in enumerate(requests):
            token_row = req["token"].to(device=device, dtype=torch.int32)  # type: ignore[union-attr]
            prompt_token_row = req["prompt_token"].to(device=device, dtype=torch.int32)  # type: ignore[union-attr]
            prompt_feat_row = req["prompt_feat"].to(device=device, dtype=dtype)  # type: ignore[union-attr]
            embedding_row = req["embedding"].to(device=device, dtype=dtype)  # type: ignore[union-attr]
            token[row, : token_lens[row]] = token_row[0, : token_lens[row]]
            prompt_token[row, : prompt_token_lens[row]] = prompt_token_row[0, : prompt_token_lens[row]]
            prompt_feat[row, : prompt_feat_lens[row]] = prompt_feat_row[0, : prompt_feat_lens[row]]
            embedding_rows.append(embedding_row[:1])

        token_len = torch.tensor(token_lens, device=device, dtype=torch.int32)
        prompt_token_len = torch.tensor(prompt_token_lens, device=device, dtype=torch.int32)
        prompt_feat_len = torch.tensor(prompt_feat_lens, device=device, dtype=torch.int32)
        embedding = torch.cat(embedding_rows, dim=0)

        cache_key = (
            batch,
            max_token_len,
            max_prompt_token_len,
            max_prompt_feat_len,
            dtype,
            device,
            bool(requests[0].get("streaming", True)),
            bool(requests[0].get("finalize", False)),
        )
        if cache_key in self._flow_batch_metadata_cache:
            self._flow_batch_stats["metadata_hits"] += 1
        else:
            self._flow_batch_stats["metadata_misses"] += 1
            self._flow_batch_metadata_cache[cache_key] = None

        with record_function(f"cosyvoice3_stage1:flow_batch_b{batch}_tok{max_token_len}"):
            feat, metadata = self.flow_model.inference(
                token=token,
                token_len=token_len,
                prompt_token=prompt_token,
                prompt_token_len=prompt_token_len,
                prompt_feat=prompt_feat,
                prompt_feat_len=prompt_feat_len,
                embedding=embedding,
                streaming=bool(requests[0].get("streaming", True)),
                finalize=bool(requests[0].get("finalize", False)),
                n_timesteps=n_timesteps,
            )
        self._flow_batch_metadata_cache[cache_key] = metadata
        self._flow_batch_stats["groups"] += 1
        self._flow_batch_stats["requests"] += batch

        generated_mel_lens = metadata["generated_mel_lens"] if metadata is not None else token_len * self.token_mel_ratio
        outputs: list[torch.Tensor] = []
        for row in range(batch):
            valid_mel = int(generated_mel_lens[row].item())
            trim_mel = max(0, int(requests[row].get("token_offset_tokens", 0))) * int(self.token_mel_ratio)
            start = min(trim_mel, valid_mel)
            outputs.append(feat[row : row + 1, :, start:valid_mel].contiguous())
        return outputs

    @torch.inference_mode()
    def _audio_from_mel(self, feat: torch.Tensor) -> torch.Tensor:
        """Run final HiFT decode for an already computed mel chunk."""
        hift_weight = self.hift.m_source.l_linear.weight
        tts_mel = feat.to(device=hift_weight.device, dtype=hift_weight.dtype)

        if tts_mel.shape[-1] == 0:
            return torch.zeros(
                (tts_mel.shape[0], 1, 0),
                device=tts_mel.device,
                dtype=tts_mel.dtype,
            )
        tts_speech, _ = self.hift.inference(speech_feat=tts_mel, finalize=True)
        return tts_speech

    @torch.inference_mode()
    def _streaming_audio_from_mel(
        self,
        feat: torch.Tensor,
        *,
        cache_state: dict[str, torch.Tensor] | None = None,
        finalize: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None]:
        """Run streaming HiFT decode for an already computed mel chunk."""
        hift_weight = self.hift.m_source.l_linear.weight
        chunk_mel = feat.to(device=hift_weight.device, dtype=hift_weight.dtype)

        cached_mel = None if not cache_state else cache_state.get("mel")
        speech_offset_obj = None if not cache_state else cache_state.get("speech_offset")
        try:
            speech_offset = int(speech_offset_obj) if speech_offset_obj is not None else 0
        except (TypeError, ValueError):
            speech_offset = 0

        if isinstance(cached_mel, torch.Tensor) and cached_mel.numel() > 0:
            cached_mel = cached_mel.to(device=chunk_mel.device, dtype=chunk_mel.dtype)
            tts_mel = torch.cat([cached_mel, chunk_mel], dim=-1) if chunk_mel.numel() > 0 else cached_mel
        else:
            tts_mel = chunk_mel

        if tts_mel.shape[-1] == 0:
            tts_speech = torch.zeros((chunk_mel.shape[0], 1, 0), device=chunk_mel.device, dtype=chunk_mel.dtype)
        else:
            tts_speech, _ = self.hift.inference(speech_feat=tts_mel, finalize=finalize)

        tts_speech = tts_speech.reshape(tts_speech.shape[0], -1)
        speech_offset = max(0, min(speech_offset, int(tts_speech.shape[-1])))
        emitted_speech = tts_speech[:, speech_offset:]

        if finalize:
            return emitted_speech.reshape(emitted_speech.shape[0], 1, -1), None

        new_state = {
            "mel": tts_mel.detach().cpu().contiguous(),
            "speech_offset": int(tts_speech.shape[-1]),
        }
        return emitted_speech.reshape(emitted_speech.shape[0], 1, -1), new_state

    @torch.inference_mode()
    def forward_streaming(
        self,
        token: torch.Tensor,
        prompt_token: torch.Tensor,
        prompt_feat: torch.Tensor,
        embedding: torch.Tensor,
        *,
        cache_state: dict[str, torch.Tensor] | None = None,
        n_timesteps: int = 10,
        token_offset_tokens: int = 0,
        finalize: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None]:
        """Decode streaming audio using cumulative mel + emitted-speech offset.

        This mirrors upstream CosyVoice3 streaming semantics more closely than
        waveform-domain overlap-add: keep a cumulative mel history per request,
        re-run causal HiFT on the history, and emit only the newly grown speech
        suffix. That preserves causal look-right handling without double
        trimming or duplicated overlap at chunk boundaries.
        """
        feat = self._forward_mel(
            token=token,
            prompt_token=prompt_token,
            prompt_feat=prompt_feat,
            embedding=embedding,
            n_timesteps=n_timesteps,
            token_offset_tokens=token_offset_tokens,
            streaming=True,
            finalize=finalize,
        )
        return self._streaming_audio_from_mel(feat, cache_state=cache_state, finalize=finalize)

    @torch.inference_mode()
    def forward(
        self,
        token: torch.Tensor,
        prompt_token: torch.Tensor,
        prompt_feat: torch.Tensor,
        embedding: torch.Tensor,
        n_timesteps: int = 10,
        token_offset_tokens: int = 0,
    ) -> torch.Tensor:
        """Generate audio waveform from speech tokens."""
        feat = self._forward_mel(
            token=token,
            prompt_token=prompt_token,
            prompt_feat=prompt_feat,
            embedding=embedding,
            n_timesteps=n_timesteps,
            token_offset_tokens=token_offset_tokens,
            streaming=False,
            finalize=True,
        )

        return self._audio_from_mel(feat)

    def load_weights(self, model_dir: str, device: torch.device) -> None:
        """Load flow.pt and hift.pt weights.

        Args:
            model_dir: Model directory containing flow.pt and hift.pt
            device: Device to load weights to
        """
        # Load flow weights
        flow_path = os.path.join(model_dir, "flow.pt")
        self.flow_model.load_state_dict(torch.load(flow_path, map_location=device), strict=True)
        flow_dtype = _cosyvoice3_flow_dtype()
        self.flow_model.to(device=device, dtype=flow_dtype).eval()
        logger.info("Loaded flow weights from %s (dtype=%s)", flow_path, flow_dtype)

        # Load hift weights
        hift_path = os.path.join(model_dir, "hift.pt")
        hift_state_dict = {
            k.replace("generator.", ""): v for k, v in torch.load(hift_path, map_location=device).items()
        }
        self.hift.load_state_dict(hift_state_dict, strict=True)
        self.hift.to(device=device, dtype=torch.float32).eval()
        logger.info(f"Loaded hift weights from {hift_path}")
