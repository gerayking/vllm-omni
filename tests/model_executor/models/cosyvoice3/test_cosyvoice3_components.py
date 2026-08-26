# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Unit tests for CosyVoice3 components."""

import types
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from tests.helpers.mark import hardware_test
from vllm_omni.model_executor.models.cosyvoice3.code2wav_core.hifigan import (
    CausalHiFTGenerator,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture
def causal_hift():
    return CausalHiFTGenerator(
        base_channels=32,
        upsample_rates=[2, 2],
        upsample_kernel_sizes=[4, 4],
        source_resblock_kernel_sizes=[3, 3],
        source_resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5]],
    )


def test_causal_hift_moves_stft_window_with_model(causal_hift):
    assert causal_hift.get_buffer("stft_window") is causal_hift.stft_window
    assert "stft_window" not in causal_hift.state_dict()
    causal_hift.to(dtype=torch.float64)
    assert causal_hift.stft_window.dtype == torch.float64


def test_causal_hift_stft_moves_window_to_input_device(causal_hift):
    waveform = torch.empty((1, 64), device="meta")

    real, imag = causal_hift._stft(waveform)

    assert real.device == waveform.device
    assert imag.device == waveform.device
    assert causal_hift.stft_window.device == waveform.device


class TestPreLookaheadLayer:
    """Tests for PreLookaheadLayer."""

    @pytest.fixture
    def layer(self):
        from vllm_omni.model_executor.models.cosyvoice3.code2wav_core.layers import PreLookaheadLayer

        return PreLookaheadLayer(in_channels=512, channels=512, pre_lookahead_len=3)

    @pytest.mark.core_model
    @pytest.mark.cpu
    def test_forward_shape(self, layer):
        """Test that output shape matches input shape."""
        batch, seq_len, channels = 2, 10, 512
        x = torch.randn(batch, seq_len, channels)

        out = layer(x)

        assert out.shape == x.shape

    @pytest.mark.core_model
    @pytest.mark.cpu
    def test_forward_with_context(self, layer):
        """Test forward with context for streaming."""
        batch, seq_len, channels = 1, 10, 512
        x = torch.randn(batch, seq_len, channels)
        context = torch.randn(batch, 3, channels)  # pre_lookahead_len=3

        layer.eval()
        out = layer(x, context=context)

        assert out.shape == x.shape

    @pytest.mark.core_model
    @pytest.mark.cpu
    def test_residual_connection(self, layer):
        """Test that residual connection is applied."""
        batch, seq_len, channels = 1, 5, 512
        x = torch.zeros(batch, seq_len, channels)

        # With zero input, output should also be close to zero due to residual
        out = layer(x)

        # Output should be close to input (residual) plus conv output
        assert out.shape == x.shape


class TestDiTAttention:
    """Tests for DiTAttention with diffusion backend."""

    @pytest.fixture
    def attention(self):
        from vllm_omni.diffusion.models.cosyvoice3_audio.cosyvoice3_dit import DiTAttention

        return DiTAttention(dim=512, heads=8, dim_head=64, dropout=0.0)

    @pytest.mark.core_model
    @hardware_test(res={"cuda": "L4"}, num_cards=1)
    def test_forward_shape(self, attention):
        """Test attention output shape."""
        batch, seq_len, dim = 2, 16, 512
        x = torch.randn(batch, seq_len, dim)

        out = attention(x)

        assert out.shape == x.shape

    @pytest.mark.core_model
    @hardware_test(res={"cuda": "L4"}, num_cards=1)
    def test_forward_with_mask(self, attention):
        """Test attention with mask."""
        batch, seq_len, dim = 2, 16, 512
        x = torch.randn(batch, seq_len, dim)
        mask = torch.ones(batch, seq_len, dtype=torch.bool)
        mask[:, -3:] = False  # Mask last 3 positions

        out = attention(x, mask=mask)

        assert out.shape == x.shape
        # Masked positions should be zero
        assert torch.allclose(out[:, -3:], torch.zeros_like(out[:, -3:]))

    @pytest.mark.core_model
    @hardware_test(res={"cuda": "L4"}, num_cards=1)
    def test_qkv_projections(self, attention):
        """Test that Q/K/V projections exist and have correct dimensions."""
        assert hasattr(attention, "to_q")
        assert hasattr(attention, "to_k")
        assert hasattr(attention, "to_v")
        assert attention.to_q.out_features == 512  # heads * dim_head
        assert attention.to_k.out_features == 512
        assert attention.to_v.out_features == 512

    @pytest.mark.core_model
    @pytest.mark.cpu
    def test_varlen_attention_passes_2d_mask_and_casts_qkv(self, monkeypatch):
        """Varlen path passes a 2D validity mask through AttentionMetadata."""
        from vllm_omni.diffusion.models.cosyvoice3_audio.cosyvoice3_dit import DiTAttention

        class CaptureAttention(nn.Module):
            def __init__(self):
                super().__init__()
                self.last_metadata = None
                self.last_dtype = None

            def forward(self, q, k, v, attn_metadata=None):
                self.last_metadata = attn_metadata
                self.last_dtype = q.dtype
                return torch.zeros_like(q)

        monkeypatch.setenv("COSYVOICE3_VARLEN_ATTENTION", "1")
        monkeypatch.setenv("COSYVOICE3_VARLEN_ATTENTION_DTYPE", "bf16")
        attention = DiTAttention(dim=16, heads=2, dim_head=8, dropout=0.0)
        capture = CaptureAttention()
        attention.attn = capture

        x = torch.randn(2, 5, 16)
        mask = torch.tensor([[True, True, True, False, False], [True, True, True, True, False]])

        out = attention(x, mask=mask)

        assert out.shape == x.shape
        assert capture.last_metadata is not None
        assert torch.equal(capture.last_metadata.attn_mask, mask)
        assert capture.last_dtype == torch.bfloat16
        assert out.dtype == x.dtype
        assert torch.allclose(out[~mask], torch.zeros_like(out[~mask]))


class TestDiTBlock:
    """Tests for DiTBlock."""

    @pytest.fixture
    def block(self):
        from vllm_omni.diffusion.models.cosyvoice3_audio.cosyvoice3_dit import DiTBlock

        return DiTBlock(dim=512, heads=8, dim_head=64, ff_mult=4, dropout=0.0)

    @pytest.mark.core_model
    @hardware_test(res={"cuda": "L4"}, num_cards=1)
    def test_forward_shape(self, block):
        """Test block output shape."""
        batch, seq_len, dim = 2, 16, 512
        x = torch.randn(batch, seq_len, dim)
        t = torch.randn(batch, dim)  # Timestep embedding

        out = block(x, t)

        assert out.shape == x.shape

    @pytest.mark.core_model
    @hardware_test(res={"cuda": "L4"}, num_cards=1)
    def test_adalayernorm_modulation(self, block):
        """Test that AdaLayerNorm modulates based on timestep."""
        batch, seq_len, dim = 1, 8, 512
        x = torch.randn(batch, seq_len, dim)
        t1 = torch.zeros(batch, dim)
        t2 = torch.ones(batch, dim)

        out1 = block(x, t1)
        out2 = block(x, t2)

        # Different timesteps should produce different outputs
        assert not torch.allclose(out1, out2)


class TestDiT:
    """Tests for the full DiT model."""

    @pytest.fixture
    def dit(self):
        from vllm_omni.diffusion.models.cosyvoice3_audio.cosyvoice3_dit import DiT

        return DiT(
            dim=256,
            depth=2,
            heads=4,
            dim_head=64,
            dropout=0.0,
            ff_mult=2,
            mel_dim=80,
            mu_dim=80,
            spk_dim=80,
            long_skip_connection=True,
        )

    @pytest.mark.core_model
    @hardware_test(res={"cuda": "L4"}, num_cards=1)
    def test_forward_shape(self, dit):
        """Test DiT forward output shape."""
        batch, mel_dim, seq_len = 1, 80, 32
        x = torch.randn(batch, mel_dim, seq_len)
        mask = torch.ones(batch, 1, seq_len)
        mu = torch.randn(batch, mel_dim, seq_len)
        t = torch.tensor([0.5])
        spks = torch.randn(batch, 80)
        cond = torch.randn(batch, mel_dim, seq_len)

        out = dit(x, mask, mu, t, spks=spks, cond=cond)

        assert out.shape == (batch, mel_dim, seq_len)

    @pytest.mark.core_model
    @hardware_test(res={"cuda": "L4"}, num_cards=1)
    def test_timestep_embedding(self, dit):
        """Test that different timesteps produce different outputs."""
        batch, mel_dim, seq_len = 1, 80, 16
        x = torch.randn(batch, mel_dim, seq_len)
        mask = torch.ones(batch, 1, seq_len)
        mu = torch.randn(batch, mel_dim, seq_len)
        spks = torch.randn(batch, 80)
        cond = torch.randn(batch, mel_dim, seq_len)

        out1 = dit(x, mask, mu, torch.tensor([0.0]), spks=spks, cond=cond)
        out2 = dit(x, mask, mu, torch.tensor([1.0]), spks=spks, cond=cond)

        assert not torch.allclose(out1, out2)


class TestCFM:
    """Tests for Conditional Flow Matching classes."""

    @pytest.fixture
    def dummy_estimator(self):
        """Create a dummy estimator for testing."""

        class DummyEstimator(nn.Module):
            def __init__(self, mel_dim=80):
                super().__init__()
                self.mel_dim = mel_dim

            def forward(self, x, mask, mu, t, spks=None, cond=None):
                return torch.zeros_like(x)

        return DummyEstimator()

    @pytest.mark.core_model
    @pytest.mark.cpu
    def test_causal_conditional_cfm_forward(self, dummy_estimator):
        """Test CausalConditionalCFM forward pass."""
        from omegaconf import DictConfig

        from vllm_omni.model_executor.models.cosyvoice3.code2wav_core.cfm import CausalConditionalCFM

        cfm_params = DictConfig(
            {
                "sigma_min": 1e-6,
                "solver": "euler",
                "t_scheduler": "cosine",
                "training_cfg_rate": 0.2,
                "inference_cfg_rate": 0.7,
            }
        )

        cfm = CausalConditionalCFM(
            in_channels=80,
            cfm_params=cfm_params,
            n_spks=1,
            spk_emb_dim=80,
            estimator=dummy_estimator,
        )

        batch, mel_dim, seq_len = 1, 80, 32
        mu = torch.randn(batch, mel_dim, seq_len)
        mask = torch.ones(batch, 1, seq_len)
        spks = torch.randn(batch, 80)
        cond = torch.randn(batch, mel_dim, seq_len)

        out, _ = cfm(mu, mask, n_timesteps=2, spks=spks, cond=cond)

        assert out.shape == mu.shape

    @pytest.mark.core_model
    @pytest.mark.cpu
    def test_causal_conditional_cfm_batches_cfg_estimator(self):
        """Batched flow calls should invoke the CFG estimator with 2B rows."""
        from omegaconf import DictConfig

        from vllm_omni.model_executor.models.cosyvoice3.code2wav_core.cfm import CausalConditionalCFM

        class RecordingEstimator(nn.Module):
            def __init__(self):
                super().__init__()
                self.shapes: list[tuple[int, ...]] = []

            def forward(self, x, mask, mu, t, spks=None, cond=None):
                self.shapes.append(tuple(x.shape))
                return torch.zeros_like(x)

        estimator = RecordingEstimator()
        cfm_params = DictConfig(
            {
                "sigma_min": 1e-6,
                "solver": "euler",
                "t_scheduler": "cosine",
                "training_cfg_rate": 0.2,
                "inference_cfg_rate": 0.7,
            }
        )
        cfm = CausalConditionalCFM(
            in_channels=80,
            cfm_params=cfm_params,
            n_spks=1,
            spk_emb_dim=80,
            estimator=estimator,
        )

        batch, mel_dim, seq_len = 3, 80, 16
        mu = torch.randn(batch, mel_dim, seq_len)
        mask = torch.ones(batch, 1, seq_len)
        spks = torch.randn(batch, 80)
        cond = torch.randn(batch, mel_dim, seq_len)

        out, _ = cfm(mu, mask, n_timesteps=2, spks=spks, cond=cond)

        assert out.shape == mu.shape
        assert estimator.shapes == [(2 * batch, mel_dim, seq_len)] * 2


class TestSDPAFallback:
    """Test SDPA fallback for float32 inputs."""

    @pytest.mark.core_model
    @hardware_test(res={"cuda": "L4"}, num_cards=1)
    def test_float32_uses_sdpa(self):
        """Test that float32 inputs use SDPA fallback."""
        from vllm_omni.diffusion.attention.layer import Attention

        attn = Attention(
            num_heads=8,
            head_size=64,
            causal=False,
            softmax_scale=1.0 / 8.0,
        )

        batch, seq_len, heads, dim = 1, 16, 8, 64
        q = torch.randn(batch, seq_len, heads, dim, dtype=torch.float32)
        k = torch.randn(batch, seq_len, heads, dim, dtype=torch.float32)
        v = torch.randn(batch, seq_len, heads, dim, dtype=torch.float32)

        # Should not raise error - SDPA fallback handles float32
        out = attn(q, k, v)

        assert out.shape == (batch, seq_len, heads, dim)
        assert out.dtype == torch.float32


def test_code2wav_forward_finalizes_hift_tail():
    from vllm_omni.model_executor.models.cosyvoice3.cosyvoice3_code2wav import CosyVoice3Code2Wav

    class DummyHiFT(nn.Module):
        def __init__(self):
            super().__init__()
            self.m_source = SimpleNamespace(l_linear=SimpleNamespace(weight=torch.ones(1, dtype=torch.float32)))
            self.finalize_calls: list[bool] = []

        def inference(self, speech_feat, finalize=True):
            self.finalize_calls.append(bool(finalize))
            return torch.zeros((speech_feat.shape[0], 1, speech_feat.shape[-1]), dtype=speech_feat.dtype), None

    model = object.__new__(CosyVoice3Code2Wav)
    nn.Module.__init__(model)
    model.hift = DummyHiFT()
    forward_mel_calls = []

    def fake_forward_mel(**kwargs):
        forward_mel_calls.append(kwargs)
        return torch.ones((1, 80, 8), dtype=torch.float32)

    model._forward_mel = fake_forward_mel

    out = model.forward(
        token=torch.tensor([[1, 2, 3]], dtype=torch.int32),
        prompt_token=torch.tensor([[4, 5]], dtype=torch.int32),
        prompt_feat=torch.ones((1, 4, 80), dtype=torch.float32),
        embedding=torch.ones((1, 192), dtype=torch.float32),
    )

    assert out.shape == (1, 1, 8)
    assert model.hift.finalize_calls == [True]
    assert forward_mel_calls[0]["token_offset_tokens"] == 0


def test_code2wav_streaming_batch_pads_codec_tokens_and_preserves_lengths():
    from vllm_omni.model_executor.models.cosyvoice3.cosyvoice3_code2wav import CosyVoice3Code2Wav

    model = object.__new__(CosyVoice3Code2Wav)
    model.flow_model = SimpleNamespace(pre_lookahead_len=1, token_mel_ratio=2)
    forward_mel_calls = []

    def fake_forward_mel(self, **kwargs):
        forward_mel_calls.append(kwargs)
        return torch.arange(2 * 80 * 8, dtype=torch.float32).reshape(2, 80, 8)

    def fake_stream_hift(self, feat, *, cache_state=None, finalize=False):
        return feat, None

    model._forward_mel = types.MethodType(fake_forward_mel, model)
    model._stream_hift_from_feat = types.MethodType(fake_stream_hift, model)
    items = [
        {
            "token": torch.ones(1, 3, dtype=torch.int32),
            "prompt_token": torch.ones(1, 4, dtype=torch.int32),
            "prompt_feat": torch.ones(1, 8, 80),
            "embedding": torch.ones(1, 192),
            "finalize": False,
        },
        {
            "token": torch.ones(1, 5, dtype=torch.int32),
            "prompt_token": torch.ones(1, 4, dtype=torch.int32),
            "prompt_feat": torch.ones(1, 8, 80),
            "embedding": torch.ones(1, 192),
            "finalize": False,
        },
    ]

    results = model.forward_streaming_batch(items)

    assert len(forward_mel_calls) == 1
    call = forward_mel_calls[0]
    assert call["token"].shape == (2, 5)
    assert torch.equal(call["token_lens"], torch.tensor([3, 5], dtype=torch.int32))
    assert results[0][0].shape[-1] == 4
    assert results[1][0].shape[-1] == 8


def test_cfm_cuda_graph_env_defaults_disabled(monkeypatch):
    from vllm_omni.model_executor.models.cosyvoice3.code2wav_core.cfm import (
        _cosyvoice3_cfm_cuda_graph_batch_buckets,
        _cosyvoice3_cfm_cuda_graph_enabled,
        _cosyvoice3_cfm_cuda_graph_max_graphs,
        _cosyvoice3_cfm_cuda_graph_profile_shapes,
        _cosyvoice3_cfm_cuda_graph_timestep_buckets,
    )

    monkeypatch.delenv("COSYVOICE3_CFM_CUDA_GRAPH", raising=False)
    monkeypatch.delenv("COSYVOICE3_CFM_CUDAGRAPH", raising=False)
    monkeypatch.delenv("COSYVOICE3_CFM_CUDA_GRAPH_MAX_GRAPHS", raising=False)
    monkeypatch.delenv("COSYVOICE3_CFM_CUDA_GRAPH_TIMESTEP_BUCKETS", raising=False)
    monkeypatch.delenv("COSYVOICE3_CFM_CUDA_GRAPH_BATCH_BUCKETS", raising=False)
    monkeypatch.delenv("COSYVOICE3_CFM_CUDA_GRAPH_PROFILE_SHAPES", raising=False)

    assert _cosyvoice3_cfm_cuda_graph_enabled() is False
    assert _cosyvoice3_cfm_cuda_graph_max_graphs() == 12
    assert _cosyvoice3_cfm_cuda_graph_timestep_buckets() == (330, 450, 570, 690, 780, 900)
    assert _cosyvoice3_cfm_cuda_graph_batch_buckets() == (1, 2)
    assert _cosyvoice3_cfm_cuda_graph_profile_shapes() is False


def test_cfm_cuda_graph_env_accepts_enabled(monkeypatch):
    from vllm_omni.model_executor.models.cosyvoice3.code2wav_core.cfm import (
        _cosyvoice3_cfm_cuda_graph_batch_buckets,
        _cosyvoice3_cfm_cuda_graph_enabled,
        _cosyvoice3_cfm_cuda_graph_max_graphs,
        _cosyvoice3_cfm_cuda_graph_profile_shapes,
        _cosyvoice3_cfm_cuda_graph_timestep_buckets,
    )

    monkeypatch.setenv("COSYVOICE3_CFM_CUDA_GRAPH", "1")
    monkeypatch.setenv("COSYVOICE3_CFM_CUDA_GRAPH_MAX_GRAPHS", "2")
    monkeypatch.setenv("COSYVOICE3_CFM_CUDA_GRAPH_TIMESTEP_BUCKETS", "16, 8,16,32")
    monkeypatch.setenv("COSYVOICE3_CFM_CUDA_GRAPH_BATCH_BUCKETS", "1,2,4")
    monkeypatch.setenv("COSYVOICE3_CFM_CUDA_GRAPH_PROFILE_SHAPES", "true")

    assert _cosyvoice3_cfm_cuda_graph_enabled() is True
    assert _cosyvoice3_cfm_cuda_graph_max_graphs() == 2
    assert _cosyvoice3_cfm_cuda_graph_timestep_buckets() == (8, 16, 32)
    assert _cosyvoice3_cfm_cuda_graph_batch_buckets() == (1, 2, 4)
    assert _cosyvoice3_cfm_cuda_graph_profile_shapes() is True

    monkeypatch.setenv("COSYVOICE3_CFM_CUDA_GRAPH_TIMESTEP_BUCKETS", "exact")
    monkeypatch.setenv("COSYVOICE3_CFM_CUDA_GRAPH_BATCH_BUCKETS", "exact")
    assert _cosyvoice3_cfm_cuda_graph_timestep_buckets() == ()
    assert _cosyvoice3_cfm_cuda_graph_batch_buckets() == ()

    monkeypatch.delenv("COSYVOICE3_CFM_CUDA_GRAPH", raising=False)
    monkeypatch.setenv("COSYVOICE3_CFM_CUDAGRAPH", "1")
    assert _cosyvoice3_cfm_cuda_graph_enabled() is True


def test_conditional_cfm_cuda_graph_cpu_fallback_updates_stats(monkeypatch):
    from vllm_omni.model_executor.models.cosyvoice3.code2wav_core.cfm import (
        ConditionalCFM,
    )

    class ZeroEstimator(nn.Module):
        def forward(self, x, mask, mu, t, spks, cond):
            return torch.zeros_like(x)

    monkeypatch.setenv("COSYVOICE3_CFM_CUDA_GRAPH", "1")
    cfm = ConditionalCFM(
        in_channels=2,
        cfm_params=SimpleNamespace(
            solver="euler",
            t_scheduler="linear",
            training_cfg_rate=0.0,
            inference_cfg_rate=0.0,
        ),
        n_spks=1,
        spk_emb_dim=2,
        estimator=ZeroEstimator(),
    )

    x = torch.ones(1, 2, 3)
    t_span = torch.linspace(0, 1, 3)
    mask = torch.ones(1, 1, 3)
    mu = torch.zeros_like(x)
    spks = torch.ones(1, 2)
    cond = torch.zeros_like(x)

    out = cfm.solve_euler(x, t_span=t_span, mu=mu, mask=mask, spks=spks, cond=cond)

    assert torch.equal(out, x)
    stats = cfm.get_cuda_graph_stats()
    assert stats["calls"] == 1
    assert stats["total_euler_calls"] == 1
    assert stats["fallbacks"] == 1
    assert stats["replays"] == 0
    assert stats["replay_hit_rate"] == 0.0


def test_cfm_cuda_graph_cache_full_falls_back_without_evicting(monkeypatch):
    from vllm_omni.model_executor.models.cosyvoice3.code2wav_core.cfm import (
        CUDAGraphCFMEulerRunner,
        _CFMEulerGraphEntry,
    )

    class FakeGraph:
        def replay(self):
            pass

    class FakeCFM:
        t_scheduler = "linear"
        inference_cfg_rate = 0.0
        estimator = nn.Identity()

    def make_entry(x, t_span, mu, mask, spks, cond):
        return _CFMEulerGraphEntry(
            graph=FakeGraph(),
            static_x=x.clone(),
            static_t_span=t_span.clone(),
            static_mu=mu.clone(),
            static_mask=mask.clone(),
            static_spks=spks.clone(),
            static_cond=cond.clone(),
            static_x_in=torch.empty(0),
            static_mask_in=torch.empty(0),
            static_mu_in=torch.empty(0),
            static_t_in=torch.empty(0),
            static_spks_in=torch.empty(0),
            static_cond_in=torch.empty(0),
            static_out=x + 1.0,
        )

    runner = CUDAGraphCFMEulerRunner(enabled=True, max_graphs=1)
    monkeypatch.setattr(
        runner,
        "_ineligible_reason",
        lambda cfm, *, x, mu, mask, spks, cond: None,
    )

    def capture(cfm, *, x, t_span, mu, mask, spks, cond, bucket_batch, bucket_timesteps):
        assert bucket_batch == x.size(0)
        assert bucket_timesteps == x.size(2)
        runner._stats["captures"] += 1
        return make_entry(x, t_span, mu, mask, spks, cond)

    monkeypatch.setattr(runner, "_capture", capture)

    t_span = torch.linspace(0, 1, 3)
    spks = torch.ones(1, 2)

    def replay_for(length: int):
        x = torch.zeros(1, 2, length)
        return runner.try_replay(
            FakeCFM(),
            x=x,
            t_span=t_span,
            mu=torch.zeros_like(x),
            mask=torch.ones(1, 1, length),
            spks=spks,
            cond=torch.zeros_like(x),
        )

    assert torch.equal(replay_for(4), torch.ones(1, 2, 4))
    assert replay_for(5) is None
    assert torch.equal(replay_for(4), torch.ones(1, 2, 4))

    stats = runner.stats()
    assert stats["captures"] == 1
    assert stats["shape_misses"] == 2
    assert stats["cache_full_fallbacks"] == 1
    assert stats["fallbacks"] == 1
    assert stats["replays"] == 2
    assert stats["cache_hits"] == 1
    assert stats["unique_graphs"] == 1


def test_cfm_cuda_graph_bucket_reuses_padded_static_buffers(monkeypatch):
    from vllm_omni.model_executor.models.cosyvoice3.code2wav_core.cfm import (
        CUDAGraphCFMEulerRunner,
        _CFMEulerGraphEntry,
    )

    class FakeCFM:
        t_scheduler = "linear"
        inference_cfg_rate = 0.0
        estimator = nn.Identity()

    runner = CUDAGraphCFMEulerRunner(
        enabled=True,
        max_graphs=1,
        timestep_buckets=(8,),
        batch_buckets=(2,),
    )
    monkeypatch.setattr(runner, "_ineligible_reason", lambda cfm, *, x, mu, mask, spks, cond: None)

    def capture(cfm, *, x, t_span, mu, mask, spks, cond, bucket_batch, bucket_timesteps):
        entry = _CFMEulerGraphEntry(
            graph=None,
            static_x=torch.zeros(bucket_batch, x.size(1), bucket_timesteps),
            static_t_span=t_span.clone(),
            static_mu=torch.zeros(bucket_batch, mu.size(1), bucket_timesteps),
            static_mask=torch.zeros(bucket_batch, mask.size(1), bucket_timesteps),
            static_spks=torch.zeros(bucket_batch, spks.size(1)),
            static_cond=torch.zeros(bucket_batch, cond.size(1), bucket_timesteps),
            static_x_in=torch.empty(0),
            static_mask_in=torch.empty(0),
            static_mu_in=torch.empty(0),
            static_t_in=torch.empty(0),
            static_spks_in=torch.empty(0),
            static_cond_in=torch.empty(0),
            static_out=torch.zeros(bucket_batch, x.size(1), bucket_timesteps),
        )

        class FakeGraph:
            def replay(self):
                entry.static_out.copy_(entry.static_x + 1)

        entry.graph = FakeGraph()
        runner._copy_static_inputs(
            entry,
            x=x,
            t_span=t_span,
            mu=mu,
            mask=mask,
            spks=spks,
            cond=cond,
        )
        runner._stats["captures"] += 1
        return entry

    monkeypatch.setattr(runner, "_capture", capture)
    t_span = torch.linspace(0, 1, 3)

    def replay(batch: int, length: int):
        x = torch.arange(batch * 2 * length, dtype=torch.float32).reshape(batch, 2, length)
        return x, runner.try_replay(
            FakeCFM(),
            x=x,
            t_span=t_span,
            mu=torch.zeros_like(x),
            mask=torch.ones(batch, 1, length),
            spks=torch.ones(batch, 2),
            cond=torch.zeros_like(x),
        )

    x1, out1 = replay(1, 5)
    x2, out2 = replay(2, 7)

    assert torch.equal(out1, x1 + 1)
    assert torch.equal(out2, x2 + 1)
    entry = next(iter(runner._cache.values()))
    assert entry.static_x.shape == (2, 2, 8)
    assert torch.count_nonzero(entry.static_x[:, :, 7:]) == 0

    stats = runner.stats()
    assert stats["captures"] == 1
    assert stats["cache_hits"] == 1
    assert stats["shape_misses"] == 1
    assert stats["bucketed_calls"] == 2
    assert stats["replays"] == 2


@pytest.mark.core_model
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA graph capture")
def test_conditional_cfm_cuda_graph_replays_match_eager(monkeypatch):
    from vllm_omni.model_executor.models.cosyvoice3.code2wav_core.cfm import (
        ConditionalCFM,
    )

    class TinyEstimator(nn.Module):
        def forward(self, x, mask, mu, t, spks, cond):
            spk_term = spks.sum(dim=1, keepdim=True).unsqueeze(-1)
            t_term = t.view(-1, 1, 1)
            return (x * 0.125 + mu * 0.25 + cond * 0.5 + spk_term * 0.01 + t_term * 0.05) * mask

    def build_cfm():
        return (
            ConditionalCFM(
                in_channels=2,
                cfm_params=SimpleNamespace(
                    solver="euler",
                    t_scheduler="linear",
                    training_cfg_rate=0.0,
                    inference_cfg_rate=0.0,
                ),
                n_spks=1,
                spk_emb_dim=2,
                estimator=TinyEstimator().cuda().eval(),
            )
            .cuda()
            .eval()
        )

    device = torch.device("cuda")
    x = torch.randn(2, 2, 4, device=device)
    t_span = torch.linspace(0, 1, 4, device=device)
    mask = torch.ones(2, 1, 4, device=device)
    mu = torch.randn_like(x)
    spks = torch.randn(2, 2, device=device)
    cond = torch.randn_like(x)

    monkeypatch.delenv("COSYVOICE3_CFM_CUDA_GRAPH", raising=False)
    eager = build_cfm()
    eager_out = eager.solve_euler(x.clone(), t_span=t_span, mu=mu, mask=mask, spks=spks, cond=cond)

    monkeypatch.setenv("COSYVOICE3_CFM_CUDA_GRAPH", "1")
    graph = build_cfm()
    graph_out = graph.solve_euler(x.clone(), t_span=t_span, mu=mu, mask=mask, spks=spks, cond=cond)
    graph_out_2 = graph.solve_euler(x.clone(), t_span=t_span, mu=mu, mask=mask, spks=spks, cond=cond)
    torch.accelerator.synchronize()

    assert torch.allclose(graph_out, eager_out)
    assert torch.allclose(graph_out_2, eager_out)
    stats = graph.get_cuda_graph_stats()
    assert stats["captures"] == 1
    assert stats["replays"] == 2
    assert stats["shape_misses"] == 1
    assert stats["unique_graphs"] == 1
    assert stats["fallbacks"] == 0
    assert stats["replay_hit_rate"] == 1.0


@pytest.mark.core_model
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA graph capture")
def test_conditional_cfm_bucketed_cuda_graph_matches_eager(monkeypatch):
    from vllm_omni.model_executor.models.cosyvoice3.code2wav_core.cfm import (
        ConditionalCFM,
    )

    class TinyEstimator(nn.Module):
        def forward(self, x, mask, mu, t, spks, cond):
            spk_term = spks.sum(dim=1, keepdim=True).unsqueeze(-1)
            t_term = t.view(-1, 1, 1)
            return (x * 0.125 + mu * 0.25 + cond * 0.5 + spk_term * 0.01 + t_term * 0.05) * mask

    def build_cfm():
        return (
            ConditionalCFM(
                in_channels=2,
                cfm_params=SimpleNamespace(
                    solver="euler",
                    t_scheduler="linear",
                    training_cfg_rate=0.0,
                    inference_cfg_rate=0.0,
                ),
                n_spks=1,
                spk_emb_dim=2,
                estimator=TinyEstimator().cuda().eval(),
            )
            .cuda()
            .eval()
        )

    def inputs(batch: int, length: int):
        x = torch.randn(batch, 2, length, device="cuda")
        return {
            "x": x,
            "t_span": torch.linspace(0, 1, 4, device="cuda"),
            "mu": torch.randn_like(x),
            "mask": torch.ones(batch, 1, length, device="cuda"),
            "spks": torch.randn(batch, 2, device="cuda"),
            "cond": torch.randn_like(x),
        }

    first = inputs(1, 5)
    second = inputs(2, 7)
    monkeypatch.delenv("COSYVOICE3_CFM_CUDA_GRAPH", raising=False)
    eager = build_cfm()
    eager_first = eager.solve_euler(**first)
    eager_second = eager.solve_euler(**second)

    monkeypatch.setenv("COSYVOICE3_CFM_CUDA_GRAPH", "1")
    monkeypatch.setenv("COSYVOICE3_CFM_CUDA_GRAPH_TIMESTEP_BUCKETS", "8")
    monkeypatch.setenv("COSYVOICE3_CFM_CUDA_GRAPH_BATCH_BUCKETS", "2")
    graph = build_cfm()
    graph_first = graph.solve_euler(**first)
    graph_second = graph.solve_euler(**second)
    torch.accelerator.synchronize()

    assert torch.allclose(graph_first, eager_first)
    assert torch.allclose(graph_second, eager_second)
    stats = graph.get_cuda_graph_stats()
    assert stats["captures"] == 1
    assert stats["cache_hits"] == 1
    assert stats["shape_misses"] == 1
    assert stats["bucketed_calls"] == 2
    assert stats["fallbacks"] == 0
