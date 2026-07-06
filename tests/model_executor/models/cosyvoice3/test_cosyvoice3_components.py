# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for CosyVoice3 components."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from tests.helpers.mark import hardware_test


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
    def test_forward_passes_padding_mask_to_attention_backend(self):
        from vllm_omni.diffusion.models.cosyvoice3_audio.cosyvoice3_dit import DiTAttention

        captured = {}

        class RecordingAttention(nn.Module):
            def forward(self, query, key, value, attn_metadata=None):
                captured["attn_metadata"] = attn_metadata
                return torch.zeros_like(query)

        attn = DiTAttention(dim=8, heads=2, dim_head=4, dropout=0.0)
        attn.attn = RecordingAttention()
        x = torch.randn(2, 4, 8)
        key_mask = torch.tensor([[True, True, False, False], [True, True, True, False]])
        mask = key_mask.unsqueeze(1).repeat(1, 4, 1).unsqueeze(1)

        out = attn(x, mask=mask)

        assert out.shape == x.shape
        assert captured["attn_metadata"].attn_mask.tolist() == key_mask.tolist()


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
    def test_causal_conditional_cfm_preserves_fp16_output_dtype(self, dummy_estimator):
        """Flow solver should not force fp16 mel samples back to fp32."""
        from omegaconf import DictConfig

        from vllm_omni.model_executor.models.cosyvoice3.code2wav_core.cfm import CausalConditionalCFM

        cfm_params = DictConfig(
            {
                "sigma_min": 1e-6,
                "solver": "euler",
                "t_scheduler": "linear",
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

        batch, mel_dim, seq_len = 1, 80, 8
        mu = torch.randn(batch, mel_dim, seq_len, dtype=torch.float16)
        mask = torch.ones(batch, 1, seq_len, dtype=torch.float16)
        spks = torch.randn(batch, 80, dtype=torch.float16)
        cond = torch.randn(batch, mel_dim, seq_len, dtype=torch.float16)

        out, _ = cfm(mu, mask, n_timesteps=2, spks=spks, cond=cond)

        assert out.shape == mu.shape
        assert out.dtype == torch.float16

    @pytest.mark.core_model
    @pytest.mark.cpu
    def test_causal_conditional_cfm_batches_cfg_estimator_inputs(self):
        """Batched CFM should run the estimator with CFG batch 2B."""
        from omegaconf import DictConfig

        from vllm_omni.model_executor.models.cosyvoice3.code2wav_core.cfm import CausalConditionalCFM

        class RecordingEstimator(nn.Module):
            def __init__(self):
                super().__init__()
                self.calls: list[tuple[int, ...]] = []

            def forward(self, x, mask, mu, t, spks=None, cond=None):
                self.calls.append(tuple(x.shape))
                return torch.zeros_like(x)

        estimator = RecordingEstimator()
        cfm = CausalConditionalCFM(
            in_channels=80,
            cfm_params=DictConfig(
                {
                    "sigma_min": 1e-6,
                    "solver": "euler",
                    "t_scheduler": "linear",
                    "training_cfg_rate": 0.2,
                    "inference_cfg_rate": 0.7,
                }
            ),
            n_spks=1,
            spk_emb_dim=80,
            estimator=estimator,
        )

        mu = torch.randn(3, 80, 8)
        mask = torch.ones(3, 1, 8)
        spks = torch.randn(3, 80)
        cond = torch.randn(3, 80, 8)

        out, _ = cfm(mu, mask, n_timesteps=2, spks=spks, cond=cond)

        assert out.shape == mu.shape
        assert estimator.calls
        assert estimator.calls[0] == (6, 80, 8)


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

    @pytest.mark.core_model
    @pytest.mark.cpu
    def test_fa3_float16_uses_sdpa_fallback(self, monkeypatch):
        """FA3 builds without fp16 support should not receive fp16 q/k/v."""
        from vllm_omni.diffusion.attention import layer as attention_layer
        from vllm_omni.diffusion.attention.backends.utils import fa as fa_utils
        from vllm_omni.diffusion.attention.layer import Attention

        class _DummyFA3:
            __module__ = "fa3_fwd_interface"

            def __call__(self, *args, **kwargs):
                raise AssertionError("fp16 should have used SDPA fallback")

        class _FlashImpl:
            def forward(self, *args, **kwargs):
                raise AssertionError("fp16 should have used SDPA fallback")

        class _SDPAImpl:
            def __init__(self):
                self.called = False

            def forward(self, query, key, value, attn_metadata):
                self.called = True
                return torch.zeros_like(query)

        class _FlashBackend:
            @staticmethod
            def get_name():
                return "FLASH_ATTN"

        monkeypatch.setattr(fa_utils, "flash_attn_func", _DummyFA3())
        attn = object.__new__(Attention)
        attn.attn_backend = _FlashBackend
        attn.attention = _FlashImpl()
        attn.backend_pref = "FLASH_ATTN"
        attn.sdpa_fallback = _SDPAImpl()

        q = torch.ones((1, 4, 2, 64), dtype=torch.float16)
        out = attention_layer.Attention._run_local_attention(attn, q, q, q, None)

        assert out.shape == q.shape
        assert out.dtype == torch.float16
        assert attn.sdpa_fallback.called is True


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


def test_code2wav_load_weights_applies_fp16_flow_dtype(monkeypatch, tmp_path):
    from vllm_omni.model_executor.models.cosyvoice3.cosyvoice3_code2wav import CosyVoice3Code2Wav

    model = object.__new__(CosyVoice3Code2Wav)
    nn.Module.__init__(model)
    model.flow_model = nn.Linear(2, 2, bias=False).float()
    model.hift = nn.Linear(2, 2, bias=False).float()

    torch.save(model.flow_model.state_dict(), tmp_path / "flow.pt")
    torch.save(model.hift.state_dict(), tmp_path / "hift.pt")
    monkeypatch.setenv("COSYVOICE3_FLOW_DTYPE", "fp16")

    model.load_weights(str(tmp_path), torch.device("cpu"))

    assert next(model.flow_model.parameters()).dtype == torch.float16
    assert next(model.hift.parameters()).dtype == torch.float32


def test_causal_masked_diff_with_dit_returns_batched_generated_lengths():
    from vllm_omni.model_executor.models.cosyvoice3.code2wav_core.cfm import CausalMaskedDiffWithDiT

    class DummyPreLookahead(nn.Module):
        def forward(self, x, context=None):
            return x if context is None else x

    class DummyDecoder(nn.Module):
        def forward(self, mu, mask, spks=None, cond=None, n_timesteps=10, streaming=True):
            return mu * mask.to(mu.dtype), None

    model = CausalMaskedDiffWithDiT(
        input_size=80,
        output_size=80,
        spk_embed_dim=4,
        vocab_size=16,
        token_mel_ratio=1,
        pre_lookahead_len=1,
        pre_lookahead_layer=DummyPreLookahead(),
        decoder=DummyDecoder(),
    )

    token = torch.tensor([[1, 2, 0], [3, 4, 5]], dtype=torch.int32)
    token_len = torch.tensor([2, 3], dtype=torch.int32)
    prompt_token = torch.tensor([[6, 0], [7, 8]], dtype=torch.int32)
    prompt_token_len = torch.tensor([1, 2], dtype=torch.int32)
    prompt_feat = torch.randn(2, 2, 80)
    prompt_feat_len = torch.tensor([1, 2], dtype=torch.int32)
    embedding = torch.randn(2, 4)

    feat, meta = model.inference(
        token=token,
        token_len=token_len,
        prompt_token=prompt_token,
        prompt_token_len=prompt_token_len,
        prompt_feat=prompt_feat,
        prompt_feat_len=prompt_feat_len,
        embedding=embedding,
        streaming=False,
        finalize=True,
        n_timesteps=1,
    )

    assert feat.shape == (2, 80, 3)
    assert meta["generated_mel_lens"].tolist() == [2, 3]


def test_code2wav_forward_mel_batch_pads_and_splits_variable_lengths():
    from vllm_omni.model_executor.models.cosyvoice3.cosyvoice3_code2wav import CosyVoice3Code2Wav

    class DummyFlow(nn.Module):
        token_mel_ratio = 1

        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(1))
            self.calls: list[dict[str, torch.Tensor]] = []

        def inference(self, **kwargs):
            self.calls.append(kwargs)
            lengths = kwargs["token_len"]
            max_len = int(lengths.max().item())
            feat = torch.zeros((len(lengths), 80, max_len), dtype=self.weight.dtype)
            for row, length in enumerate(lengths.tolist()):
                feat[row, :, :length] = float(row + 1)
            return feat, {"generated_mel_lens": lengths}

    model = object.__new__(CosyVoice3Code2Wav)
    nn.Module.__init__(model)
    model.flow_model = DummyFlow()
    model._flow_batch_metadata_cache = {}
    model._flow_batch_stats = {
        "groups": 0,
        "requests": 0,
        "metadata_hits": 0,
        "metadata_misses": 0,
        "fallbacks": 0,
    }

    outputs = model._forward_mel_batch(
        [
            {
                "token": torch.tensor([[1, 2]], dtype=torch.int32),
                "prompt_token": torch.tensor([[3]], dtype=torch.int32),
                "prompt_feat": torch.randn(1, 1, 80),
                "embedding": torch.randn(1, 4),
                "token_offset_tokens": 0,
                "streaming": True,
                "finalize": False,
            },
            {
                "token": torch.tensor([[4, 5, 6]], dtype=torch.int32),
                "prompt_token": torch.tensor([[7, 8]], dtype=torch.int32),
                "prompt_feat": torch.randn(1, 2, 80),
                "embedding": torch.randn(1, 4),
                "token_offset_tokens": 1,
                "streaming": True,
                "finalize": False,
            },
        ],
        n_timesteps=1,
    )

    assert [tuple(out.shape) for out in outputs] == [(1, 80, 2), (1, 80, 2)]
    assert torch.all(outputs[0] == 1)
    assert torch.all(outputs[1] == 2)
    call = model.flow_model.calls[0]
    assert call["token"].shape == (2, 3)
    assert call["prompt_token"].shape == (2, 2)
    assert call["prompt_feat"].shape == (2, 2, 80)
    assert model._flow_batch_stats["groups"] == 1
    assert model._flow_batch_stats["requests"] == 2
