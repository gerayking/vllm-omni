# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import functools
from threading import Lock
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest
import torch
import torch.nn as nn
from vllm.v1.outputs import SamplerOutput
from vllm.v1.sample.logits_processor.state import LogitsProcessors
from vllm.v1.sample.metadata import SamplingMetadata

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

if TYPE_CHECKING:
    from vllm_omni.model_executor.models.cosyvoice3.cosyvoice3 import CosyVoice3Model


@functools.lru_cache(maxsize=1)
def _cosyvoice3_model_and_runner():
    """Defer heavy Omni/vLLM imports until a test runs (avoids duplicate CustomOp init)."""
    from vllm_omni.model_executor.models.cosyvoice3.cosyvoice3 import CosyVoice3Model
    from vllm_omni.worker.gpu_ar_model_runner import GPUARModelRunner

    return CosyVoice3Model, GPUARModelRunner


class _DummyCode2Wav:
    def __init__(
        self,
        vocab_size: int,
        num_samples: int = 32,
        outputs: list[tuple[torch.Tensor, dict[str, object] | None]] | None = None,
        batch_error: Exception | None = None,
    ):
        self.input_embedding = SimpleNamespace(num_embeddings=vocab_size)
        self.num_samples = num_samples
        self.outputs = list(outputs or [])
        self.batch_error = batch_error
        self.forward_calls: list[dict[str, object]] = []
        self.forward_streaming_calls: list[dict[str, object]] = []
        self.forward_mel_batch_calls: list[dict[str, object]] = []
        self.audio_from_mel_calls: list[torch.Tensor] = []
        self.streaming_audio_from_mel_calls: list[dict[str, object]] = []
        self._flow_batch_stats = {
            "groups": 0,
            "requests": 0,
            "metadata_hits": 0,
            "metadata_misses": 0,
            "fallbacks": 0,
        }

    def forward(self, **kwargs):
        self.forward_calls.append(kwargs)
        token = kwargs["token"]
        num_samples = int(token.shape[-1])
        return torch.linspace(-1.0, 1.0, max(num_samples, 1), dtype=torch.float32).reshape(1, 1, -1)

    def forward_streaming(self, **kwargs):
        self.forward_streaming_calls.append(kwargs)
        if self.outputs:
            return self.outputs.pop(0)

        token = kwargs["token"]
        num_samples = int(token.shape[-1])
        audio = torch.linspace(-1.0, 1.0, max(num_samples, 1), dtype=torch.float32).reshape(1, 1, -1)
        new_state = None
        if not kwargs.get("finalize", False):
            new_state = {
                "mel": torch.ones((1, 80, max(num_samples, 1)), dtype=torch.float32),
                "speech_offset": audio.shape[-1],
            }
        return audio, new_state

    def _forward_mel_batch(self, requests, n_timesteps=10):
        if self.batch_error is not None:
            raise self.batch_error
        self.forward_mel_batch_calls.append({"requests": requests, "n_timesteps": n_timesteps})
        feats = []
        for idx, req in enumerate(requests):
            length = int(req["token"].shape[-1])
            offset = int(req.get("token_offset_tokens", 0))
            mel_len = max(1, length - offset)
            feats.append(torch.full((1, 80, mel_len), float(idx + 1), dtype=torch.float32))
        self._flow_batch_stats["groups"] += 1
        self._flow_batch_stats["requests"] += len(requests)
        return feats

    def _audio_from_mel(self, feat):
        self.audio_from_mel_calls.append(feat)
        value = float(feat.reshape(-1)[0].item()) if feat.numel() else 0.0
        return torch.full((1, 1, max(int(feat.shape[-1]), 1)), value, dtype=torch.float32)

    def _streaming_audio_from_mel(self, feat, *, cache_state=None, finalize=False):
        self.streaming_audio_from_mel_calls.append(
            {"feat": feat, "cache_state": cache_state, "finalize": finalize}
        )
        value = float(feat.reshape(-1)[0].item()) if feat.numel() else 0.0
        audio = torch.full((1, 1, max(int(feat.shape[-1]), 1)), value, dtype=torch.float32)
        new_state = None
        if not finalize:
            new_state = {
                "mel": feat.detach().cpu().contiguous(),
                "speech_offset": audio.shape[-1],
            }
        return audio, new_state


def _make_code2wav_model(
    *,
    with_stride_cfg: bool = False,
    num_samples: int = 32,
    outputs: list[tuple[torch.Tensor, dict[str, object] | None]] | None = None,
    batch_error: Exception | None = None,
) -> CosyVoice3Model:
    CosyVoice3Model, _ = _cosyvoice3_model_and_runner()
    model = object.__new__(CosyVoice3Model)
    nn.Module.__init__(model)
    model.model_stage = "cosyvoice3_code2wav"
    hift_cfg = {} if not with_stride_cfg else {"upsample_rates": [8, 5, 3], "istft_params": {"hop_len": 4}}
    model.config = SimpleNamespace(
        sample_rate=24000,
        hift=hift_cfg,
        token_frame_rate=25 if with_stride_cfg else 0,
        token_mel_ratio=2 if with_stride_cfg else 0,
    )
    model.code2wav = _DummyCode2Wav(
        vocab_size=4,
        num_samples=num_samples,
        outputs=outputs,
        batch_error=batch_error,
    )
    # Short-circuit the lazy TensorRT estimator swap: these tests exercise the
    # forward audio logic, not the TRT path. On a GPU CI runner the swap would
    # otherwise run and dereference ``self.model_dir`` (only set in __init__,
    # which this fixture bypasses via object.__new__).
    model._code2wav_trt_done = True
    model.source_cache_len = 4
    model.speech_window = torch.hamming_window(8, periodic=False)
    model._stream_audio_cache_by_req = {}
    model._stream_audio_cache_lock = Lock()
    model._stream_vocoder_cache_by_req = {}
    return model


def _make_talker_model() -> CosyVoice3Model:
    CosyVoice3Model, _ = _cosyvoice3_model_and_runner()
    model = object.__new__(CosyVoice3Model)
    nn.Module.__init__(model)
    model.model_stage = "cosyvoice3_talker"
    model.config = SimpleNamespace(
        llm={
            "speech_token_size": 6561,
            "eos_token_id": 6562,
            "sampling": {
                "top_p": 0.8,
                "top_k": 25,
                "win_size": 10,
                "tau_r": 0.1,
            },
        },
        vocab_size=151923,
    )
    return model


def _make_sampling_metadata(
    *,
    output_token_ids: list[list[int]],
    repetition_penalty: float = 2.0,
) -> SamplingMetadata:
    return SamplingMetadata(
        temperature=torch.tensor([1.0], dtype=torch.float32),
        all_greedy=False,
        all_random=True,
        top_p=torch.tensor([0.8], dtype=torch.float32),
        top_k=torch.tensor([25], dtype=torch.int32),
        generators={},
        max_num_logprobs=None,
        no_penalties=False,
        prompt_token_ids=None,
        frequency_penalties=torch.zeros(1, dtype=torch.float32),
        presence_penalties=torch.zeros(1, dtype=torch.float32),
        repetition_penalties=torch.tensor([repetition_penalty], dtype=torch.float32),
        output_token_ids=output_token_ids,
        allowed_token_ids_mask=None,
        bad_words_token_ids={},
        logitsprocs=LogitsProcessors(),
    )


def test_forward_prefers_token_offset_when_present():
    model = _make_code2wav_model()

    runtime_info = [
        {
            "embed": {
                "speech_token": torch.tensor([[1, 2, 3]], dtype=torch.long),
                "speech_feat": torch.tensor([[[0.1, 0.2], [0.3, 0.4]]], dtype=torch.float32),
                "embedding": torch.tensor([[0.5, 0.6]], dtype=torch.float32),
            },
            "meta": {"left_context_size": 2},
        }
    ]

    out = model.forward(
        input_ids=torch.tensor([0, 1, 2], dtype=torch.long),
        positions=torch.tensor([0, 1, 2], dtype=torch.long),
        model_intermediate_buffer=runtime_info,
        seq_token_counts=[3],
    )

    assert len(out.multimodal_outputs["audio"]) == 1
    assert out.multimodal_outputs["audio"][0].numel() > 0
    assert len(model.code2wav.forward_streaming_calls) == 1
    call = model.code2wav.forward_streaming_calls[0]
    assert call["token"].shape == (1, 3)
    assert call["token_offset_tokens"] == 2
    assert call["finalize"] is False


def test_forward_falls_back_to_left_context_size_for_backward_compat():
    model = _make_code2wav_model()

    runtime_info = [
        {
            "embed": {
                "speech_token": torch.tensor([[1, 2, 3]], dtype=torch.long),
                "speech_feat": torch.tensor([[[0.1, 0.2], [0.3, 0.4]]], dtype=torch.float32),
                "embedding": torch.tensor([[0.5, 0.6]], dtype=torch.float32),
            },
            "meta": {"left_context_size": 2},
        }
    ]

    model.forward(
        input_ids=torch.tensor([0, 1, 2], dtype=torch.long),
        positions=torch.tensor([0, 1, 2], dtype=torch.long),
        model_intermediate_buffer=runtime_info,
        seq_token_counts=[3],
    )

    assert model.code2wav.forward_streaming_calls[0]["token_offset_tokens"] == 2


def test_forward_ignores_single_request_padded_tail_tokens():
    model = _make_code2wav_model(with_stride_cfg=True)
    runtime_info = [
        {
            "embed": {
                "speech_token": torch.tensor([[1, 2, 3]], dtype=torch.long),
                "speech_feat": torch.tensor([[[0.1, 0.2], [0.3, 0.4]]], dtype=torch.float32),
                "embedding": torch.tensor([[0.5, 0.6]], dtype=torch.float32),
            },
            "meta": {"left_context_size": 0},
        }
    ]

    out = model.forward(
        input_ids=torch.tensor([0, 1, 2, 3, 3], dtype=torch.long),
        positions=torch.tensor([0, 1, 2, 3, 4], dtype=torch.long),
        model_intermediate_buffer=runtime_info,
        seq_token_counts=[3],
    )

    # The padded tail must not contribute to code2wav length.
    assert out.multimodal_outputs["audio"][0].numel() == 3
    assert model.code2wav.forward_streaming_calls[0]["token"].tolist() == [[0, 1, 2]]


def test_forward_uses_non_stream_decode_without_chunk_metadata():
    model = _make_code2wav_model()

    runtime_info = [
        {
            "embed": {
                "speech_token": torch.tensor([[1, 2, 3]], dtype=torch.long),
                "speech_feat": torch.tensor([[[0.1, 0.2], [0.3, 0.4]]], dtype=torch.float32),
                "embedding": torch.tensor([[0.5, 0.6]], dtype=torch.float32),
            },
            "ids": {"prompt": [101, 102]},
            "generated_len": 3,
        }
    ]

    out = model.forward(
        input_ids=torch.tensor([0, 1, 2], dtype=torch.long),
        positions=torch.tensor([0, 1, 2], dtype=torch.long),
        model_intermediate_buffer=runtime_info,
        seq_token_counts=[3],
    )

    assert out.multimodal_outputs["audio"][0].numel() == 3
    assert len(model.code2wav.forward_calls) == 1
    assert len(model.code2wav.forward_streaming_calls) == 0
    call = model.code2wav.forward_calls[0]
    assert call["token"].tolist() == [[0, 1, 2]]
    assert call["token_offset_tokens"] == 0


def test_forward_passes_speaker_id_to_single_code2wav_call():
    model = _make_code2wav_model()

    runtime_info = [
        {
            "embed": {
                "speech_token": torch.tensor([[1, 2, 3]], dtype=torch.long),
                "speech_feat": torch.ones((1, 6, 80), dtype=torch.float32),
                "embedding": torch.ones((1, 4), dtype=torch.float32),
            },
            "meta": {"left_context_size": 0},
            "speaker": ["speaker-a"],
        }
    ]

    model.forward(
        input_ids=torch.tensor([0, 1, 2], dtype=torch.long),
        positions=torch.arange(3, dtype=torch.long),
        model_intermediate_buffer=runtime_info,
        seq_token_counts=[3],
    )

    assert model.code2wav.forward_streaming_calls[0]["spk_id"] == "speaker-a"


def test_forward_uses_non_stream_talker_prefill_offset():
    model = _make_code2wav_model()

    runtime_info = [
        {
            "embed": {
                "speech_token": torch.tensor([[1, 2, 3]], dtype=torch.long),
                "speech_feat": torch.tensor([[[0.1, 0.2], [0.3, 0.4]]], dtype=torch.float32),
                "embedding": torch.tensor([[0.5, 0.6]], dtype=torch.float32),
            },
            "meta": {"talker_prefill_offset": 3},
        }
    ]

    model.forward(
        input_ids=torch.tensor([0, 1, 2], dtype=torch.long),
        positions=torch.tensor([0, 1, 2], dtype=torch.long),
        model_intermediate_buffer=runtime_info,
        seq_token_counts=[3],
    )

    assert model.code2wav.forward_calls[0]["token_offset_tokens"] == 3


def test_forward_flow_batch_disabled_preserves_single_request_path(monkeypatch):
    monkeypatch.delenv("COSYVOICE3_FLOW_BATCH", raising=False)
    model = _make_code2wav_model(with_stride_cfg=True)
    runtime_info = [
        {
            "embed": {
                "speech_token": torch.tensor([[1, 2]], dtype=torch.long),
                "speech_feat": torch.ones((1, 4, 80), dtype=torch.float32),
                "embedding": torch.ones((1, 4), dtype=torch.float32),
            },
            "meta": {"left_context_size": 0},
        },
        {
            "embed": {
                "speech_token": torch.tensor([[1, 2, 3]], dtype=torch.long),
                "speech_feat": torch.ones((1, 6, 80), dtype=torch.float32),
                "embedding": torch.ones((1, 4), dtype=torch.float32),
            },
            "meta": {"left_context_size": 0},
        },
    ]

    model.forward(
        input_ids=torch.tensor([0, 1, 0, 1, 2], dtype=torch.long),
        positions=torch.arange(5, dtype=torch.long),
        model_intermediate_buffer=runtime_info,
        seq_token_counts=[2, 3],
    )

    assert len(model.code2wav.forward_mel_batch_calls) == 0
    assert len(model.code2wav.forward_streaming_calls) == 2


def test_forward_flow_batch_enabled_groups_heterogeneous_requests(monkeypatch):
    monkeypatch.setenv("COSYVOICE3_FLOW_BATCH", "1")
    monkeypatch.setenv("COSYVOICE3_FLOW_BATCH_MAX_SIZE", "4")
    model = _make_code2wav_model(with_stride_cfg=True)
    runtime_info = [
        {
            "embed": {
                "speech_token": torch.tensor([[1, 2]], dtype=torch.long),
                "speech_feat": torch.ones((1, 4, 80), dtype=torch.float32),
                "embedding": torch.ones((1, 4), dtype=torch.float32),
            },
            "meta": {"left_context_size": 0, "stream_finished": torch.tensor(False)},
            "speaker": ["speaker-a"],
        },
        {
            "embed": {
                "speech_token": torch.tensor([[1, 2, 3]], dtype=torch.long),
                "speech_feat": torch.ones((1, 6, 80), dtype=torch.float32),
                "embedding": torch.ones((1, 4), dtype=torch.float32),
            },
            "meta": {"left_context_size": 1, "stream_finished": torch.tensor(False)},
            "speaker": ["speaker-b"],
        },
    ]

    out = model.forward(
        input_ids=torch.tensor([0, 1, 0, 1, 2], dtype=torch.long),
        positions=torch.arange(5, dtype=torch.long),
        model_intermediate_buffer=runtime_info,
        seq_token_counts=[2, 3],
    )

    assert len(model.code2wav.forward_streaming_calls) == 0
    assert len(model.code2wav.forward_mel_batch_calls) == 1
    batch_call = model.code2wav.forward_mel_batch_calls[0]
    assert len(batch_call["requests"]) == 2
    assert [req["token"].shape[-1] for req in batch_call["requests"]] == [2, 3]
    assert [req["spk_id"] for req in batch_call["requests"]] == ["speaker-a", "speaker-b"]
    assert [audio.tolist() for audio in out.multimodal_outputs["audio"]] == [
        [1.0, 1.0],
        [2.0, 2.0],
    ]


def test_forward_flow_batch_falls_back_to_single_request_on_batch_error(monkeypatch):
    monkeypatch.setenv("COSYVOICE3_FLOW_BATCH", "1")
    model = _make_code2wav_model(with_stride_cfg=True, batch_error=RuntimeError("batch boom"))
    runtime_info = [
        {
            "embed": {
                "speech_token": torch.tensor([[1, 2]], dtype=torch.long),
                "speech_feat": torch.ones((1, 4, 80), dtype=torch.float32),
                "embedding": torch.ones((1, 4), dtype=torch.float32),
            },
            "meta": {"left_context_size": 0},
        },
        {
            "embed": {
                "speech_token": torch.tensor([[1, 2, 3]], dtype=torch.long),
                "speech_feat": torch.ones((1, 6, 80), dtype=torch.float32),
                "embedding": torch.ones((1, 4), dtype=torch.float32),
            },
            "meta": {"left_context_size": 0},
        },
    ]

    model.forward(
        input_ids=torch.tensor([0, 1, 0, 1, 2], dtype=torch.long),
        positions=torch.arange(5, dtype=torch.long),
        model_intermediate_buffer=runtime_info,
        seq_token_counts=[2, 3],
    )

    assert len(model.code2wav.forward_mel_batch_calls) == 0
    assert len(model.code2wav.forward_streaming_calls) == 2
    assert model.code2wav._flow_batch_stats["fallbacks"] == 2


def test_forward_reuses_streaming_cache_state_between_chunks():
    model = _make_code2wav_model(
        outputs=[
            (
                torch.arange(4, dtype=torch.float32).reshape(1, 1, -1),
                {"mel": torch.ones((1, 80, 3), dtype=torch.float32), "speech_offset": 4},
            ),
            (
                torch.full((1, 1, 2), 9.0, dtype=torch.float32),
                {"mel": torch.ones((1, 80, 5), dtype=torch.float32), "speech_offset": 6},
            ),
        ]
    )
    runtime_info = [
        {
            "embed": {
                "speech_token": torch.tensor([[1, 2, 3]], dtype=torch.long),
                "speech_feat": torch.tensor([[[0.1, 0.2], [0.3, 0.4]]], dtype=torch.float32),
                "embedding": torch.tensor([[0.5, 0.6]], dtype=torch.float32),
            },
            "meta": {
                "req_id": ["rid-stream"],
                "stream_finished": torch.tensor(False),
                "left_context_size": 0,
            },
        }
    ]

    out1 = model.forward(
        input_ids=torch.tensor([0, 1, 2], dtype=torch.long),
        positions=torch.tensor([0, 1, 2], dtype=torch.long),
        model_intermediate_buffer=runtime_info,
        seq_token_counts=[3],
    )
    assert out1.multimodal_outputs["audio"][0].tolist() == [0.0, 1.0, 2.0, 3.0]
    assert model.code2wav.forward_streaming_calls[0]["cache_state"] is None

    out2 = model.forward(
        input_ids=torch.tensor([0, 1, 2], dtype=torch.long),
        positions=torch.tensor([0, 1, 2], dtype=torch.long),
        model_intermediate_buffer=runtime_info,
        seq_token_counts=[3],
    )
    assert out2.multimodal_outputs["audio"][0].tolist() == [9.0, 9.0]
    cache_state = model.code2wav.forward_streaming_calls[1]["cache_state"]
    assert cache_state is not None
    assert cache_state["speech_offset"] == 4
    assert "rid-stream" in model._stream_vocoder_cache_by_req


def test_forward_clears_streaming_cache_on_terminal_chunk():
    model = _make_code2wav_model(
        outputs=[
            (
                torch.arange(4, dtype=torch.float32).reshape(1, 1, -1),
                {"mel": torch.ones((1, 80, 3), dtype=torch.float32), "speech_offset": 4},
            ),
            (
                torch.full((1, 1, 1), 7.0, dtype=torch.float32),
                None,
            ),
        ]
    )
    runtime_info = [
        {
            "embed": {
                "speech_token": torch.tensor([[1, 2, 3]], dtype=torch.long),
                "speech_feat": torch.tensor([[[0.1, 0.2], [0.3, 0.4]]], dtype=torch.float32),
                "embedding": torch.tensor([[0.5, 0.6]], dtype=torch.float32),
            },
            "meta": {
                "req_id": ["rid-stream"],
                "stream_finished": torch.tensor(False),
                "left_context_size": 0,
            },
        }
    ]

    model.forward(
        input_ids=torch.tensor([0, 1, 2], dtype=torch.long),
        positions=torch.tensor([0, 1, 2], dtype=torch.long),
        model_intermediate_buffer=runtime_info,
        seq_token_counts=[3],
    )
    assert "rid-stream" in model._stream_vocoder_cache_by_req

    runtime_info[0]["meta"]["stream_finished"] = torch.tensor(True)
    out = model.forward(
        input_ids=torch.tensor([0, 1, 2], dtype=torch.long),
        positions=torch.tensor([0, 1, 2], dtype=torch.long),
        model_intermediate_buffer=runtime_info,
        seq_token_counts=[3],
    )
    assert out.multimodal_outputs["audio"][0].tolist() == [7.0]
    assert "rid-stream" not in model._stream_vocoder_cache_by_req


def test_sample_uses_ras_rejection_for_recent_repetition():
    model = _make_talker_model()
    metadata = _make_sampling_metadata(output_token_ids=[[1] * 10])
    logits = torch.tensor([[-1e9, 10.0, 0.0]], dtype=torch.float32)

    out = model.sample(logits, metadata)

    assert out is not None
    assert out.sampled_token_ids.tolist() == [[2]]


def test_sample_tolerates_padded_rows_without_history():
    model = _make_talker_model()
    metadata = _make_sampling_metadata(output_token_ids=[[1] * 10])
    logits = torch.tensor(
        [
            [-1e9, 10.0, 0.0],
            [-1e9, 0.0, 10.0],
        ],
        dtype=torch.float32,
    )

    out = model.sample(logits, metadata)

    assert out is not None
    assert out.sampled_token_ids.shape == (2, 1)


def test_gpu_ar_model_runner_prefers_model_sampler_when_opted_in():
    metadata = _make_sampling_metadata(output_token_ids=[[1, 2, 3]])
    expected = SamplerOutput(
        sampled_token_ids=torch.tensor([[7]], dtype=torch.int32),
        logprobs_tensors=None,
    )
    calls: list[torch.Tensor] = []

    class _DummyInputBatch:
        def __init__(self):
            self.sampling_metadata = metadata
            self.updated = False

        def update_async_output_token_ids(self):
            # After PR 3681 fix, update_async_output_token_ids is called
            # BEFORE model sampler path to ensure async placeholder repair
            # runs for all sampling paths
            self.updated = True

    _, GPUARModelRunner = _cosyvoice3_model_and_runner()
    runner = object.__new__(GPUARModelRunner)
    runner.input_batch = _DummyInputBatch()
    runner.model = SimpleNamespace(
        prefer_model_sampler=True,
        sample=lambda logits, sampling_metadata: calls.append(logits.clone()) or expected,
    )
    runner.sampler = lambda **_: (_ for _ in ()).throw(AssertionError("fallback sampler should not be used"))

    out = runner._sample(torch.tensor([[0.1, 0.2]], dtype=torch.float32), spec_decode_metadata=None)

    assert out is expected
    assert runner.input_batch.updated is True
    assert len(calls) == 1


def test_gpu_ar_model_runner_supplies_req_output_history_to_model_sampler():
    metadata = _make_sampling_metadata(output_token_ids=[])
    seen_histories: list[list[list[int]]] = []

    class _DummyInputBatch:
        def __init__(self):
            self.sampling_metadata = metadata
            self.req_output_token_ids = [[1, 2, 3]]
            self.req_ids = ["rid-1"]
            self.sampled_token_ids_cpu = None
            self.async_copy_ready_event = None
            self.prev_req_id_to_index = None
            self.update_async_called = False

        def update_async_output_token_ids(self):
            # After PR 3681 fix, update_async_output_token_ids is called
            # BEFORE model sampler path to ensure async placeholder repair
            # runs for all sampling paths
            self.update_async_called = True

    _, GPUARModelRunner = _cosyvoice3_model_and_runner()
    runner = object.__new__(GPUARModelRunner)
    runner.input_batch = _DummyInputBatch()
    runner.model = SimpleNamespace(
        prefer_model_sampler=True,
        sample=lambda logits, sampling_metadata: (
            seen_histories.append([list(x) for x in sampling_metadata.output_token_ids])
            or SamplerOutput(sampled_token_ids=torch.tensor([[7]], dtype=torch.int32), logprobs_tensors=None)
        ),
    )
    runner.sampler = lambda **_: (_ for _ in ()).throw(AssertionError("fallback sampler should not be used"))

    runner._sample(torch.tensor([[0.1, 0.2]], dtype=torch.float32), spec_decode_metadata=None)

    assert runner.input_batch.update_async_called is True
    assert seen_histories == [[[1, 2, 3]]]


def test_gpu_ar_model_runner_repairs_async_placeholders_for_model_sampler():
    metadata = _make_sampling_metadata(output_token_ids=[])
    seen_histories: list[list[list[int]]] = []

    class _ReadyEvent:
        def __init__(self):
            self.synced = False

        def synchronize(self):
            self.synced = True

    class _DummyInputBatch:
        def __init__(self):
            self.sampling_metadata = metadata
            self.req_output_token_ids = [[11, -1]]
            self.req_ids = ["rid-1"]
            self.sampled_token_ids_cpu = torch.tensor([[29]], dtype=torch.int32)
            self.async_copy_ready_event = _ReadyEvent()
            self.prev_req_id_to_index = {"rid-1": 0}
            self.update_async_called = False

        def update_async_output_token_ids(self):
            # After PR 3681 fix, update_async_output_token_ids is called
            # BEFORE model sampler path to ensure async placeholder repair
            # runs for all sampling paths (model sampler + fallback sampler)
            self.update_async_called = True

    _, GPUARModelRunner = _cosyvoice3_model_and_runner()
    runner = object.__new__(GPUARModelRunner)
    runner.input_batch = _DummyInputBatch()
    runner.model = SimpleNamespace(
        prefer_model_sampler=True,
        sample=lambda logits, sampling_metadata: (
            seen_histories.append([list(x) for x in sampling_metadata.output_token_ids])
            or SamplerOutput(sampled_token_ids=torch.tensor([[7]], dtype=torch.int32), logprobs_tensors=None)
        ),
    )
    runner.sampler = lambda **_: (_ for _ in ()).throw(AssertionError("fallback sampler should not be used"))

    runner._sample(torch.tensor([[0.1, 0.2]], dtype=torch.float32), spec_decode_metadata=None)

    assert runner.input_batch.async_copy_ready_event.synced is True
    assert runner.input_batch.update_async_called is True
    assert seen_histories == [[[11, 29]]]


def test_cosyvoice3_flow_dtype_env_defaults_to_fp32(monkeypatch):
    from vllm_omni.model_executor.models.cosyvoice3.cosyvoice3_code2wav import _cosyvoice3_flow_dtype

    monkeypatch.delenv("COSYVOICE3_FLOW_DTYPE", raising=False)

    assert _cosyvoice3_flow_dtype() == torch.float32


def test_cosyvoice3_flow_dtype_env_accepts_fp16(monkeypatch):
    from vllm_omni.model_executor.models.cosyvoice3.cosyvoice3_code2wav import _cosyvoice3_flow_dtype

    monkeypatch.setenv("COSYVOICE3_FLOW_DTYPE", "fp16")

    assert _cosyvoice3_flow_dtype() == torch.float16


def test_cosyvoice3_flow_batch_env_defaults_disabled(monkeypatch):
    from vllm_omni.model_executor.models.cosyvoice3.cosyvoice3_code2wav import (
        _cosyvoice3_flow_batch_enabled,
        _cosyvoice3_flow_batch_max_size,
        _cosyvoice3_flow_batch_wait_ms,
    )

    monkeypatch.delenv("COSYVOICE3_FLOW_BATCH", raising=False)
    monkeypatch.delenv("COSYVOICE3_FLOW_BATCH_MAX_SIZE", raising=False)
    monkeypatch.delenv("COSYVOICE3_FLOW_BATCH_WAIT_MS", raising=False)

    assert _cosyvoice3_flow_batch_enabled() is False
    assert _cosyvoice3_flow_batch_max_size() == 4
    assert _cosyvoice3_flow_batch_wait_ms() == 10


def test_resolve_flow_estimator_onnx_skips_fp32_fallback_for_fp16_flow(monkeypatch, tmp_path):
    CosyVoice3Model, _ = _cosyvoice3_model_and_runner()
    model = SimpleNamespace(
        model_dir=str(tmp_path),
        config=SimpleNamespace(
            flow_estimator_onnx_path="missing.fp16.onnx",
            flow_estimator_onnx_path_fp32="flow.decoder.estimator.fp32.onnx",
            flow_estimator_onnx_repo=None,
        ),
    )
    (tmp_path / "flow.decoder.estimator.fp32.onnx").write_bytes(b"onnx")
    monkeypatch.delenv("COSYVOICE3_ESTIMATOR_ONNX", raising=False)
    monkeypatch.setenv("COSYVOICE3_FLOW_DTYPE", "fp16")

    assert CosyVoice3Model._resolve_flow_estimator_onnx(model) is None


def test_resolve_flow_estimator_onnx_allows_fp32_fallback_by_default(monkeypatch, tmp_path):
    CosyVoice3Model, _ = _cosyvoice3_model_and_runner()
    model = SimpleNamespace(
        model_dir=str(tmp_path),
        config=SimpleNamespace(
            flow_estimator_onnx_path="missing.fp16.onnx",
            flow_estimator_onnx_path_fp32="flow.decoder.estimator.fp32.onnx",
            flow_estimator_onnx_repo=None,
        ),
    )
    fp32_path = tmp_path / "flow.decoder.estimator.fp32.onnx"
    fp32_path.write_bytes(b"onnx")
    monkeypatch.delenv("COSYVOICE3_ESTIMATOR_ONNX", raising=False)
    monkeypatch.delenv("COSYVOICE3_FLOW_DTYPE", raising=False)

    assert CosyVoice3Model._resolve_flow_estimator_onnx(model) == str(fp32_path)
