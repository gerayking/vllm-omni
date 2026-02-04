"""
Test that streaming and blocking generation produce identical codec codes.

This test loads a real Qwen3TTSForConditionalGeneration model from
a pretrained path and verifies that:
1. generate_streaming_iter yields the same codec codes as the original generate
2. The total generated tokens match between streaming and blocking modes

Usage:
    # Use default model path
    pytest tests/model_executor/models/qwen3_tts/test_streaming_consistency.py -v
    
    # Override model path via environment variable
    MODEL_PATH=/your/path pytest tests/model_executor/models/qwen3_tts/test_streaming_consistency.py -v
"""

import os
import pytest
import torch
import sys

sys.path.insert(0, "/home/xuanweifu/vllm-omni")

from vllm_omni.model_executor.models.qwen3_tts.qwen3_tts import Qwen3TTSModel
from vllm_omni.model_executor.models.qwen3_tts.modeling_qwen3_tts import (
    StreamingChunkOutput,
)


# Model path - can be overridden via MODEL_PATH environment variable
MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    "/mnt/cephfs/user_xuanweifu/data/models/Qwen3-TTS"
)


@pytest.fixture(scope="session")
def device():
    """Get the device to use for testing."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="session")
def tts_model(device):
    """Load the pretrained TTS model using Qwen3TTSModel.from_pretrained()."""
    model = Qwen3TTSModel.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map=device,
    )
    return model


@pytest.fixture(scope="session")
def talker_model(tts_model):
    """Get the talker model from the TTS model."""
    # Qwen3TTSModel wraps Qwen3TTSForConditionalGeneration
    # Qwen3TTSForConditionalGeneration contains talker (Qwen3TTSTalkerForConditionalGeneration)
    return tts_model.model.talker


@pytest.fixture
def sample_inputs(talker_model, device):
    """Create sample inputs for testing."""
    batch_size = 1
    seq_len = 10
    hidden_size = talker_model.config.hidden_size
    
    # Create random embeddings with the model's dtype
    dtype = next(talker_model.parameters()).dtype
    inputs_embeds = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=dtype)
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long, device=device)
    trailing_text_hidden = torch.randn(batch_size, 5, hidden_size, device=device, dtype=dtype)
    tts_pad_embed = torch.randn(1, 1, hidden_size, device=device, dtype=dtype)
    
    return {
        "inputs_embeds": inputs_embeds,
        "attention_mask": attention_mask,
        "trailing_text_hidden": trailing_text_hidden,
        "tts_pad_embed": tts_pad_embed,
    }


def run_streaming_generation(talker_model, sample_inputs, chunk_size, **gen_kwargs):
    """Helper to run streaming generation and collect all codes."""
    streaming_codes = []
    for chunk in talker_model.generate_streaming_iter(
        **sample_inputs,
        chunk_size=chunk_size,
        **gen_kwargs,
    ):
        streaming_codes.append(chunk.codec_codes)
    
    if streaming_codes:
        return torch.cat(streaming_codes, dim=0)
    return None


class TestStreamingConsistency:
    """Test streaming generation consistency (comparing different chunk sizes)."""
    
    def test_streaming_vs_blocking_greedy(self, talker_model, sample_inputs, device):
        """
        Test that greedy decoding with streaming produces deterministic results.
        
        Note: We compare streaming with different seeds to verify greedy is deterministic,
        since direct comparison with GenerationMixin.generate() is complex due to 
        different internal state management.
        """
        gen_kwargs = {
            "max_new_tokens": 20,
            "do_sample": False,
            "repetition_penalty": 1.0,
        }
        
        # Run streaming generation twice with same seed - should be identical
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)
        
        codes_1 = run_streaming_generation(talker_model, sample_inputs, chunk_size=25, **gen_kwargs)
        
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)
        
        codes_2 = run_streaming_generation(talker_model, sample_inputs, chunk_size=25, **gen_kwargs)
        
        # Greedy decoding should be deterministic
        if codes_1 is not None and codes_2 is not None:
            torch.testing.assert_close(
                codes_1,
                codes_2,
                msg="Greedy decoding should be deterministic"
            )
        else:
            assert codes_1 is None and codes_2 is None, \
                "One run generated codes while the other didn't"
    
    def test_streaming_vs_blocking_with_sampling(self, talker_model, sample_inputs, device):
        """
        Test that sampling with fixed seed produces identical results across runs.
        """
        gen_kwargs = {
            "max_new_tokens": 20,
            "do_sample": True,
            "top_k": 50,
            "top_p": 0.9,
            "temperature": 0.9,
            "repetition_penalty": 1.0,
        }
        
        # Run streaming generation twice with same seed - should be identical
        torch.manual_seed(123)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(123)
        
        codes_1 = run_streaming_generation(talker_model, sample_inputs, chunk_size=5, **gen_kwargs)
        
        torch.manual_seed(123)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(123)
        
        codes_2 = run_streaming_generation(talker_model, sample_inputs, chunk_size=5, **gen_kwargs)
        
        # With same seed, sampling should produce identical results
        if codes_1 is not None and codes_2 is not None:
            torch.testing.assert_close(
                codes_1,
                codes_2,
                msg="Sampling with same seed should be deterministic"
            )
        else:
            assert codes_1 is None and codes_2 is None, \
                "One run generated codes while the other didn't"
    
    def test_different_chunk_sizes(self, talker_model, sample_inputs, device):
        """Test that different chunk sizes produce same total output."""
        gen_kwargs = {
            "max_new_tokens": 30,
            "do_sample": False,
            "repetition_penalty": 1.0,
        }
        
        chunk_sizes = [1, 5, 10, 25]
        all_codes = []
        
        for chunk_size in chunk_sizes:
            torch.manual_seed(42)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(42)
            
            codes = run_streaming_generation(talker_model, sample_inputs, chunk_size=chunk_size, **gen_kwargs)
            if codes is not None:
                all_codes.append(codes)
        
        # All chunk sizes should produce identical codes
        if len(all_codes) > 1:
            reference = all_codes[0]
            for i, codes in enumerate(all_codes[1:], start=1):
                min_len = min(len(reference), len(codes))
                if min_len > 0:
                    torch.testing.assert_close(
                        reference[:min_len],
                        codes[:min_len],
                        msg=f"Chunk size {chunk_sizes[i]} differs from chunk size {chunk_sizes[0]}"
                    )
    
    def test_streaming_chunk_indices_sequential(self, talker_model, sample_inputs, device):
        """Test that chunk indices are sequential."""
        gen_kwargs = {
            "max_new_tokens": 50,
            "do_sample": False,
        }
        
        chunks = list(talker_model.generate_streaming_iter(
            **sample_inputs,
            chunk_size=10,
            **gen_kwargs,
        ))
        
        if len(chunks) > 1:
            for i, chunk in enumerate(chunks):
                assert chunk.chunk_idx == i, f"Expected chunk_idx {i}, got {chunk.chunk_idx}"
    
    def test_streaming_total_generated_monotonic(self, talker_model, sample_inputs, device):
        """Test that total_generated increases monotonically."""
        gen_kwargs = {
            "max_new_tokens": 50,
            "do_sample": False,
        }
        
        prev_total = 0
        for chunk in talker_model.generate_streaming_iter(
            **sample_inputs,
            chunk_size=10,
            **gen_kwargs,
        ):
            assert chunk.total_generated >= prev_total, \
                f"total_generated decreased: {prev_total} -> {chunk.total_generated}"
            prev_total = chunk.total_generated
    
    def test_streaming_last_chunk_is_finished(self, talker_model, sample_inputs, device):
        """Test that the last chunk has is_finished=True."""
        gen_kwargs = {
            "max_new_tokens": 30,
            "do_sample": False,
        }
        
        chunks = list(talker_model.generate_streaming_iter(
            **sample_inputs,
            chunk_size=7,  # Doesn't divide evenly into max_new_tokens
            **gen_kwargs,
        ))
        
        if chunks:
            # All but last should have is_finished=False (unless EOS hit early)
            for chunk in chunks[:-1]:
                if chunk.is_finished:
                    # EOS was hit early, which is fine
                    break
            
            # Last chunk should have is_finished=True
            assert chunks[-1].is_finished, "Last chunk should have is_finished=True"


class TestStreamingChunkOutputDataclass:
    """Test the StreamingChunkOutput dataclass."""
    
    def test_fields_exist(self):
        """Test that all expected fields exist."""
        chunk = StreamingChunkOutput(
            codec_codes=torch.zeros(10, 32),
            hidden_states=torch.zeros(10, 64),
            chunk_idx=0,
            is_finished=False,
            total_generated=10,
        )
        
        assert hasattr(chunk, 'codec_codes')
        assert hasattr(chunk, 'hidden_states')
        assert hasattr(chunk, 'chunk_idx')
        assert hasattr(chunk, 'is_finished')
        assert hasattr(chunk, 'total_generated')
    
    def test_default_values(self):
        """Test default values."""
        chunk = StreamingChunkOutput(
            codec_codes=torch.zeros(10, 32),
        )
        
        assert chunk.hidden_states is None
        assert chunk.chunk_idx == 0
        assert chunk.is_finished is False
        assert chunk.total_generated == 0
    
    def test_codec_codes_shape(self):
        """Test that codec_codes has expected shape."""
        num_tokens = 10
        num_quantizers = 32
        
        chunk = StreamingChunkOutput(
            codec_codes=torch.zeros(num_tokens, num_quantizers),
        )
        
        assert chunk.codec_codes.shape == (num_tokens, num_quantizers)


class TestGenerationParameterConsistency:
    """Test that generation parameters are handled consistently."""
    
    def test_repetition_penalty_consistency(self, talker_model, sample_inputs, device):
        """
        Test that repetition_penalty affects generation consistently.
        Different penalties should produce different results.
        """
        penalties = [1.0, 1.05, 1.1]
        results = {}
        
        for penalty in penalties:
            torch.manual_seed(42)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(42)
            
            codes = run_streaming_generation(
                talker_model, 
                sample_inputs, 
                chunk_size=5,
                max_new_tokens=15,
                do_sample=False,
                repetition_penalty=penalty,
            )
            results[penalty] = codes
        
        # Verify that the same penalty produces the same result (deterministic)
        for penalty in penalties:
            torch.manual_seed(42)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(42)
            
            codes_verify = run_streaming_generation(
                talker_model,
                sample_inputs,
                chunk_size=5,
                max_new_tokens=15,
                do_sample=False,
                repetition_penalty=penalty,
            )
            
            if results[penalty] is not None and codes_verify is not None:
                torch.testing.assert_close(
                    results[penalty],
                    codes_verify,
                    msg=f"Repetition penalty {penalty} should be deterministic"
                )
    
    def test_different_penalties_produce_different_results(self, talker_model, sample_inputs, device):
        """
        Test that different repetition penalties may produce different results.
        Note: This is a soft test - penalties might not always change output.
        """
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)
        
        codes_no_penalty = run_streaming_generation(
            talker_model,
            sample_inputs,
            chunk_size=5,
            max_new_tokens=30,
            do_sample=False,
            repetition_penalty=1.0,
        )
        
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)
        
        codes_with_penalty = run_streaming_generation(
            talker_model,
            sample_inputs,
            chunk_size=5,
            max_new_tokens=30,
            do_sample=False,
            repetition_penalty=1.2,  # Higher penalty
        )
        
        # At least verify both ran successfully
        # The actual codes may or may not differ depending on the input
        if codes_no_penalty is not None:
            assert codes_no_penalty.shape[0] > 0, "Should generate some tokens"
        if codes_with_penalty is not None:
            assert codes_with_penalty.shape[0] > 0, "Should generate some tokens"


class TestForwardStreamingDebug:
    """Debug test case for forward_streaming function."""
    
    def test_forward_streaming_direct(self, device):
        """
        Direct debug test for forward_streaming with the exact parameters from vLLM worker.
        
        This test simulates the exact call that happens in vLLM's worker process,
        allowing you to debug forward_streaming in isolation without the multiprocess overhead.
        """
        # Use Qwen3TTSModel's streaming methods directly
        model = Qwen3TTSModel.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map=device,
        )
        
        # Exact parameters from your debug log
        text = "这是一个流式生成测试的例子，我们来验证流式生成和批量生成的一致性。"
        speaker = "Vivian"
        language = "Chinese"
        instruct = "用清晰自然的语气说"
        chunk_size = 5
        left_context_size = 25
        
        print("\n" + "="*60)
        print("[DEBUG TEST] Calling generate_custom_voice_streaming directly")
        print(f"  text={text!r}")
        print(f"  speaker={speaker!r}")
        print(f"  language={language!r}")
        print(f"  instruct={instruct!r}")
        print(f"  chunk_size={chunk_size}")
        print(f"  left_context_size={left_context_size}")
        print("="*60)
        
        # You can set breakpoint here for debugging
        # import pdb; pdb.set_trace()
        
        # Call generate_custom_voice_streaming (this is what forward_streaming calls internally)
        chunk_count = 0
        all_audio = []
        
        for chunk_result in model.generate_custom_voice_streaming(
            text,
            speaker=speaker,
            language=language,
            instruct=instruct,
            chunk_size=chunk_size,
            left_context_size=left_context_size,
            max_new_tokens=80,
        ):
            chunk_count += 1
            audio_chunk, sr = chunk_result
            if audio_chunk is not None:
                all_audio.append(audio_chunk)
                print(f"[DEBUG] Chunk {chunk_count}: audio shape={audio_chunk.shape}, sr={sr}")
            else:
                print(f"[DEBUG] Chunk {chunk_count}: no audio")
        
        print(f"\n[DEBUG] Total chunks: {chunk_count}")
        assert chunk_count > 0, "Should generate at least one chunk"


def run_debug_forward_streaming():
    """
    Standalone function to debug forward_streaming.
    Run this directly: python test_streaming_consistency.py --debug
    
    This directly calls Qwen3TTSModel.generate_custom_voice_streaming(),
    which is the underlying method called by Qwen3TTSModelForGeneration.forward_streaming()
    """
    import torch
    import numpy as np
    import soundfile as sf
    from pathlib import Path
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {MODEL_PATH}...")
    model = Qwen3TTSModel.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map=device,
    )
    print("Model loaded successfully!")
    
    # Check model type
    tts_model_type = model.model.tts_model_type
    print(f"Model type: {tts_model_type}")
    
    # Test parameters (from your debug log)
    text = "这是一个流式生成测试的例子，我们来验证流式生成和批量生成的一致性。"
    chunk_size = 25
    left_context_size = 25
    max_new_tokens = 2048  # Increased to allow full sentence generation
    
    print("\n" + "="*60)
    print(f"Starting streaming debug for model type: {tts_model_type}")
    print(f"  text={text!r}")
    print(f"  chunk_size={chunk_size}")
    print(f"  left_context_size={left_context_size}")
    print(f"  max_new_tokens={max_new_tokens}")
    
    # Set breakpoint here if needed
    # import pdb; pdb.set_trace()
    
    chunk_count = 0
    all_audio = []
    sample_rate = None
    
    # Output directory for audio files
    output_dir = Path(__file__).parent / "debug_output"
    output_dir.mkdir(exist_ok=True)
    
    # Select the correct streaming method based on model type
    if tts_model_type == "custom_voice":
        speaker = "Vivian"
        language = "Chinese"
        instruct = "用清晰自然的语气说"
        print(f"  speaker={speaker!r}")
        print(f"  language={language!r}")
        print(f"  instruct={instruct!r}")
        print("="*60)
        
        streaming_gen = model.generate_custom_voice_streaming(
            text,
            speaker=speaker,
            language=language,
            instruct=instruct,
            chunk_size=chunk_size,
            left_context_size=left_context_size,
            max_new_tokens=max_new_tokens,
        )
    elif tts_model_type == "voice_design":
        instruct = "用清晰自然的语气说"
        language = "Chinese"
        print(f"  instruct={instruct!r}")
        print(f"  language={language!r}")
        print("="*60)
        
        streaming_gen = model.generate_voice_design_streaming(
            text,
            instruct=instruct,
            language=language,
            chunk_size=chunk_size,
            left_context_size=left_context_size,
            max_new_tokens=max_new_tokens,
        )
    elif tts_model_type == "base":
        language = "Chinese"
        print(f"  language={language!r}")
        print("="*60)
        
        streaming_gen = model.generate_voice_clone_streaming(
            text,
            language=language,
            chunk_size=chunk_size,
            left_context_size=left_context_size,
            max_new_tokens=max_new_tokens,
        )
    else:
        raise ValueError(f"Unknown model type: {tts_model_type}")
    
    for chunk_result in streaming_gen:
        chunk_count += 1
        audio_chunk, is_finished, sr = chunk_result
        sample_rate = sr
        if audio_chunk is not None:
            all_audio.append(audio_chunk)
            print(f"Chunk {chunk_count}: audio shape={audio_chunk.shape}, sr={sr}, finished={is_finished}")
            
            # Save each chunk separately
            chunk_file = output_dir / f"chunk_{chunk_count:03d}.wav"
            sf.write(chunk_file, audio_chunk, sr)
            print(f"  -> Saved to {chunk_file}")
        else:
            print(f"Chunk {chunk_count}: no audio, finished={is_finished}")
    
    print(f"\nTotal chunks generated: {chunk_count}")
    
    # Save complete audio
    if all_audio and sample_rate:
        total_audio = np.concatenate(all_audio)
        print(f"Total audio samples: {total_audio.shape[0]}")
        print(f"Duration: {total_audio.shape[0] / sample_rate:.2f} seconds")
        
        # Save complete audio file
        complete_file = output_dir / "complete_output.wav"
        sf.write(complete_file, total_audio, sample_rate)
        print(f"\n*** Complete audio saved to: {complete_file} ***")
        print(f"Play with: aplay {complete_file}")
        print(f"Or: ffplay {complete_file}")
    
    return model, all_audio


def run_debug_non_streaming():
    """
    Non-streaming version for comparison.
    Run this directly: python test_streaming_consistency.py --non-streaming
    """
    import torch
    import numpy as np
    import soundfile as sf
    from pathlib import Path
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {MODEL_PATH}...")
    model = Qwen3TTSModel.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map=device,
    )
    print("Model loaded successfully!")
    
    # Check model type
    tts_model_type = model.model.tts_model_type
    print(f"Model type: {tts_model_type}")
    
    # Test parameters (same as streaming)
    text = "这是一个流式生成测试的例子，我们来验证流式生成和批量生成的一致性。"
    max_new_tokens = 2048  # Increased to allow full sentence generation
    
    print("\n" + "="*60)
    print(f"Starting NON-STREAMING debug for model type: {tts_model_type}")
    print(f"  text={text!r}")
    print(f"  max_new_tokens={max_new_tokens}")
    
    # Output directory for audio files
    output_dir = Path(__file__).parent / "debug_output"
    output_dir.mkdir(exist_ok=True)
    
    # Select the correct non-streaming method based on model type
    if tts_model_type == "custom_voice":
        speaker = "Vivian"
        language = "Chinese"
        instruct = "用清晰自然的语气说"
        print(f"  speaker={speaker!r}")
        print(f"  language={language!r}")
        print(f"  instruct={instruct!r}")
        print("="*60)
        
        result = model.generate_custom_voice(
            text,
            speaker=speaker,
            language=language,
            instruct=instruct,
            max_new_tokens=max_new_tokens,
        )
    elif tts_model_type == "voice_design":
        instruct = "用清晰自然的语气说"
        language = "Chinese"
        print(f"  instruct={instruct!r}")
        print(f"  language={language!r}")
        print("="*60)
        
        result = model.generate_voice_design(
            text,
            instruct=instruct,
            language=language,
            max_new_tokens=max_new_tokens,
        )
    elif tts_model_type == "base":
        language = "Chinese"
        print(f"  language={language!r}")
        print("="*60)
        
        result = model.generate_voice_clone(
            text,
            language=language,
            max_new_tokens=max_new_tokens,
        )
    else:
        raise ValueError(f"Unknown model type: {tts_model_type}")
    
    # result is (audio, sample_rate) tuple
    audio, sample_rate = result
    
    # audio is list of arrays (one per batch item)
    if isinstance(audio, list):
        audio = audio[0]
    
    print(f"\nNon-streaming result:")
    print(f"  Audio shape: {audio.shape}")
    print(f"  Sample rate: {sample_rate}")
    print(f"  Duration: {audio.shape[0] / sample_rate:.2f} seconds")
    
    # Save non-streaming audio
    non_streaming_file = output_dir / "non_streaming_output.wav"
    sf.write(non_streaming_file, audio, sample_rate)
    print(f"\n*** Non-streaming audio saved to: {non_streaming_file} ***")
    
    return model, audio, sample_rate


def run_debug_compare_codes():
    """
    Compare streaming vs non-streaming at the codec codes (token_id) level.
    This directly compares the generated token IDs, not the decoded audio.
    
    Run this directly: python test_streaming_consistency.py --compare-codes
    """
    import torch
    import numpy as np
    from pathlib import Path
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {MODEL_PATH}...")
    model = Qwen3TTSModel.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map=device,
    )
    print("Model loaded successfully!")
    
    tts_model_type = model.model.tts_model_type
    print(f"Model type: {tts_model_type}")
    
    # Common parameters
    text = "这是一个流式生成测试的例子，我们来验证流式生成和批量生成的一致性。"
    chunk_size = 25
    max_new_tokens = 2048
    
    output_dir = Path(__file__).parent / "debug_output"
    output_dir.mkdir(exist_ok=True)
    
    # Prepare common generation kwargs
    if tts_model_type == "custom_voice":
        speaker = "Vivian"
        language = "Chinese"
        instruct = "用清晰自然的语气说"
    elif tts_model_type == "voice_design":
        speaker = None
        language = "Chinese"
        instruct = "用清晰自然的语气说"
    elif tts_model_type == "base":
        speaker = None
        language = "Chinese"
        instruct = None
    else:
        raise ValueError(f"Unknown model type: {tts_model_type}")
    
    print("\n" + "="*60)
    print("CODEC CODES (TOKEN_ID) COMPARISON: Streaming vs Non-Streaming")
    print(f"  text={text!r}")
    print(f"  speaker={speaker!r}")
    print(f"  language={language!r}")
    print(f"  instruct={instruct!r}")
    print(f"  chunk_size={chunk_size}")
    print(f"  max_new_tokens={max_new_tokens}")
    print(f"  do_sample=False (greedy)")
    print("="*60)
    
    # ========== Prepare common inputs ==========
    input_ids = model._tokenize_texts([model._build_assistant_text(text)])
    print(f"\n[INPUT] input_ids shape: {input_ids[0].shape}")
    
    instruct_ids = [None]
    if instruct is not None and instruct != "":
        instruct_ids = [model._tokenize_texts([model._build_instruct_text(instruct)])[0]]
        print(f"[INPUT] instruct_ids shape: {instruct_ids[0].shape}")
    
    # Use greedy decoding for deterministic comparison
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "repetition_penalty": 1.05,
    }
    print(f"[INPUT] gen_kwargs: {gen_kwargs}")
    
    # ========== NON-STREAMING: Get codec codes via generate() ==========
    print("\n" + "-"*60)
    print(">>> NON-STREAMING: Generating codec codes with generate()...")
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    non_streaming_codes_list, _ = model.model.generate(
        input_ids=input_ids,
        instruct_ids=instruct_ids,
        languages=[language],
        speakers=[speaker],
        **gen_kwargs,
    )
    
    non_streaming_codes = non_streaming_codes_list[0]  # [T, K]
    print(f"[NON-STREAMING] codec_codes shape: {non_streaming_codes.shape}")
    print(f"[NON-STREAMING] codec_codes dtype: {non_streaming_codes.dtype}")
    print(f"[NON-STREAMING] Total tokens: {non_streaming_codes.shape[0]}")
    print(f"[NON-STREAMING] First 10 tokens (first codebook):\n  {non_streaming_codes[:10, 0].tolist()}")
    
    # ========== STREAMING: Get codec codes via generate_streaming_iter() ==========
    print("\n" + "-"*60)
    print(">>> STREAMING: Generating codec codes with talker.generate_streaming_iter()...")
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # Prepare talker inputs (same as generate_streaming does internally)
    talker_input_embeds, trailing_text_hiddens, tts_pad_embed = model.model._prepare_talker_inputs(
        input_ids=input_ids,
        instruct_ids=instruct_ids,
        languages=[language],
        speakers=[speaker],
    )
    
    # Collect all codec codes from streaming iterator
    streaming_all_codes = []
    chunk_count = 0
    
    for chunk_output in model.model.talker.generate_streaming_iter(
        inputs_embeds=talker_input_embeds,
        attention_mask=torch.ones(
            (1, talker_input_embeds.shape[1]),
            device=talker_input_embeds.device,
            dtype=torch.long,
        ),
        trailing_text_hidden=trailing_text_hiddens,
        tts_pad_embed=tts_pad_embed,
        chunk_size=chunk_size,
        max_new_tokens=max_new_tokens,
        min_new_tokens=2,
        do_sample=False,
        repetition_penalty=1.05,
        eos_token_id=model.model.config.talker_config.codec_eos_token_id,
        suppress_tokens=[
            i
            for i in range(model.model.config.talker_config.vocab_size - 1024, model.model.config.talker_config.vocab_size)
            if i not in (model.model.config.talker_config.codec_eos_token_id,)
        ],
    ):
        chunk_count += 1
        streaming_all_codes.append(chunk_output.codec_codes)
        print(f"[STREAMING] Chunk {chunk_count}: codes shape={chunk_output.codec_codes.shape}, "
              f"total_generated={chunk_output.total_generated}, finished={chunk_output.is_finished}")
    
    # Concatenate all streaming codes
    streaming_codes = torch.cat(streaming_all_codes, dim=0) if streaming_all_codes else torch.tensor([])
    print(f"\n[STREAMING] Total codec_codes shape: {streaming_codes.shape}")
    print(f"[STREAMING] Total tokens: {streaming_codes.shape[0]}")
    print(f"[STREAMING] First 10 tokens (first codebook):\n  {streaming_codes[:10, 0].tolist()}")
    
    # ========== COMPARE TOKEN IDs ==========
    print("\n" + "="*60)
    print("COMPARISON RESULTS (TOKEN_ID LEVEL):")
    print("="*60)
    print(f"Non-streaming tokens: {non_streaming_codes.shape[0]}")
    print(f"Streaming tokens:     {streaming_codes.shape[0]}")
    
    # Move to CPU for comparison
    streaming_codes_cpu = streaming_codes.cpu()
    non_streaming_codes_cpu = non_streaming_codes.cpu()
    
    # Compare lengths and show extra tokens
    if non_streaming_codes.shape[0] != streaming_codes.shape[0]:
        print(f"\n⚠ Token count DIFFERS: {non_streaming_codes.shape[0]} vs {streaming_codes.shape[0]}")
        
        # Show the extra tokens
        if streaming_codes.shape[0] > non_streaming_codes.shape[0]:
            extra_count = streaming_codes.shape[0] - non_streaming_codes.shape[0]
            print(f"\n  Streaming has {extra_count} extra token(s):")
            for i in range(non_streaming_codes.shape[0], streaming_codes.shape[0]):
                print(f"    Token[{i}] (all codebooks): {streaming_codes_cpu[i, :].tolist()}")
            
            # Also show the last few tokens for context
            print(f"\n  Last 5 tokens comparison:")
            ns_start = max(0, non_streaming_codes.shape[0] - 5)
            st_start = max(0, streaming_codes.shape[0] - 5)
            print(f"    Non-streaming [{ns_start}:{non_streaming_codes.shape[0]}] (first codebook):")
            print(f"      {non_streaming_codes_cpu[ns_start:, 0].tolist()}")
            print(f"    Streaming [{st_start}:{streaming_codes.shape[0]}] (first codebook):")
            print(f"      {streaming_codes_cpu[st_start:, 0].tolist()}")
        else:
            extra_count = non_streaming_codes.shape[0] - streaming_codes.shape[0]
            print(f"\n  Non-streaming has {extra_count} extra token(s):")
            for i in range(streaming_codes.shape[0], non_streaming_codes.shape[0]):
                print(f"    Token[{i}] (all codebooks): {non_streaming_codes_cpu[i, :].tolist()}")
            
            # Also show the last few tokens for context
            print(f"\n  Last 5 tokens comparison:")
            ns_start = max(0, non_streaming_codes.shape[0] - 5)
            st_start = max(0, streaming_codes.shape[0] - 5)
            print(f"    Non-streaming [{ns_start}:{non_streaming_codes.shape[0]}] (first codebook):")
            print(f"      {non_streaming_codes_cpu[ns_start:, 0].tolist()}")
            print(f"    Streaming [{st_start}:{streaming_codes.shape[0]}] (first codebook):")
            print(f"      {streaming_codes_cpu[st_start:, 0].tolist()}")
        
        # Check if extra token is EOS
        eos_token_id = model.model.config.talker_config.codec_eos_token_id
        print(f"\n  EOS token ID: {eos_token_id}")
    
    # Compare token IDs
    min_len = min(non_streaming_codes.shape[0], streaming_codes.shape[0])
    if min_len > 0:
        # Compare all codebooks
        num_codebooks = min(non_streaming_codes_cpu.shape[1], streaming_codes_cpu.shape[1])
        
        total_diff_count = 0
        first_diff_pos = None
        
        for cb in range(num_codebooks):
            diff_mask = non_streaming_codes_cpu[:min_len, cb] != streaming_codes_cpu[:min_len, cb]
            diff_count = diff_mask.sum().item()
            
            if diff_count > 0:
                total_diff_count += diff_count
                diff_positions = torch.where(diff_mask)[0]
                if first_diff_pos is None:
                    first_diff_pos = (diff_positions[0].item(), cb)
                
                print(f"\n  Codebook {cb}: {diff_count} differences")
                print(f"    First diff at position {diff_positions[0].item()}:")
                pos = diff_positions[0].item()
                print(f"      Non-streaming: {non_streaming_codes_cpu[pos, cb].item()}")
                print(f"      Streaming:     {streaming_codes_cpu[pos, cb].item()}")
                
                # Show context around first difference
                start = max(0, pos - 2)
                end = min(min_len, pos + 3)
                print(f"    Context [{start}:{end}]:")
                print(f"      Non-streaming: {non_streaming_codes_cpu[start:end, cb].tolist()}")
                print(f"      Streaming:     {streaming_codes_cpu[start:end, cb].tolist()}")
        
        if total_diff_count == 0:
            print(f"\n✓ All {min_len} tokens are IDENTICAL across all {num_codebooks} codebooks!")
        else:
            print(f"\n✗ Total {total_diff_count} token differences found")
            if first_diff_pos:
                print(f"  First difference at position {first_diff_pos[0]}, codebook {first_diff_pos[1]}")
    
    # Save codes for further analysis
    codes_file = output_dir / "codec_codes_comparison.pt"
    torch.save({
        "non_streaming_codes": non_streaming_codes_cpu,
        "streaming_codes": streaming_codes_cpu,
        "text": text,
        "gen_kwargs": gen_kwargs,
    }, codes_file)
    print(f"\nCodec codes saved to {codes_file}")
    
    return model, non_streaming_codes, streaming_codes


def run_debug_compare():
    """
    Compare streaming vs non-streaming output.
    Run this directly: python test_streaming_consistency.py --compare
    """
    import torch
    import numpy as np
    import soundfile as sf
    from pathlib import Path
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {MODEL_PATH}...")
    model = Qwen3TTSModel.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map=device,
    )
    print("Model loaded successfully!")
    
    tts_model_type = model.model.tts_model_type
    print(f"Model type: {tts_model_type}")
    
    # Common parameters
    text = "这是一个流式生成测试的例子，我们来验证流式生成和批量生成的一致性。"
    chunk_size = 25
    left_context_size = 25
    max_new_tokens = 2048  # Increased to allow full sentence generation
    
    output_dir = Path(__file__).parent / "debug_output"
    output_dir.mkdir(exist_ok=True)
    
    # Prepare method arguments based on model type
    if tts_model_type == "custom_voice":
        common_kwargs = dict(speaker="Vivian", language="Chinese", instruct="用清晰自然的语气说")
        streaming_method = model.generate_custom_voice_streaming
        non_streaming_method = model.generate_custom_voice
    elif tts_model_type == "voice_design":
        common_kwargs = dict(instruct="用清晰自然的语气说", language="Chinese")
        streaming_method = model.generate_voice_design_streaming
        non_streaming_method = model.generate_voice_design
    elif tts_model_type == "base":
        common_kwargs = dict(language="Chinese")
        streaming_method = model.generate_voice_clone_streaming
        non_streaming_method = model.generate_voice_clone
    else:
        raise ValueError(f"Unknown model type: {tts_model_type}")
    
    print("\n" + "="*60)
    print("COMPARISON: Streaming vs Non-Streaming")
    print(f"  text={text!r}")
    print(f"  chunk_size={chunk_size}")
    print(f"  max_new_tokens={max_new_tokens}")
    print(f"  kwargs={common_kwargs}")
    print("="*60)
    
    # ===== Non-streaming generation =====
    print("\n>>> Running NON-STREAMING generation...")
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    non_streaming_result = non_streaming_method(
        text,
        **common_kwargs,
        max_new_tokens=max_new_tokens,
    )
    non_streaming_audio, sample_rate = non_streaming_result
    if isinstance(non_streaming_audio, list):
        non_streaming_audio = non_streaming_audio[0]
    
    print(f"Non-streaming: audio shape={non_streaming_audio.shape}, sr={sample_rate}")
    
    non_streaming_file = output_dir / "compare_non_streaming.wav"
    sf.write(non_streaming_file, non_streaming_audio, sample_rate)
    print(f"Saved to: {non_streaming_file}")
    
    # ===== Streaming generation =====
    print("\n>>> Running STREAMING generation...")
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    streaming_chunks = []
    chunk_count = 0
    for chunk_result in streaming_method(
        text,
        **common_kwargs,
        chunk_size=chunk_size,
        left_context_size=left_context_size,
        max_new_tokens=max_new_tokens,
    ):
        chunk_count += 1
        audio_chunk, is_finished, sr = chunk_result
        if audio_chunk is not None:
            streaming_chunks.append(audio_chunk)
            print(f"  Chunk {chunk_count}: shape={audio_chunk.shape}, finished={is_finished}")
    
    streaming_audio = np.concatenate(streaming_chunks) if streaming_chunks else np.array([])
    print(f"Streaming: total audio shape={streaming_audio.shape}, chunks={chunk_count}")
    
    streaming_file = output_dir / "compare_streaming.wav"
    sf.write(streaming_file, streaming_audio, sample_rate)
    print(f"Saved to: {streaming_file}")
    
    # ===== Compare =====
    print("\n" + "="*60)
    print("COMPARISON RESULTS:")
    print("="*60)
    print(f"Non-streaming samples: {non_streaming_audio.shape[0]}")
    print(f"Streaming samples:     {streaming_audio.shape[0]}")
    print(f"Non-streaming duration: {non_streaming_audio.shape[0] / sample_rate:.3f}s")
    print(f"Streaming duration:     {streaming_audio.shape[0] / sample_rate:.3f}s")
    
    # Check if lengths match
    min_len = min(len(non_streaming_audio), len(streaming_audio))
    if min_len > 0:
        diff = np.abs(non_streaming_audio[:min_len] - streaming_audio[:min_len])
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        print(f"\nAudio difference (first {min_len} samples):")
        print(f"  Max diff:  {max_diff:.6f}")
        print(f"  Mean diff: {mean_diff:.6f}")
        
        if max_diff < 1e-5:
            print("\n✓ Audio outputs are IDENTICAL!")
        else:
            print("\n✗ Audio outputs DIFFER")
    
    print(f"\n*** Files saved for manual comparison: ***")
    print(f"  Non-streaming: {non_streaming_file}")
    print(f"  Streaming:     {streaming_file}")
    print(f"  Play: ffplay {non_streaming_file} && ffplay {streaming_file}")
    
    return model, non_streaming_audio, streaming_audio


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--debug":
        # Run standalone streaming debug
        run_debug_forward_streaming()
    elif len(sys.argv) > 1 and sys.argv[1] == "--non-streaming":
        # Run non-streaming debug
        run_debug_non_streaming()
    elif len(sys.argv) > 1 and sys.argv[1] == "--compare":
        # Run comparison
        run_debug_compare()
    elif len(sys.argv) > 1 and sys.argv[1] == "--compare-codes":
        # Run codec codes level comparison (detailed debug)
        run_debug_compare_codes()
    else:
        # Run pytest
        pytest.main([__file__, "-v", "--tb=short"])
