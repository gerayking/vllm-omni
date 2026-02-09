"""OpenAI-compatible client for Qwen3-TTS via /v1/audio/speech endpoint.

This script demonstrates how to use the OpenAI-compatible speech API
to generate audio from text using Qwen3-TTS models.

Examples:
    # CustomVoice task (predefined speaker)
    python openai_speech_client.py --text "Hello, how are you?" --voice Vivian

    # CustomVoice with emotion instruction
    python openai_speech_client.py --text "I'm so happy!" --voice Vivian \
        --instructions "Speak with excitement"

    # VoiceDesign task (voice from description)
    python openai_speech_client.py --text "Hello world" \
        --task-type VoiceDesign \
        --instructions "A warm, friendly female voice"

    # Base task (voice cloning)
    python openai_speech_client.py --text "Hello world" \
        --task-type Base \
        --ref-audio "https://example.com/reference.wav" \
        --ref-text "This is the reference transcript"

    # Streaming mode (raw audio chunks)
    python openai_speech_client.py --text "Hello, how are you?" --voice Vivian \
        --stream --response-format pcm

    # Streaming mode with SSE format
    python openai_speech_client.py --text "Hello, how are you?" --voice Vivian \
        --stream --stream-format sse
"""

import argparse
import base64
import io
import json
import os
import struct
import time

import httpx

# Default server configuration
DEFAULT_API_BASE = "http://localhost:8000"
DEFAULT_API_KEY = "EMPTY"


def encode_audio_to_base64(audio_path: str) -> str:
    """Encode a local audio file to base64 data URL."""
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Detect MIME type from extension
    audio_path_lower = audio_path.lower()
    if audio_path_lower.endswith(".wav"):
        mime_type = "audio/wav"
    elif audio_path_lower.endswith((".mp3", ".mpeg")):
        mime_type = "audio/mpeg"
    elif audio_path_lower.endswith(".flac"):
        mime_type = "audio/flac"
    elif audio_path_lower.endswith(".ogg"):
        mime_type = "audio/ogg"
    else:
        mime_type = "audio/wav"  # Default

    with open(audio_path, "rb") as f:
        audio_bytes = f.read()
    audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{audio_b64}"


def run_tts_generation(args) -> None:
    """Run TTS generation via OpenAI-compatible /v1/audio/speech API."""

    # Build request payload
    payload = {
        "model": args.model,
        "input": args.text,
        "voice": args.voice,
        "response_format": args.response_format,
    }

    # Add optional parameters
    if args.instructions:
        payload["instructions"] = args.instructions
    if args.task_type:
        payload["task_type"] = args.task_type
    if args.language:
        payload["language"] = args.language
    if args.max_new_tokens:
        payload["max_new_tokens"] = args.max_new_tokens

    # Voice clone parameters (Base task)
    if args.ref_audio:
        if args.ref_audio.startswith(("http://", "https://")):
            payload["ref_audio"] = args.ref_audio
        else:
            payload["ref_audio"] = encode_audio_to_base64(args.ref_audio)
    if args.ref_text:
        payload["ref_text"] = args.ref_text
    if args.x_vector_only:
        payload["x_vector_only_mode"] = True

    print(f"Model: {args.model}")
    print(f"Task type: {args.task_type or 'CustomVoice'}")
    print(f"Text: {args.text}")
    print(f"Voice: {args.voice}")
    print("Generating audio...")

    # Make the API call
    api_url = f"{args.api_base}/v1/audio/speech"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {args.api_key}",
    }

    with httpx.Client(timeout=300.0) as client:
        response = client.post(api_url, json=payload, headers=headers)

    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        print(response.text)
        return

    # Check for JSON error response (only if content is valid UTF-8 text)
    try:
        text = response.content.decode("utf-8")
        if text.startswith('{"error"'):
            print(f"Error: {text}")
            return
    except UnicodeDecodeError:
        pass  # Binary audio data, not an error

    # Save audio response
    output_path = args.output or "tts_output.wav"
    with open(output_path, "wb") as f:
        f.write(response.content)
    print(f"Audio saved to: {output_path}")


def _decode_audio_chunk(chunk_bytes: bytes, response_format: str) -> tuple[bytes, int]:
    """Decode an audio chunk and return raw PCM data and sample rate.

    For PCM format, returns raw bytes directly (assumed 16-bit mono).
    For WAV/FLAC/etc., uses soundfile to decode into PCM samples.

    Returns:
        Tuple of (pcm_int16_bytes, sample_rate).
    """
    if response_format == "pcm":
        # Raw PCM int16, assume default sample rate; caller must track it
        return chunk_bytes, 0

    try:
        import soundfile
    except ImportError:
        raise ImportError(
            "soundfile is required to decode non-PCM streaming chunks. "
            "Install with: pip install soundfile"
        )

    import numpy as np

    audio_data, sr = soundfile.read(io.BytesIO(chunk_bytes), dtype="float32")
    # Convert to int16 PCM
    pcm = np.clip(audio_data, -1.0, 1.0)
    pcm = (pcm * 32767).astype(np.int16)
    return pcm.tobytes(), sr


def _write_wav_file(path: str, pcm_data: bytes, sample_rate: int,
                    num_channels: int = 1, bits_per_sample: int = 16) -> None:
    """Write raw PCM data as a WAV file with proper header."""
    byte_rate = sample_rate * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8
    data_size = len(pcm_data)
    # RIFF header
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + data_size,       # ChunkSize
        b"WAVE",
        b"fmt ",
        16,                   # Subchunk1Size (PCM)
        1,                    # AudioFormat (PCM)
        num_channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        b"data",
        data_size,
    )
    with open(path, "wb") as f:
        f.write(header)
        f.write(pcm_data)


def run_tts_streaming(args) -> None:
    """Run TTS generation with streaming output via /v1/audio/speech API.

    Collects all audio chunks, decodes them into raw PCM, concatenates,
    and writes a single complete WAV file.
    """

    # Build request payload
    payload = {
        "model": args.model,
        "input": args.text,
        "voice": args.voice,
        "response_format": args.response_format,
        "stream": True,
        "stream_format": args.stream_format,
    }

    # Add optional parameters
    if args.instructions:
        payload["instructions"] = args.instructions
    if args.task_type:
        payload["task_type"] = args.task_type
    if args.language:
        payload["language"] = args.language
    if args.max_new_tokens:
        payload["max_new_tokens"] = args.max_new_tokens
    if args.chunk_size:
        payload["chunk_size"] = args.chunk_size
    if args.left_context_size is not None:
        payload["left_context_size"] = args.left_context_size

    # Voice clone parameters (Base task)
    if args.ref_audio:
        if args.ref_audio.startswith(("http://", "https://")):
            payload["ref_audio"] = args.ref_audio
        else:
            payload["ref_audio"] = encode_audio_to_base64(args.ref_audio)
    if args.ref_text:
        payload["ref_text"] = args.ref_text
    if args.x_vector_only:
        payload["x_vector_only_mode"] = True

    print(f"Model: {args.model}")
    print(f"Task type: {args.task_type or 'CustomVoice'}")
    print(f"Text: {args.text}")
    print(f"Voice: {args.voice}")
    print(f"Stream format: {args.stream_format}")
    print(f"Response format: {args.response_format}")
    print("Generating audio (streaming)...")

    api_url = f"{args.api_base}/v1/audio/speech"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {args.api_key}",
    }

    output_path = args.output or "tts_output_streaming.wav"
    use_sse = args.stream_format == "sse"
    response_format = args.response_format or "wav"

    # Create a directory to save individual chunk files
    output_base = os.path.splitext(output_path)[0]
    chunks_dir = f"{output_base}_chunks"
    os.makedirs(chunks_dir, exist_ok=True)
    print(f"Chunk files will be saved to: {chunks_dir}/")

    start_time = time.time()
    first_chunk_time = None
    chunk_count = 0
    total_bytes = 0

    # Collect all decoded PCM data from chunks
    pcm_chunks: list[bytes] = []
    detected_sample_rate: int = 0

    with httpx.Client(timeout=300.0) as client:
        with client.stream("POST", api_url, json=payload, headers=headers) as response:
            if response.status_code != 200:
                print(f"Error: {response.status_code}")
                for chunk in response.iter_bytes():
                    print(chunk.decode("utf-8", errors="replace"))
                return

            if use_sse:
                # SSE mode: parse events, decode base64 audio, extract PCM
                for line in response.iter_lines():
                    if not line.startswith("data: "):
                        continue
                    data_str = line[6:]  # Remove "data: " prefix
                    if data_str == "[DONE]":
                        print("\nStream completed.")
                        break
                    try:
                        event = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    if "error" in event:
                        print(f"\nError: {event['error']}")
                        return

                    if "audio" in event:
                        audio_bytes = base64.b64decode(event["audio"])
                        chunk_count += 1
                        total_bytes += len(audio_bytes)
                        if first_chunk_time is None:
                            first_chunk_time = time.time()

                        # Save individual chunk file (raw encoded audio)
                        chunk_ext = response_format if response_format != "pcm" else "pcm"
                        chunk_file = os.path.join(chunks_dir, f"chunk_{chunk_count:04d}.{chunk_ext}")
                        with open(chunk_file, "wb") as cf:
                            cf.write(audio_bytes)

                        # Use sample_rate from SSE event if available
                        sr_from_event = event.get("sample_rate", 0)

                        # Decode chunk to raw PCM
                        pcm_data, sr = _decode_audio_chunk(
                            audio_bytes, response_format,
                        )
                        if sr > 0:
                            detected_sample_rate = sr
                        elif sr_from_event > 0:
                            detected_sample_rate = sr_from_event
                        pcm_chunks.append(pcm_data)

                        print(
                            f"\rChunk {chunk_count}: "
                            f"{len(audio_bytes)} bytes "
                            f"(PCM: {len(pcm_data)} bytes, "
                            f"total: {total_bytes})",
                            end="", flush=True,
                        )
                        finished = event.get("finished", False)
                        if finished:
                            print("\nStream completed (finished flag).")
                            break
            else:
                # Raw audio mode: collect raw bytes from each chunk
                for chunk in response.iter_bytes(chunk_size=4096):
                    if first_chunk_time is None:
                        first_chunk_time = time.time()
                    chunk_count += 1
                    total_bytes += len(chunk)

                    # Save individual chunk file
                    chunk_ext = response_format if response_format != "pcm" else "pcm"
                    chunk_file = os.path.join(chunks_dir, f"chunk_{chunk_count:04d}.{chunk_ext}")
                    with open(chunk_file, "wb") as cf:
                        cf.write(chunk)

                    # Decode chunk to raw PCM
                    pcm_data, sr = _decode_audio_chunk(chunk, response_format)
                    if sr > 0:
                        detected_sample_rate = sr
                    pcm_chunks.append(pcm_data)

                    print(
                        f"\rChunk {chunk_count}: "
                        f"{len(chunk)} bytes "
                        f"(PCM: {len(pcm_data)} bytes, "
                        f"total: {total_bytes})",
                        end="", flush=True,
                    )
                print()

    # Concatenate all PCM chunks and write as a complete WAV file
    if not pcm_chunks:
        print("No audio data received.")
        return

    all_pcm = b"".join(pcm_chunks)

    # Fallback sample rate if none detected
    if detected_sample_rate <= 0:
        detected_sample_rate = 24000
        print(f"Warning: sample rate not detected, defaulting to {detected_sample_rate} Hz")

    _write_wav_file(output_path, all_pcm, detected_sample_rate)

    elapsed = time.time() - start_time
    ttfa = (first_chunk_time - start_time) if first_chunk_time else elapsed
    num_samples = len(all_pcm) // 2  # int16 = 2 bytes per sample
    duration = num_samples / detected_sample_rate if detected_sample_rate > 0 else 0
    print(f"Audio saved to: {output_path}")
    print(f"Chunk files saved to: {chunks_dir}/ ({chunk_count} files)")
    print(f"Sample rate: {detected_sample_rate} Hz")
    print(f"Audio duration: {duration:.2f}s")
    print(f"Total chunks: {chunk_count}, Total received bytes: {total_bytes}")
    print(f"Total PCM bytes: {len(all_pcm)} ({num_samples} samples)")
    print(f"Time to first audio: {ttfa:.3f}s")
    print(f"Total time: {elapsed:.3f}s")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="OpenAI-compatible client for Qwen3-TTS via /v1/audio/speech",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Server configuration
    parser.add_argument(
        "--api-base",
        type=str,
        default=DEFAULT_API_BASE,
        help=f"API base URL (default: {DEFAULT_API_BASE})",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=DEFAULT_API_KEY,
        help="API key (default: EMPTY)",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        help="Model name/path",
    )

    # Task configuration
    parser.add_argument(
        "--task-type",
        "-t",
        type=str,
        default=None,
        choices=["CustomVoice", "VoiceDesign", "Base"],
        help="TTS task type (default: CustomVoice)",
    )

    # Input text
    parser.add_argument(
        "--text",
        type=str,
        required=True,
        help="Text to synthesize",
    )

    # Voice/speaker
    parser.add_argument(
        "--voice",
        type=str,
        default="Vivian",
        help="Speaker/voice name (default: Vivian). Options: Vivian, Ryan, etc.",
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help="Language: Auto, Chinese, English, etc.",
    )
    parser.add_argument(
        "--instructions",
        type=str,
        default=None,
        help="Voice style/emotion instructions",
    )

    # Base (voice clone) parameters
    parser.add_argument(
        "--ref-audio",
        type=str,
        default=None,
        help="Reference audio file path or URL for voice cloning (Base task)",
    )
    parser.add_argument(
        "--ref-text",
        type=str,
        default=None,
        help="Reference audio transcript for voice cloning (Base task)",
    )
    parser.add_argument(
        "--x-vector-only",
        action="store_true",
        help="Use x-vector only mode for voice cloning (no ICL)",
    )

    # Generation parameters
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Maximum new tokens to generate",
    )

    # Output
    parser.add_argument(
        "--response-format",
        type=str,
        default="wav",
        choices=["wav", "mp3", "flac", "pcm", "aac", "opus"],
        help="Audio output format (default: wav)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output audio file path (default: tts_output.wav)",
    )

    # Streaming parameters
    parser.add_argument(
        "--stream",
        action="store_true",
        default=False,
        help="Enable streaming audio output",
    )
    parser.add_argument(
        "--stream-format",
        type=str,
        default="audio",
        choices=["audio", "sse"],
        help="Stream format: 'audio' for raw bytes, 'sse' for Server-Sent Events (default: audio)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Codec frames per chunk for streaming (default: 5)",
    )
    parser.add_argument(
        "--left-context-size",
        type=int,
        default=None,
        help="Left context frames for streaming (default: 25)",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.stream:
        run_tts_streaming(args)
    else:
        run_tts_generation(args)
