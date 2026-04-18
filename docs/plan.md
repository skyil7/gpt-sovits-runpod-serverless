# GPT-SoVITS RunPod Serverless Implementation Plan

## Goal

Build a RunPod Serverless Queue worker for GPT-SoVITS TTS inference only.

This repository is not a general GPT-SoVITS application wrapper. It will not include
WebUI, training, UVR5, ASR, dataset slicing, labeling, Gradio, or a long-running
FastAPI API server. The runtime entrypoint is a RunPod `handler.py` that receives a
job input, runs TTS inference, and returns generated audio.

## Fixed Scope

- Deployment target: RunPod Serverless Queue endpoint.
- Model family: GPT-SoVITS `v2ProPlus` only.
- Runtime feature: TTS inference only.
- Model configuration: RunPod environment variables, not a Hugging Face manifest.
- Reference audio input: per-request reference voice, with optional auxiliary
  reference voices.
- Output: base64 audio by default, object storage URL optionally for larger audio.

Other GPT-SoVITS versions are intentionally not supported in the first productized
worker. The code should keep a small version abstraction internally, but the allowed
version set should initially contain only `v2ProPlus`.

## Why v2ProPlus Only

GPT-SoVITS versions have different required assets and loading paths. Supporting all
versions would require version-specific validation for GPT weights, SoVITS weights,
vocoder assets, speaker verification assets, supported languages, and output
behavior.

For RunPod Serverless this increases:

- Cold start variance, especially if assets are downloaded at runtime.
- Docker image size, if all assets are baked into one image.
- User configuration errors, because environment variable combinations become
  version-dependent.
- Hub test matrix complexity.

`v2ProPlus` is the intended stable target for this worker. Other versions can be
added later only by explicitly expanding the version resolver, path validation, and
Hub tests.

## Architecture

```text
RunPod job input
  -> handler.py
  -> request validation
  -> reference audio resolution into temp files
  -> GPT-SoVITS TTS pipeline
  -> audio encoding
  -> base64 or object storage URL response
```

The worker should import and call GPT-SoVITS internals directly:

- `GPT_SoVITS.TTS_infer_pack.TTS.TTS_Config`
- `GPT_SoVITS.TTS_infer_pack.TTS.TTS`

Do not start GPT-SoVITS `api_v2.py` through uvicorn. A Queue Serverless worker should
load the model once at worker startup and reuse it across jobs.

## Repository Layout

```text
.
├─ handler.py
├─ Dockerfile
├─ requirements-serverless.txt
├─ test_input.json
├─ README.md
├─ .runpod/
│  ├─ hub.json
│  └─ tests.json
├─ docs/
│  └─ plan.md
├─ src/gpt_sovits_serverless/
│  ├─ settings.py
│  ├─ hf_resolver.py
│  ├─ pipeline.py
│  ├─ request_schema.py
│  ├─ reference_audio.py
│  ├─ audio_encode.py
│  └─ errors.py
└─ tests/
   ├─ test_settings.py
   ├─ test_request_schema.py
   └─ test_reference_audio.py
```

GPT-SoVITS source should be pinned. Prefer cloning a known commit in the Dockerfile
or using a git submodule pinned to a commit. Do not build from floating `main`.

## Environment Variables

### Custom Fine-Tuned Weights

```text
HF_MODEL_REPO_ID=your-org/your-gpt-sovits-model
HF_MODEL_REVISION=main
GPT_WEIGHT_PATH=s1v3.ckpt
SOVITS_WEIGHT_PATH=v2Pro/s2Gv2ProPlus.pth
```

`GPT_WEIGHT_PATH` and `SOVITS_WEIGHT_PATH` are relative to the Hugging Face snapshot
root for `HF_MODEL_REPO_ID`.

### Pretrained/Base Assets

```text
HF_PRETRAINED_REPO_ID=lj1995/GPT-SoVITS
HF_PRETRAINED_REVISION=main
BERT_MODEL_PATH=chinese-roberta-wwm-ext-large
CNHUBERT_MODEL_PATH=chinese-hubert-base
SV_MODEL_PATH=sv/pretrained_eres2netv2w24s4ep4.ckpt
```

For production, prefer baking the v2ProPlus base assets into the Docker image where
practical. Runtime Hugging Face download should still be supported for flexibility.

### Runtime

```text
DEVICE=cuda
IS_HALF=true
OUTPUT_MODE=base64
DEFAULT_MEDIA_TYPE=wav
MAX_TEXT_CHARS=1200
MAX_REFERENCE_AUDIO_MB=20
MODEL_CACHE_DIR=/runpod-volume/gpt-sovits
HF_TOKEN=
```

### Object Storage Output

Use this when audio can exceed RunPod response payload limits.

```text
OUTPUT_MODE=s3
S3_ENDPOINT_URL=
S3_BUCKET=
S3_ACCESS_KEY_ID=
S3_SECRET_ACCESS_KEY=
S3_PUBLIC_BASE_URL=
```

## Hugging Face Model Repo Layout

No manifest is required. The model repo only needs files matching the configured
relative paths.

Example:

```text
.
├─ s1v3.ckpt
├─ v2Pro/
│  └─ s2Gv2ProPlus.pth
└─ references/
   └─ default.wav
```

Example environment:

```text
HF_MODEL_REPO_ID=aeryai/example-voice-v2proplus
GPT_WEIGHT_PATH=s1v3.ckpt
SOVITS_WEIGHT_PATH=v2Pro/s2Gv2ProPlus.pth
```

## RunPod Request Schema

```json
{
  "input": {
    "text": "안녕하세요. 이 문장을 합성합니다.",
    "text_lang": "ko",
    "reference_voice": {
      "source": "base64",
      "data": "UklGR...",
      "media_type": "wav",
      "prompt_text": "참조 음성 원문입니다.",
      "prompt_lang": "ko"
    },
    "aux_reference_voices": [],
    "generation": {
      "top_k": 15,
      "top_p": 1,
      "temperature": 1,
      "text_split_method": "cut5",
      "batch_size": 1,
      "batch_threshold": 0.75,
      "split_bucket": true,
      "speed_factor": 1.0,
      "seed": -1,
      "parallel_infer": true,
      "repetition_penalty": 1.35,
      "sample_steps": 32,
      "super_sampling": false,
      "media_type": "wav"
    }
  }
}
```

Supported `reference_voice.source` values:

- `base64`: request contains base64 audio bytes.
- `url`: worker downloads audio from a URL.
- `hf`: path inside the configured `HF_MODEL_REPO_ID` snapshot.
- `path`: local path inside the container, intended for advanced deployments.

`aux_reference_voices` follows the same source structure. Each resolved reference
audio is written to a per-job temp directory and cleaned up after inference.

## RunPod Response Schema

Default base64 response:

```json
{
  "status": "success",
  "audio": {
    "media_type": "wav",
    "encoding": "base64",
    "data": "UklGR..."
  },
  "meta": {
    "version": "v2ProPlus",
    "inference_ms": 3200
  }
}
```

Object storage response:

```json
{
  "status": "success",
  "audio": {
    "media_type": "wav",
    "encoding": "url",
    "url": "https://..."
  },
  "meta": {
    "version": "v2ProPlus",
    "inference_ms": 3200
  }
}
```

## Implementation Steps

1. Add `settings.py`.
   - Parse environment variables.
   - Validate booleans, paths, limits, and output mode.
   - Keep `SUPPORTED_VERSIONS = {"v2ProPlus"}`.

2. Add `hf_resolver.py`.
   - Resolve Hugging Face snapshots with `huggingface_hub.snapshot_download`.
   - Support `HF_TOKEN`.
   - Validate relative paths and block path traversal.
   - Cache under `MODEL_CACHE_DIR`.

3. Add `pipeline.py`.
   - Build a `TTS_Config` dictionary for v2ProPlus.
   - Resolve GPT weight, SoVITS weight, BERT, CNHuBERT, and SV assets.
   - Initialize `TTS` once at worker startup or lazily on first request.

4. Add `request_schema.py`.
   - Validate `text`, `text_lang`, `reference_voice`, and generation parameters.
   - Apply defaults matching GPT-SoVITS `api_v2.py`.
   - Enforce `MAX_TEXT_CHARS`.

5. Add `reference_audio.py`.
   - Resolve `base64`, `url`, `hf`, and `path` references.
   - Enforce `MAX_REFERENCE_AUDIO_MB`.
   - Write request-scoped temp files.

6. Add `audio_encode.py`.
   - Convert generated numpy audio to `wav`, `ogg`, `aac`, or `raw`.
   - Return base64 by default.
   - Upload to S3-compatible storage when configured.

7. Add `handler.py`.
   - Start RunPod serverless with `runpod.serverless.start({"handler": handler})`.
   - Validate input.
   - Run inference.
   - Return structured success/error output.

8. Add `Dockerfile`.
   - Use a CUDA PyTorch RunPod-compatible base image.
   - Install system dependencies: `ffmpeg`, `libsox-dev`, git, build essentials as
     needed.
   - Clone GPT-SoVITS at a pinned commit.
   - Install GPT-SoVITS inference dependencies and `requirements-serverless.txt`.
   - Set `CMD ["python", "-u", "handler.py"]`.

9. Add RunPod Hub files.
   - `.runpod/hub.json`: `type=serverless`, `category=audio`, GPU config, env
     inputs for model repo and weight paths.
   - `.runpod/tests.json`: short smoke test with a public test model and reference.

10. Add tests.
    - Unit tests for env parsing, path validation, request validation, and reference
      audio decoding.
    - GPU integration test can be documented for RunPod rather than run in local CI.

## RunPod Hub Requirements

RunPod Hub requires:

- `handler.py`
- `Dockerfile`
- `README.md`
- `.runpod/hub.json`
- `.runpod/tests.json`
- A GitHub release, because Hub indexes releases rather than arbitrary commits.

The Hub listing should expose the model-related environment variables directly. The
description should state that this worker is GPT-SoVITS v2ProPlus TTS inference only.

## Payload Limits

RunPod response payload limits can be reached with longer WAV outputs, especially
when base64 encoded. The default base64 response is intended for short TTS outputs.
For longer generation, use object storage output mode and return a URL.

## Non-Goals

- WebUI.
- Gradio.
- GPT-SoVITS training.
- Dataset preparation.
- UVR5.
- ASR.
- Voice separation.
- Long-running FastAPI server.
- Multi-version GPT-SoVITS runtime switching.
- Streaming audio in the first implementation.

