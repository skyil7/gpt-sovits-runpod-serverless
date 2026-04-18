# GPT-SoVITS RunPod Serverless

RunPod Serverless Queue worker for GPT-SoVITS `v2ProPlus` TTS inference.

This repository is intentionally inference-only. It does not run GPT-SoVITS WebUI,
Gradio, FastAPI, training, ASR, UVR5, slicing, labeling, or dataset tools.

## Request

Send jobs to the RunPod endpoint with a text payload and a per-request reference
voice.

```json
{
  "input": {
    "text": "Hello from RunPod.",
    "text_lang": "en",
    "reference_voice": {
      "source": "url",
      "url": "https://raw.githubusercontent.com/skyil7/gpt-sovits-runpod-serverless/main/examples/reference_mlk_5s.wav",
      "media_type": "wav",
      "prompt_text": "I have a dream",
      "prompt_lang": "en"
    },
    "aux_reference_voices": [],
    "generation": {
      "media_type": "wav",
      "text_split_method": "cut5",
      "batch_size": 1,
      "parallel_infer": true
    }
  }
}
```

Supported reference sources:

- `base64`: `data` contains base64 audio bytes.
- `url`: `url` or `data` contains an HTTP(S) audio URL.
- `hf`: `path` points to a file inside `HF_MODEL_REPO_ID`, or `HF_PRETRAINED_REPO_ID` when no custom repo is set.
- `path`: local container path. Disabled unless `ALLOW_LOCAL_REFERENCE_PATHS=true`.

Reference audio accepts common input container types such as `wav`, `flac`, `mp3`,
`ogg`, `aac`, `m4a`, and `webm`. Generated output is controlled separately by
`generation.media_type`.

GPT-SoVITS requires the primary reference audio to be between 3 and 10 seconds.

Supported languages are GPT-SoVITS v2 languages: `auto`, `auto_yue`, `en`, `zh`,
`ja`, `yue`, `ko`, `all_zh`, `all_ja`, `all_yue`, and `all_ko`.

## Response

Default `OUTPUT_MODE=base64` returns:

```json
{
  "status": "success",
  "audio": {
    "media_type": "wav",
    "content_type": "audio/wav",
    "encoding": "base64",
    "data": "UklGR..."
  },
  "meta": {
    "version": "v2ProPlus",
    "sample_rate": 32000,
    "inference_ms": 3200
  }
}
```

Use `OUTPUT_MODE=s3` for longer outputs that may exceed RunPod response payload
limits.

## Environment

Custom fine-tuned model weights are configured through environment variables, not a
Hugging Face manifest.

```text
HF_MODEL_REPO_ID=your-org/your-gpt-sovits-model
HF_MODEL_REVISION=main
GPT_WEIGHT_PATH=s1v3.ckpt
SOVITS_WEIGHT_PATH=v2Pro/s2Gv2ProPlus.pth
```

Base v2ProPlus assets are resolved separately:

```text
HF_PRETRAINED_REPO_ID=lj1995/GPT-SoVITS
HF_PRETRAINED_REVISION=main
BASE_GPT_WEIGHT_PATH=s1v3.ckpt
BASE_SOVITS_WEIGHT_PATH=v2Pro/s2Gv2ProPlus.pth
BERT_MODEL_PATH=chinese-roberta-wwm-ext-large
CNHUBERT_MODEL_PATH=chinese-hubert-base
SV_MODEL_PATH=sv/pretrained_eres2netv2w24s4ep4.ckpt
```

Runtime settings:

```text
MODEL_VERSION=v2ProPlus
GPT_SOVITS_ROOT=/opt/GPT-SoVITS
DEVICE=cuda
IS_HALF=true
LOAD_MODEL_AT_STARTUP=true
OUTPUT_MODE=base64
DEFAULT_MEDIA_TYPE=wav
MAX_TEXT_CHARS=1200
MAX_REFERENCE_AUDIO_MB=20
MODEL_CACHE_DIR=/runpod-volume/gpt-sovits
HF_TOKEN=
```

S3-compatible output:

```text
OUTPUT_MODE=s3
S3_ENDPOINT_URL=
S3_BUCKET=
S3_ACCESS_KEY_ID=
S3_SECRET_ACCESS_KEY=
S3_REGION=
S3_PUBLIC_BASE_URL=
```

## Build

```bash
docker build -t gpt-sovits-runpod-serverless .
```

The Dockerfile pins GPT-SoVITS to commit
`2d9193b0d3c0eae0c3a14d8c68a839f1bae157dc`. Update `GPT_SOVITS_REF` deliberately
when upgrading the upstream runtime.

## Local Unit Tests

The unit tests validate the serverless wrapper without loading GPT-SoVITS models.

```bash
python -m pip install -r requirements-dev.txt
python -m pytest
```

GPU inference should be verified inside a RunPod worker or a matching CUDA
container, because the model load requires GPT-SoVITS dependencies and v2ProPlus
weights.
