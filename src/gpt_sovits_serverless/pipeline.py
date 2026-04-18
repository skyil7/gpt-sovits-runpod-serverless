from __future__ import annotations

import os
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import InferenceError, ModelLoadError
from .hf_resolver import link_or_copy, resolve_asset_path
from .settings import Settings


SV_EXPECTED_RELATIVE_PATH = "GPT_SoVITS/pretrained_models/sv/pretrained_eres2netv2w24s4ep4.ckpt"


@dataclass(frozen=True)
class ResolvedModelAssets:
    gpt_weight: Path
    sovits_weight: Path
    base_gpt_weight: Path
    base_sovits_weight: Path
    bert_model: Path
    cnhubert_model: Path
    sv_model: Path


class GPTSoVITSPipeline:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._load_lock = threading.RLock()
        self._inference_lock = threading.RLock()
        self._tts = None
        self._assets: ResolvedModelAssets | None = None

    @property
    def loaded(self) -> bool:
        return self._tts is not None

    def load(self) -> None:
        with self._load_lock:
            if self._tts is not None:
                return
            try:
                self._prepare_import_context()
                assets = self._resolve_assets()
                self._prepare_gpt_sovits_runtime_files(assets)
                from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config

                config = TTS_Config(self._tts_config_dict(assets))
                self._tts = TTS(config)
                self._assets = assets
            except Exception as exc:
                raise ModelLoadError(f"Failed to load GPT-SoVITS model: {exc}") from exc

    def synthesize(self, inputs: dict[str, Any]) -> tuple[int, Any]:
        self.load()
        assert self._tts is not None
        with self._inference_lock:
            try:
                generator = self._tts.run(inputs)
                sample_rate, audio = next(generator)
            except StopIteration as exc:
                raise InferenceError("GPT-SoVITS returned no audio") from exc
            except Exception as exc:
                raise InferenceError(f"GPT-SoVITS inference failed: {exc}") from exc
        return int(sample_rate), audio

    def _prepare_import_context(self) -> None:
        root = self.settings.gpt_sovits_root.expanduser().resolve()
        if not root.exists():
            raise ModelLoadError(f"GPT_SOVITS_ROOT does not exist: {root}")
        package_root = root / "GPT_SoVITS"
        if not package_root.exists():
            raise ModelLoadError(f"GPT-SoVITS package directory does not exist: {package_root}")
        for path in (str(root), str(package_root)):
            if path not in sys.path:
                sys.path.insert(0, path)
        os.chdir(root)

    def _resolve_assets(self) -> ResolvedModelAssets:
        custom_repo = self.settings.hf_model_repo_id
        custom_revision = self.settings.hf_model_revision
        pretrained_repo = self.settings.hf_pretrained_repo_id
        pretrained_revision = self.settings.hf_pretrained_revision

        gpt_repo = custom_repo or pretrained_repo
        gpt_revision = custom_revision if custom_repo else pretrained_revision

        return ResolvedModelAssets(
            gpt_weight=resolve_asset_path(
                self.settings.gpt_weight_path,
                repo_id=gpt_repo,
                revision=gpt_revision,
                cache_dir=self.settings.hf_cache_dir,
                token=self.settings.hf_token,
            ),
            sovits_weight=resolve_asset_path(
                self.settings.sovits_weight_path,
                repo_id=gpt_repo,
                revision=gpt_revision,
                cache_dir=self.settings.hf_cache_dir,
                token=self.settings.hf_token,
            ),
            base_gpt_weight=resolve_asset_path(
                self.settings.base_gpt_weight_path,
                repo_id=pretrained_repo,
                revision=pretrained_revision,
                cache_dir=self.settings.hf_cache_dir,
                token=self.settings.hf_token,
            ),
            base_sovits_weight=resolve_asset_path(
                self.settings.base_sovits_weight_path,
                repo_id=pretrained_repo,
                revision=pretrained_revision,
                cache_dir=self.settings.hf_cache_dir,
                token=self.settings.hf_token,
            ),
            bert_model=resolve_asset_path(
                self.settings.bert_model_path,
                repo_id=pretrained_repo,
                revision=pretrained_revision,
                cache_dir=self.settings.hf_cache_dir,
                token=self.settings.hf_token,
            ),
            cnhubert_model=resolve_asset_path(
                self.settings.cnhubert_model_path,
                repo_id=pretrained_repo,
                revision=pretrained_revision,
                cache_dir=self.settings.hf_cache_dir,
                token=self.settings.hf_token,
            ),
            sv_model=resolve_asset_path(
                self.settings.sv_model_path,
                repo_id=pretrained_repo,
                revision=pretrained_revision,
                cache_dir=self.settings.hf_cache_dir,
                token=self.settings.hf_token,
            ),
        )

    def _prepare_gpt_sovits_runtime_files(self, assets: ResolvedModelAssets) -> None:
        expected_sv_path = self.settings.gpt_sovits_root.resolve() / SV_EXPECTED_RELATIVE_PATH
        link_or_copy(assets.sv_model, expected_sv_path)

    def _tts_config_dict(self, assets: ResolvedModelAssets) -> dict[str, dict[str, Any]]:
        base_v2proplus = {
            "device": self.settings.device,
            "is_half": self.settings.is_half,
            "version": "v2ProPlus",
            "t2s_weights_path": str(assets.base_gpt_weight),
            "vits_weights_path": str(assets.base_sovits_weight),
            "bert_base_path": str(assets.bert_model),
            "cnhuhbert_base_path": str(assets.cnhubert_model),
        }
        custom = {
            **base_v2proplus,
            "t2s_weights_path": str(assets.gpt_weight),
            "vits_weights_path": str(assets.sovits_weight),
        }
        return {
            "v2ProPlus": base_v2proplus,
            "custom": custom,
        }
