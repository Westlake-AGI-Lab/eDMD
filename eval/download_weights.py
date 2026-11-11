#!/usr/bin/env python3
"""Download the scorer weights from the Hugging Face Hub and write the manifest the scorers read.

    python eval/download_weights.py [--dir eval/weights] [--metrics hpsv3 pickscore ...]

Only the files the scorers load are fetched, at pinned revisions: about 82 GB for all seven metrics, 70 GB of it
HPSv3 and HPSv3++ with their Qwen-VL bases. Writes ``<dir>/manifest.json`` (``{repo id: directory}``, merged with an
existing one); a ``--dir`` other than the default needs ``export EDMD_SCORER_MANIFEST=<dir>/manifest.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from scoring.data import DEFAULT_MANIFEST, MANIFEST_ENVS, METRIC_REPOS, METRICS  # noqa: E402

TOKENIZER = ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"]

#: repo id -> (revision, files the scorers load).
FILES = {
    "MizzenAI/HPSv3": ("4f81e3e09edd82fe3c5f636444c721b592a735ca", ["HPSv3.safetensors"]),
    "Qwen/Qwen2-VL-7B-Instruct": ("eed13092ef92e448dd6875b2a00151bd3f7db0ac", [
        "config.json", "generation_config.json", "model.safetensors.index.json",
        *[f"model-{i:05d}-of-00005.safetensors" for i in range(1, 6)],
        "preprocessor_config.json", "chat_template.json", *TOKENIZER]),
    "Junjun2333/HPSv3-PlusPlus": ("3eb759987639ebeaa04de4b3b8786bb14611309a", ["hpsv3++.pth"]),
    "Qwen/Qwen3-VL-8B-Instruct": ("0c351dd01ed87e9c1b53cbc748cba10e6187ff3b", [
        "config.json", "generation_config.json", "model.safetensors.index.json",
        *[f"model-{i:05d}-of-00004.safetensors" for i in range(1, 5)],
        "preprocessor_config.json", "video_preprocessor_config.json", "chat_template.json", *TOKENIZER]),
    # PickScore: the model from PickScore_v1, the processor from CLIP-H
    "yuvalkirstain/PickScore_v1": ("a4e4367c6dfa7288a00c550414478f865b875800", ["config.json", "model.safetensors"]),
    "laion/CLIP-ViT-H-14-laion2B-s32B-b79K": ("1c2b8495b28150b8a4922ee1c8edee224c284c0c", [
        "config.json", "preprocessor_config.json", *TOKENIZER, "special_tokens_map.json",
        "open_clip_pytorch_model.bin"]),            # the OpenCLIP backbone HPSv2.1 is created from
    "openai/clip-vit-large-patch14": ("32bd64288804d66eefd0ccbe215aa642df71cc41", [
        "config.json", "model.safetensors", "preprocessor_config.json", *TOKENIZER, "special_tokens_map.json"]),
    "trl-lib/ddpo-aesthetic-predictor": ("cf4bd10345610935a80a5d3be9a3133c50f121e4", ["aesthetic-model.pth"]),
    "xswu/HPSv2": ("697403c78157020a1ae59d23f111aa58ced35b0a", ["HPS_v2.1_compressed.pt"]),
    # ImageReward: the checkpoint and BLIP text-encoder config, and BLIP's bert-base-uncased tokenizer
    "THUDM/ImageReward": ("5736be03b2652728fb87788c9797b0570450ab72", ["ImageReward.pt", "med_config.json"]),
    "google-bert/bert-base-uncased": ("86b5e0934494bd15c9632b12f734a8a67f723594", [
        "config.json", "tokenizer.json", "tokenizer_config.json", "vocab.txt"]),
    "openai/clip-vit-base-patch32": ("3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268", [
        "config.json", "pytorch_model.bin", "preprocessor_config.json", *TOKENIZER, "special_tokens_map.json"]),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", type=Path, default=DEFAULT_MANIFEST.parent, help="where the weights go")
    parser.add_argument("--metrics", nargs="+", choices=METRICS, default=list(METRICS))
    args = parser.parse_args()

    from huggingface_hub import snapshot_download

    root = args.dir.expanduser().resolve()
    manifest_file = root / "manifest.json"
    manifest = json.loads(manifest_file.read_text(encoding="utf-8")) if manifest_file.is_file() else {}
    repos = list(dict.fromkeys(repo for metric in args.metrics for repo in METRIC_REPOS[metric]))
    for repo in repos:
        revision, files = FILES[repo]
        target = root / repo.replace("/", "--")
        print(f"{repo}@{revision[:10]}: {len(files)} files -> {target}", flush=True)
        snapshot_download(repo, revision=revision, allow_patterns=files, local_dir=target)
        missing = [name for name in files if not (target / name).is_file()]
        if missing:
            sys.exit(f"{repo}: missing after download: {missing}")
        manifest[repo] = str(target)
    root.mkdir(parents=True, exist_ok=True)
    temporary = manifest_file.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(manifest_file)
    print(f"manifest: {manifest_file}")
    if manifest_file != DEFAULT_MANIFEST.resolve():
        print(f"export {MANIFEST_ENVS[0]}={manifest_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
