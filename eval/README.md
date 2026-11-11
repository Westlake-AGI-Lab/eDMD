# Evaluation metrics

The reward models of the paper's DrawBench-1000 evaluation, one module per metric under `scoring/`. Their formulas and
preprocessing follow [CDM](https://github.com/byliutao/CDM)'s evaluation code (MIT License, see [`NOTICE`](NOTICE)).

| Metric | Module | Official repository | Weights (Hugging Face) | Batch |
|---|---|---|---|---|
| HPSv3++ | [`scoring/hpsv3pp.py`](scoring/hpsv3pp.py) | [PlantPotatoOnMoon/HPSv3-PlusPlus](https://github.com/PlantPotatoOnMoon/HPSv3-PlusPlus) | [Junjun2333/HPSv3-PlusPlus](https://huggingface.co/Junjun2333/HPSv3-PlusPlus), [Qwen/Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) | 1 |
| HPSv3 | [`scoring/hpsv3.py`](scoring/hpsv3.py) | [MizzenAI/HPSv3](https://github.com/MizzenAI/HPSv3) | [MizzenAI/HPSv3](https://huggingface.co/MizzenAI/HPSv3), [Qwen/Qwen2-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct) | 1 |
| PickScore | [`scoring/pickscore.py`](scoring/pickscore.py) | [yuvalkirstain/PickScore](https://github.com/yuvalkirstain/PickScore) | [yuvalkirstain/PickScore_v1](https://huggingface.co/yuvalkirstain/PickScore_v1), [laion/CLIP-ViT-H-14-laion2B-s32B-b79K](https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K) | 8 |
| Aesthetic | [`scoring/aesthetic.py`](scoring/aesthetic.py) | [christophschuhmann/improved-aesthetic-predictor](https://github.com/christophschuhmann/improved-aesthetic-predictor) | [openai/clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14), [trl-lib/ddpo-aesthetic-predictor](https://huggingface.co/trl-lib/ddpo-aesthetic-predictor) | 8 |
| HPSv2.1 | [`scoring/hpsv2.py`](scoring/hpsv2.py) | [tgxs002/HPSv2](https://github.com/tgxs002/HPSv2) | [xswu/HPSv2](https://huggingface.co/xswu/HPSv2), [laion/CLIP-ViT-H-14-laion2B-s32B-b79K](https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K) | 8 |
| ImageReward | [`scoring/imagereward.py`](scoring/imagereward.py) | [THUDM/ImageReward](https://github.com/THUDM/ImageReward) | [THUDM/ImageReward](https://huggingface.co/THUDM/ImageReward), [google-bert/bert-base-uncased](https://huggingface.co/google-bert/bert-base-uncased) | 8 |
| CLIP score | [`scoring/clip.py`](scoring/clip.py) | [openai/CLIP](https://github.com/openai/CLIP) | [openai/clip-vit-base-patch32](https://huggingface.co/openai/clip-vit-base-patch32) | 8 |

## Install

Two Python 3.10 environments: HPSv3++ (Qwen3-VL) needs `transformers` 4.57, the other metrics 4.45.2.

```bash
python -m venv ~/venvs/edmd-score && ~/venvs/edmd-score/bin/pip install -r eval/requirements-score.txt
~/venvs/edmd-score/bin/pip install --no-deps hpsv3==1.0.0 hpsv2x==1.2.0 trl==0.12.2 image-reward==1.5 fairscale==0.4.13

python -m venv ~/venvs/edmd-hpsv3pp && ~/venvs/edmd-hpsv3pp/bin/pip install -r eval/requirements-hpsv3pp.txt
~/venvs/edmd-hpsv3pp/bin/pip install --no-deps trl==0.12.2
git clone https://github.com/PlantPotatoOnMoon/HPSv3-PlusPlus eval/HPSv3PlusPlus      # HPSv3++ is not on PyPI
git -C eval/HPSv3PlusPlus checkout 6a095f68ee98330bf22365f872ed609bd44a216f
```

## Weights

```bash
~/venvs/edmd-score/bin/python eval/download_weights.py      # ~82 GB into eval/weights/ (--metrics for a subset)
# elsewhere: --dir /path/to/scorer-weights, then export EDMD_SCORER_MANIFEST=/path/to/scorer-weights/manifest.json
```

## Scoring

An evaluation rendered by `python -m edmd.cli.evaluate generate`:

```bash
~/venvs/edmd-score/bin/python -m edmd.cli.evaluate score --dir outputs/eval/<run>
~/venvs/edmd-hpsv3pp/bin/python -m edmd.cli.evaluate score --dir outputs/eval/<run> --metrics hpsv3pp
python -m edmd.cli.evaluate summary --dir outputs/eval/<run>
```

Any directory of images (an image is scored against the prompt line given by the last number in its file name):

```bash
~/venvs/edmd-score/bin/python eval/score_images.py --images <dir> --prompts prompts.txt --out scores.json
~/venvs/edmd-hpsv3pp/bin/python eval/score_images.py --images <dir> --prompts prompts.txt --out scores.json --metrics hpsv3pp
```

For a quick check, render with `--eval drawbench1k_lite`: it scores PickScore and Aesthetic only, in the training
environment (`download_weights.py --metrics pickscore aesthetic`).
