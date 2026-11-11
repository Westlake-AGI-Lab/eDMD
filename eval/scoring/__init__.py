"""The metrics of the eDMD paper: the five preference metrics of its protocol (HPSv3, HPSv3++, PickScore, Aesthetic
and HPSv2.1) and the two it also reports (ImageReward and CLIP score).

One module per metric, plus ``data`` (weight manifest, images, prompts), ``common`` (shared image conversions) and
``run`` (build a scorer, score in fixed batches, aggregate). See ``eval/README.md``.
"""
