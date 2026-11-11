"""Prompt data for distillation.

Two prompt-file formats, chosen by suffix:

``.txt``
    One prompt per line, CDM ``mixed_200k`` style. Lines are stripped; blank lines and duplicates are kept, as in CDM.
``.jsonl``
    One JSON object per line with a string ``prompt`` field, for prompts that contain newlines or are themselves JSON
    (a rewritten Qwen prompt, say). Extra fields are kept as per-sample metadata. The output of the Qwen-Image-2.1
    prompt enhancer works as is: ``positive_prompt`` stands in for ``prompt``, and its ``parse_ok: false`` rows are
    skipped.
"""

from __future__ import annotations

import json
import os
from collections import deque
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset, Sampler


class PromptFileDataset(Dataset):
    """Prompts from a ``.txt`` (one per line) or ``.jsonl`` (``{"prompt": ...}``) file."""

    def __init__(self, path: str | os.PathLike[str], max_samples: int | None = -1) -> None:
        self.file_path = str(path)
        suffix = Path(self.file_path).suffix
        self.metadata: list[dict[str, Any]] = []
        if suffix == ".txt":
            with open(self.file_path, "r", encoding="utf-8") as handle:
                self.prompts = [line.strip() for line in handle.readlines()]
            self.metadata = [{} for _ in self.prompts]
        elif suffix == ".jsonl":
            self.prompts = []
            with open(self.file_path, "r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    if isinstance(record, dict) and record.get("parse_ok") is False:   # an enhancer failure
                        continue
                    prompt = record.get("prompt", record.get("positive_prompt")) if isinstance(record, dict) else None
                    if not isinstance(prompt, str):
                        raise ValueError(f"{self.file_path}:{line_number} has no string 'prompt' field.")
                    self.prompts.append(prompt)
                    self.metadata.append({key: value for key, value in record.items()
                                          if key not in ("prompt", "positive_prompt", "thinking")})
        else:
            raise ValueError(f"Prompt files must be .txt or .jsonl, got {self.file_path}.")
        if max_samples is not None and max_samples > 0 and len(self.prompts) > max_samples:
            self.prompts = self.prompts[:max_samples]
            self.metadata = self.metadata[:max_samples]
        if not self.prompts:
            raise ValueError(f"No prompts in {self.file_path}.")

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return {"prompt": self.prompts[idx], "metadata": self.metadata[idx]}

    @staticmethod
    def collate_fn(examples: Sequence[dict[str, Any]]) -> tuple[list[str], list[dict[str, Any]]]:
        """Collate samples as ``(prompts, metadata)``, matching CDM."""
        return [example["prompt"] for example in examples], [example["metadata"] for example in examples]

    def statistics(self) -> dict[str, int | float]:
        return prompt_statistics(self.prompts)


class DistributedKRepeatSampler(Sampler):
    """CDM's infinite distributed, globally K-repeated batch sampler, with consumption tracking for exact resume.

    A global batch of ``num_replicas * batch_size`` samples takes ``m = num_replicas * batch_size // k`` base prompts
    from a shuffled without-replacement queue, repeats each ``k`` times, shuffles, and gives each rank one contiguous
    ``batch_size`` shard. It yields forever (no ``__len__``): callers bound the iteration.
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        k: int,
        num_replicas: int,
        rank: int,
        seed: int = 0,
        track_consumption: bool = False,
    ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.k = k
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        if len(dataset) == 0 or min(batch_size, k, num_replicas) <= 0:
            raise ValueError("The dataset and batch/repeat/replica counts must be non-empty and positive.")
        if not 0 <= rank < num_replicas:
            raise ValueError("rank must be in [0, num_replicas).")

        self.total_samples = self.num_replicas * self.batch_size
        assert self.total_samples % self.k == 0, (
            f"k can not div n*b, k{k}-num_replicas{num_replicas}"
            f"-batch_size{batch_size}"
        )
        self.m = self.total_samples // self.k
        self.epoch = 0

        # Stateful without-replacement queue, kept identical to CDM.
        self._shuffled_indices: list[int] | None = None
        self._pointer = 0
        self._shuffle_epoch = 0
        self.track_consumption = track_consumption
        self._pending_states: deque[dict[str, Any]] = deque()
        self._consumed_state = self._snapshot()

    def _snapshot(self) -> dict[str, Any]:
        # Queues are replaced, never mutated, so prefetch snapshots share the list instead of copying it per batch.
        return {
            "indices": self._shuffled_indices,
            "pointer": self._pointer,
            "shuffle_epoch": self._shuffle_epoch,
            "epoch": self.epoch,
        }

    def mark_batch_consumed(self) -> None:
        """Commit one completed training batch, excluding worker prefetch."""
        if not self.track_consumption:
            raise RuntimeError("Enable track_consumption to commit training batches.")
        if not self._pending_states:
            raise RuntimeError("No yielded batch is available to commit.")
        self._consumed_state = self._pending_states.popleft()

    def _identity(self) -> dict[str, int]:
        return {
            "dataset_size": len(self.dataset), "batch_size": self.batch_size,
            "k": self.k, "num_replicas": self.num_replicas,
            "rank": self.rank, "seed": self.seed,
        }

    def state_dict(self) -> dict[str, Any]:
        state = self._consumed_state if self.track_consumption else self._snapshot()
        return {"identity": self._identity(), **state}

    def load_state_dict(self, state: dict[str, Any], *, strict: bool = True) -> None:
        """``strict``: the same replica layout too. Otherwise only the data identity must match: the queue is global,
        so a run on another number of replicas continues it from the same position."""
        identity, ours = dict(state["identity"]), self._identity()
        if not strict:
            for key in ("num_replicas", "rank"):
                identity.pop(key, None)
                ours.pop(key, None)
        if identity != ours:
            raise ValueError("Checkpoint sampler configuration does not match this run.")
        self._shuffled_indices = state["indices"]
        self._pointer = int(state["pointer"])
        self._shuffle_epoch = int(state["shuffle_epoch"])
        self.epoch = int(state["epoch"])
        self._pending_states.clear()
        self._consumed_state = self._snapshot()

    def _get_shuffled_indices(self, generator: torch.Generator) -> list[int]:
        dataset_size = len(self.dataset)
        return torch.randperm(dataset_size, generator=generator).tolist()

    def __iter__(self):
        while True:
            generator = torch.Generator()
            dataset_size = len(self.dataset)

            # As in CDM, a queue that cannot supply m more base prompts is replaced; its tail is not carried over.
            if (
                self._shuffled_indices is None
                or self._pointer + self.m > len(self._shuffled_indices)
            ):
                generator.manual_seed(self.seed + self._shuffle_epoch)
                self._shuffled_indices = self._get_shuffled_indices(generator)
                self._pointer = 0
                self._shuffle_epoch += 1

            if dataset_size >= self.m:
                indices = self._shuffled_indices[
                    self._pointer : self._pointer + self.m
                ]
                self._pointer += self.m
            else:
                # A dataset smaller than m: exhaust and reshuffle the queue as often as this global batch needs.
                indices = []
                remaining = self.m
                while remaining > 0:
                    available = len(self._shuffled_indices) - self._pointer
                    if available <= 0:
                        generator.manual_seed(self.seed + self._shuffle_epoch)
                        self._shuffled_indices = self._get_shuffled_indices(
                            generator
                        )
                        self._pointer = 0
                        self._shuffle_epoch += 1
                        available = len(self._shuffled_indices)

                    take = min(remaining, available)
                    indices.extend(
                        self._shuffled_indices[
                            self._pointer : self._pointer + take
                        ]
                    )
                    self._pointer += take
                    remaining -= take

            repeated_indices = [
                idx for idx in indices for _ in range(self.k)
            ]

            generator.manual_seed(self.seed + self.epoch)
            shuffled_order = torch.randperm(
                len(repeated_indices), generator=generator
            ).tolist()
            shuffled_samples = [
                repeated_indices[index] for index in shuffled_order
            ]

            per_card_samples = []
            for replica_index in range(self.num_replicas):
                start = replica_index * self.batch_size
                end = start + self.batch_size
                per_card_samples.append(shuffled_samples[start:end])

            self.epoch += 1
            if self.track_consumption:
                self._pending_states.append(self._snapshot())
            yield per_card_samples[self.rank]

    def set_epoch(self, epoch: int) -> None:
        """Set the seed epoch of the K-repeat shuffle; as in CDM, the base-index queue is not rewound."""
        self.epoch = epoch

    def get_progress(self) -> tuple[int, int, int]:
        """Return ``(queue_pointer, dataset_size, completed_shuffle_epoch)``."""

        dataset_size = len(self.dataset)
        return self._pointer, dataset_size, self._shuffle_epoch - 1


def prompt_statistics(prompts: Sequence[str]) -> dict[str, int | float]:
    """Prompt counts and character-length statistics (duplicates and empty prompts are reported, not filtered)."""
    num_prompts = len(prompts)
    num_unique = len(set(prompts))
    lengths = [len(prompt) for prompt in prompts]

    return {
        "num_prompts": num_prompts,
        "num_unique": num_unique,
        "num_duplicates": num_prompts - num_unique,
        "num_empty": sum(not prompt for prompt in prompts),
        "min_chars": min(lengths, default=0),
        "max_chars": max(lengths, default=0),
        "mean_chars": (
            float(sum(lengths)) / num_prompts if num_prompts else 0.0
        ),
    }


__all__ = ["DistributedKRepeatSampler", "PromptFileDataset", "prompt_statistics"]
