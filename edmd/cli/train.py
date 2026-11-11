"""Train one run under torchrun (``edmd.training.loop``).

    torchrun --standalone --nproc_per_node 8 -m edmd.cli.train --run-file configs/runs/<model>/<method>.yaml \\
        [--set key=value ...] [--resume auto|<checkpoint dir>]

Writes ``<run.output_root>/<run.name>/``: ``config.yaml`` (the resolved config, rewritten at every start), ``logs/``
and ``checkpoint-<N>/`` every ``run.checkpoint_every`` updates. ``--resume auto`` continues from the latest complete
checkpoint there (or starts fresh without one). Several nodes: torchrun's ``--nnodes``, ``--node_rank`` and
``--master_addr`` (``scripts/train.sh``).
"""

from __future__ import annotations

import sys

from ..training.loop import parse_args, train


def main(argv: list[str] | None = None) -> None:
    sys.dont_write_bytecode = True
    train(parse_args(argv))


if __name__ == "__main__":
    main()
