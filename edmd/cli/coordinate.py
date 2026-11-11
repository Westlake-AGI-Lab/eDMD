"""``python -m edmd.cli.coordinate`` -- training with periodic evaluation, for one run on one node.

    python -m edmd.cli.coordinate --nproc 8 --run-file configs/runs/<model>/<method>.yaml [--set key=value ...] \\
        [--eval-devices cuda:0 ...] [--score-python PATH] [--hpsv3pp-python PATH]

Each pass reads the run directory: its complete checkpoints and the results of the evaluation points 0,
``run.eval_every``, 2 ``run.eval_every``, ..., ``max_steps``. Every point that has a checkpoint but no complete result
is evaluated with ``edmd.cli.evaluate generate`` and ``score`` into ``<run>/eval/<eval name>/checkpoint-<N>`` (then
``checkpoint-<N>-ema`` with ``run.eval_ema``) and published to W&B when it is online (``edmd.cli.evaluate publish``);
then training continues to the next point (torchrun ``edmd.cli.train --resume auto --stop-after N``; checkpoint-0
starts the curve). Running the same command again continues the run. With ``run.eval_every: 0`` it only trains.

Scoring runs in ``--score-python`` (default: this interpreter), HPSv3++ in ``--hpsv3pp-python`` (default: this
interpreter); both are probed before the first training segment, so a missing scorer fails at once. Each step logs to
``<run>/logs/<step>_*.log``; a failure prints the end of that log.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Sequence

#: a previous segment still holding the rendezvous port: HCCL on the NPU, c10d / NCCL on a GPU
BIND_FAILURE_SIGNATURES = ("Bind_Failed", "EJ0003", "EADDRINUSE", "Address already in use")
BIND_RETRIES, RETRY_DELAY = 3, 30.0      # a just-closed port can stay in TIME_WAIT for 60 s


class JobFailed(RuntimeError):
    def __init__(self, label: str, returncode: int, log: Path):
        self.log = log
        super().__init__(f"{label} exited with status {returncode}; log: {log}")


def run_logged(command: Sequence[str], logs: Path, label: str) -> Path:
    """Run one child (this process's environment, the repository first on ``PYTHONPATH``) with its output in
    ``logs/<label>_*.log``, the command line first; raises ``JobFailed``."""
    from ..config.load import REPO

    logs.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(item for item in (str(REPO), os.environ.get("PYTHONPATH"))
                                                      if item), PYTHONUNBUFFERED="1")
    with tempfile.NamedTemporaryFile(mode="w", prefix=f"{label}_", suffix=".log", dir=logs, delete=False) as log:
        print(f"{time.strftime('%H:%M:%S')} {label}: log {log.name}", flush=True)
        log.write(shlex.join(map(str, command)) + "\n")
        log.flush()
        code = subprocess.run(list(map(str, command)), env=env, stdout=log, stderr=subprocess.STDOUT,
                              stdin=subprocess.DEVNULL).returncode
    if code:
        raise JobFailed(label, code, Path(log.name))
    return Path(log.name)


def next_stop(latest: int | None, every: int, total: int) -> int:
    """The step the next training segment stops at: 0 without a checkpoint (the curve's start), else the next
    evaluation point (``max_steps`` without evaluation)."""
    if not every:
        return total
    if latest is None:
        return 0
    return min(total, (latest // every + 1) * every)


class Driver:
    def __init__(self, args: argparse.Namespace):
        from ..config.core import ConfigError
        from ..config.load import compose, parse_assignments, run_output_dir

        self.args = args
        self.config, _ = compose(args.run_file, parse_assignments(args.set))
        self.output = run_output_dir(self.config)
        self.logs = self.output / "logs"
        self.every, self.total = int(self.config.run.eval_every), int(self.config.method.updates.max_steps)
        if self.every and self.config.eval is None:
            raise ConfigError("run.eval_every needs the run's eval config (`eval:` in the run file)")
        self.roles = ("student", "student_ema") if self.config.run.eval_ema else ("student",)
        spec = self.config.eval
        self.metrics = tuple(spec.metrics) + tuple(spec.extra_metrics) if self.every else ()
        self.results = self.output / "eval" / spec.name if self.every else None
        self.devices = ["--devices", *args.eval_devices] if args.eval_devices else []

    def scorers(self) -> list[tuple[str, list[str]]]:
        """``(interpreter, metrics)``: every metric but HPSv3++ in ``--score-python``, HPSv3++ in its own."""
        others = [metric for metric in self.metrics if metric != "hpsv3pp"]
        groups = [(self.args.score_python, others)] if others else []
        return groups + ([(self.args.hpsv3pp_python, ["hpsv3pp"])] if "hpsv3pp" in self.metrics else [])

    def train(self, stop: int) -> None:
        from ..runtime.checkpoint import complete_checkpoints

        command = [sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={self.args.nproc}",
                   "-m", "edmd.cli.train", "--run-file", self.args.run_file,
                   *[f"--set={item}" for item in self.args.set], "--resume", "auto", "--stop-after", str(stop)]
        for attempt in range(BIND_RETRIES + 1):
            try:
                run_logged(command, self.logs, f"train_to_{stop:06d}")
                break
            except JobFailed as error:
                tail = error.log.read_text(errors="replace")[-20000:]
                if attempt == BIND_RETRIES or not any(sign in tail for sign in BIND_FAILURE_SIGNATURES):
                    raise
                print(f"the rendezvous port is still bound; retrying in {RETRY_DELAY:.0f}s", flush=True)
                time.sleep(RETRY_DELAY)
        if stop not in complete_checkpoints(self.output):
            raise RuntimeError(f"the training segment exited without a complete checkpoint-{stop}")

    def evaluate(self, step: int, checkpoint: Path) -> bool:
        """Generate and score the point's missing lanes; whether anything was evaluated."""
        from ..config.core import dump_yaml, to_mapping
        from ..evaluation.scoring import missing_metrics

        evaluated = False
        for role in self.roles:
            directory = self.results / (f"checkpoint-{step}" + ("-ema" if role == "student_ema" else ""))
            if not missing_metrics(directory, self.metrics):
                continue
            spec = self.results / "eval.yaml"                 # the run's eval config, as composed
            spec.parent.mkdir(parents=True, exist_ok=True)
            spec.write_text(dump_yaml(to_mapping(self.config.eval)), encoding="utf-8")
            run_logged([sys.executable, "-m", "edmd.cli.evaluate", "generate", "--checkpoint", checkpoint,
                        "--eval", spec.resolve(), "--out", self.results, "--role", role, *self.devices],
                       self.logs, f"generate_{directory.name}")
            for python, metrics in self.scorers():
                run_logged([python, "-m", "edmd.cli.evaluate", "score", "--dir", directory, "--metrics", *metrics,
                            *self.devices], self.logs, f"score_{directory.name}_{'_'.join(metrics)}")
            missing = missing_metrics(directory, self.metrics)
            if missing:
                raise RuntimeError(f"{directory} still lacks the scores of {missing}")
            evaluated = True
        return evaluated

    def publish(self) -> None:
        """Sync every complete result to the run's W&B run (when W&B is online); a failure does not stop training."""
        from ..tracking.wandb import effective_mode

        wb = self.config.run.wandb
        if not (wb.enabled and effective_mode(wb) == "online" and (self.output / "config.yaml").is_file()):
            return
        try:
            run_logged([sys.executable, "-m", "edmd.cli.evaluate", "publish", "--run", self.output], self.logs,
                       "publish")
        except JobFailed as error:
            print(f"W&B publishing failed ({error}); training continues: publish later with "
                  f"`python -m edmd.cli.evaluate publish --run {self.output}`", flush=True)

    def run(self) -> None:
        from ..runtime.checkpoint import complete_checkpoints
        from ..tracking.media import eval_schedule

        print(f"run {self.output}: {self.total} updates on {self.args.nproc} processes"
              + (f", evaluated every {self.every} with {self.config.eval.name}" if self.every else ""), flush=True)
        for python, metrics in self.scorers():     # a missing scorer fails now, not after the first segment
            run_logged([python, "-m", "edmd.cli.evaluate", "score", "--check", "--metrics", *metrics], self.logs,
                       "check_scorers")
        published = False
        while True:
            checkpoints = complete_checkpoints(self.output)
            if self.every:
                evaluated = False
                for step in eval_schedule(self.every, self.total):
                    if step in checkpoints:
                        evaluated |= self.evaluate(step, checkpoints[step])
                if evaluated or (checkpoints and not published):
                    self.publish()
                    published = True
            latest = max(checkpoints) if checkpoints else None
            if latest is not None and latest >= self.total:
                break
            self.train(next_stop(latest, self.every, self.total))
        print(f"finished: {self.output}", flush=True)


def main(argv: Sequence[str] | None = None) -> int:
    from ..platform import current

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-file", required=True, help="configs/runs/<model>/<method>.yaml")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="override a config value, e.g. method.updates.max_steps=3000 (repeatable)")
    parser.add_argument("--nproc", type=int, default=None, help="training processes (default: every visible card)")
    parser.add_argument("--eval-devices", nargs="+",
                        help="generation and scoring devices (default: every visible card)")
    parser.add_argument("--score-python", default=sys.executable,
                        help="the interpreter of the scorers' environment (default: this one)")
    parser.add_argument("--hpsv3pp-python", default=sys.executable,
                        help="the interpreter of the HPSv3++ environment (default: this one)")
    args = parser.parse_args(argv)
    sys.dont_write_bytecode = True
    args.nproc = args.nproc or current().cards()
    try:
        Driver(args).run()
    except JobFailed as error:
        lines = error.log.read_text(errors="replace").splitlines()
        tail = ([line for line in lines if line.startswith("[rank")] or lines)[-25:]   # torchrun: the workers' errors
        print(f"Failed: {error}", f"--- the end of {error.log}", *tail, sep="\n", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
