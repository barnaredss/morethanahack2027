"""End to end: provided data -> submission CSV, validated.

    python -m possession.run                                   # default pipeline
    python -m possession.run --pipeline baseline_v0 --out submissions/x.csv
    python -m possession.run --list

The written file is byte-deterministic: "\\n" line endings, no timestamps, no randomness.
"""

import argparse
import hashlib
import os
import sys

import numpy as np

from . import pipelines
from .data import LABEL_CHARS, ROOT, load_match

sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402


def write_submission(path, codes):
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    labels = LABEL_CHARS[np.asarray(codes, dtype=np.int64)]
    body = "frame,label\n" + "".join("{},{}\n".format(i, c) for i, c in enumerate(labels))
    with open(path, "wb") as handle:
        handle.write(body.encode("ascii"))
    return path


def sha256(path):
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def validate(path, n_expected):
    """The exact checks the official scorer runs. Raises on failure."""
    frame = pm.read_label_csv(path)
    reason = pm.validate_submission(frame, n_expected)
    if reason:
        raise ValueError("{} fails validation: {}".format(path, reason))
    return frame


def main(argv=None):
    parser = argparse.ArgumentParser(description="Generate a possession submission.")
    parser.add_argument("--pipeline", default=None, help="registered pipeline name")
    parser.add_argument("--out", default=os.path.join(ROOT, "submissions", "submission.csv"))
    parser.add_argument("--list", action="store_true", help="list pipelines and exit")
    args = parser.parse_args(argv)

    if args.list:
        print("\n".join(pipelines.names()))
        return 0

    fn = pipelines.get(args.pipeline)
    name = args.pipeline or pipelines.DEFAULT
    match = load_match()
    codes = np.asarray(fn(match), dtype=np.int8)
    if codes.shape != (match.n,):
        raise ValueError("pipeline returned shape {}, expected ({},)".format(codes.shape, match.n))
    path = write_submission(args.out, codes)
    validate(path, match.n)

    counts = {c: int((codes == i).sum()) for i, c in enumerate(LABEL_CHARS)}
    poss = sum(1 for _ in _spells(codes, match))
    print("pipeline {}  ->  {}".format(name, os.path.relpath(path, ROOT)))
    print("  H {H}  A {A}  D {D}  X {X}".format(**counts))
    print("  possessions in graded halves: {}".format(poss))
    print("  sha256 {}".format(sha256(path)))
    return 0


def _spells(codes, match):
    for _, first, last in match.half_blocks():
        seg = codes[first:last + 1]
        prev = -1
        for c in seg:
            if c != prev and c in (0, 1):
                yield c
            prev = c


if __name__ == "__main__":
    sys.exit(main())
