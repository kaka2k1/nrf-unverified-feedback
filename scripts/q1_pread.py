"""Run a Q1 GPU entry point using bounded safetensors reads instead of mmap."""
from functools import partial
import runpy
import sys

from transformers import modeling_utils


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in {"annotate_bfcl", "kaggle_qlora"}:
        raise SystemExit("Usage: python -m scripts.q1_pread {annotate_bfcl|kaggle_qlora} [args]")
    program = sys.argv.pop(1)
    # Transformers reads a checkpoint one tensor at a time on device_map loads.
    # pread keeps pages of already-loaded tensors out of the process RSS.
    modeling_utils.safe_open = partial(modeling_utils.safe_open, backend="pread")
    runpy.run_module(program, run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
