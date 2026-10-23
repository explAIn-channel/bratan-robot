#!/usr/bin/env python3
"""Start the trainer with bounded native thread pools on Windows spawn."""

import os


_NATIVE_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)

for _name in _NATIVE_THREAD_ENV_VARS:
    os.environ[_name] = "1"
del _name


def main():
    """Import the heavy trainer only after native thread limits are fixed."""
    from train_rl import main as train_main

    return train_main()


if __name__ == "__main__":
    raise SystemExit(main())
