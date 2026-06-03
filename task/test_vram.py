#!/usr/bin/env python
"""Allocate a fixed amount of VRAM and hold it until interrupted.

Usage:
    python hold_vram.py            # holds 20 GB on cuda:0
    python hold_vram.py 16         # holds 16 GB
    CUDA_VISIBLE_DEVICES=3 python hold_vram.py   # pin to a specific card
"""

import os
import sys
import time

import torch


def main() -> None:
    gb = float(sys.argv[1]) if len(sys.argv) > 1 else 20.0

    if not torch.cuda.is_available():
        sys.exit("No CUDA device available.")

    device = "cuda:0"  # respects CUDA_VISIBLE_DEVICES
    n_floats = int(gb * 1024**3) // 4  # float32 = 4 bytes
    x = torch.empty(n_floats, dtype=torch.float32, device=device)

    used = x.element_size() * x.nelement() / 1024**3
    print(
        f"Holding {used:.2f} GB on {device} "
        f"(PID {os.getpid()}). Ctrl-C to release.",
        flush=True,
    )

    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        print("Releasing.")


if __name__ == "__main__":
    main()
