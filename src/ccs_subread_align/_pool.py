"""Shared multiprocessing Pool factory.

Uses ``forkserver`` on POSIX so workers fork from a tiny bootstrap process
rather than the (potentially multi-GB) parent. Falls back to ``spawn`` on
Windows, where ``forkserver`` is not available.
"""

import multiprocessing as mp
import sys


def get_pool(n_cores: int, maxtasksperchild: int = 200):
    method = "spawn" if sys.platform == "win32" else "forkserver"
    ctx = mp.get_context(method)
    return ctx.Pool(processes=n_cores, maxtasksperchild=maxtasksperchild)
