from __future__ import annotations

import random
import sys
from pathlib import Path

import torch

_TRAIN_DIR   = Path(__file__).resolve().parent
_PROJECT_DIR = _TRAIN_DIR.parent

if str(_TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(_TRAIN_DIR))

from dataset import SingleBinNumPyDataset


class ShardLoader:
    """
    Sequentially iterates through a shuffled index list of a rotating queue of binary shard files,
    ensuring every token in each shard is seen exactly once per epoch without overlaps or gaps.

    Args:
        shard_dir:      Directory containing train_shard_*.bin files.
        context_length: Number of tokens per sequence.
        batch_size:     Number of sequences per batch.
        device:         Target torch device for returned tensors.
        seed:           RNG seed for reproducible shard order and sampling.
    """

    def __init__(
        self,
        shard_dir:      str | Path,
        context_length: int,
        batch_size:     int,
        device:         torch.device,
        seed:           int = 42,
    ):
        self.shard_dir      = Path(shard_dir)
        self.context_length = context_length
        self.batch_size     = batch_size
        self.device         = device

        self.shards = sorted(self.shard_dir.glob("train_shard_*.bin"))
        if not self.shards:
            raise FileNotFoundError(
                f"No train_shard_*.bin files found in: {shard_dir}"
            )

        self._rng:     random.Random   = random.Random(seed)
        self._queue:   list[Path]      = []
        self._ds:      SingleBinNumPyDataset | None = None
        self._indices: list[int]       = []
        self._pos:     int             = 0
        self._epoch:   int             = 0

        self._load_next_shard()

    def _load_next_shard(self):
        if not self._queue:
            self._queue = list(self.shards)
            self._rng.shuffle(self._queue)
            self._epoch += 1

        path     = self._queue.pop()
        self._ds = SingleBinNumPyDataset(
            str(path), self.context_length, self.batch_size
        )
        
        # Create a randomized permutation of all valid starting positions for this shard
        self._indices = list(range(len(self._ds)))
        self._rng.shuffle(self._indices)
        self._pos = 0
        
        print(
            f"    [ShardLoader] epoch={self._epoch}  shard={path.name}"
            f"  ({len(self._ds):,} positions)",
            flush=True,
        )

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (x, y) tensors of shape (batch_size, context_length)."""
        # If the remaining tokens in this shard can't fill a complete batch, jump to the next shard
        if self._pos + self.batch_size > len(self._indices):
            self._load_next_shard()

        batch_indices = self._indices[self._pos : self._pos + self.batch_size]
        xs, ys  = [], []
        for i in batch_indices:
            x, y = self._ds[i]
            xs.append(torch.from_numpy(x))
            ys.append(torch.from_numpy(y))

        self._pos += self.batch_size
        return (
            torch.stack(xs).to(self.device),
            torch.stack(ys).to(self.device),
        )

    def fast_forward(self, num_batches: int):
        """Skip num_batches without constructing PyTorch tensors (for quick resume)."""
        print(f"    [ShardLoader] Fast-forwarding {num_batches:,} batches...")
        for _ in range(num_batches):
            if self._pos + self.batch_size > len(self._indices):
                self._load_next_shard()
            self._pos += self.batch_size
        print(f"    [ShardLoader] Resumed at epoch={self._epoch}, shard={self.shards[0].name if not self._queue else self._queue[-1].name}, pos={self._pos:,}")



class SingleBinLoader:
    """
    Loads batches from a single .bin file (e.g. val.bin or test.bin).

    Shuffles indices once on construction for randomised evaluation order.
    When all positions are exhausted, reshuffles and starts a new pass.

    Args:
        bin_path:       Path to the .bin file.
        context_length: Number of tokens per sequence.
        batch_size:     Number of sequences per batch.
        device:         Target torch device for returned tensors.
        seed:           RNG seed for reproducible sampling.
    """

    def __init__(
        self,
        bin_path:       str | Path,
        context_length: int,
        batch_size:     int,
        device:         torch.device,
        seed:           int = 43,
    ):
        self.bin_path       = Path(bin_path)
        self.context_length = context_length
        self.batch_size     = batch_size
        self.device         = device

        if not self.bin_path.exists():
            raise FileNotFoundError(f"Validation bin not found: {self.bin_path}")

        self._ds  = SingleBinNumPyDataset(str(self.bin_path), context_length, batch_size)
        self._rng = random.Random(seed)
        self._indices: list[int] = []
        self._pos: int = 0
        self._reshuffle()

        print(
            f"    [SingleBinLoader] {self.bin_path.name}"
            f"  ({len(self._ds):,} positions)",
            flush=True,
        )

    def _reshuffle(self):
        self._indices = list(range(len(self._ds)))
        self._rng.shuffle(self._indices)
        self._pos = 0

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (x, y) tensors of shape (batch_size, context_length)."""
        if self._pos + self.batch_size > len(self._indices):
            self._reshuffle()

        batch_indices = self._indices[self._pos : self._pos + self.batch_size]
        xs, ys = [], []
        for i in batch_indices:
            x, y = self._ds[i]
            xs.append(torch.from_numpy(x))
            ys.append(torch.from_numpy(y))

        self._pos += self.batch_size
        return (
            torch.stack(xs).to(self.device),
            torch.stack(ys).to(self.device),
        )