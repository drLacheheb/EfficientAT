import math
from typing import Optional, Callable
import torch
from torch import Tensor


def make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def cnn_out_size(in_size, padding, dilation, kernel, stride):
    s = in_size + 2 * padding - dilation * (kernel - 1) - 1
    return math.floor(s / stride + 1)


def collapse_dim(
    x: Tensor,
    dim: int,
    mode: str = "pool",
    pool_fn: Callable[[Tensor, int], Tensor] = torch.mean,
    combine_dim: Optional[int] = None,
):
    if mode == "combine":
        s = list(x.size())
        s[combine_dim] *= dim
        s[dim] //= dim
        return x.view(s)

    return pool_fn(x, dim)
