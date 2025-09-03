import math
from functools import partial

import torch

MAX_OP_OR_INPUT_KEY_LENGTH = 12

# all ops are implemented in torch mapping vectors of length n to vectors of length n
### binary operations


def add(x, y):
    return x + y


def mul(x, y):
    return x * y


def gate(x, y):
    return (x > 0) * y


def power(x, y):
    return x.abs() ** y


def rr_ad_gate(x, y, zero_at_bound=False):
    bound_index = torch.randint(len(x), (1,)).squeeze()

    if zero_at_bound:
        y = y - y[bound_index]

    return (x > x[bound_index]) * y


binary_ops = {
    "add": add,
    "mul": mul,
    "gate": gate,
    "power": power,
    "rr_ad_gate": rr_ad_gate,
    "rr_aad_gate": partial(rr_ad_gate, zero_at_bound=False),
}

assert (
    MAX_OP_OR_INPUT_KEY_LENGTH >= len(max(binary_ops.keys(), key=len))
), "MAX_OP_OR_INPUT_KEY_LENGTH must be greater than the length of the longest binary operation"

### unary operations


def absolute(x):
    return x.abs()


def inv(x):
    return 1 / x


def sin(x):
    return torch.sin(x * 4 * math.pi)


def exp(x):
    return torch.exp(x)


def log(x):
    return torch.log(x.abs())


def sqrt(x):
    return torch.sqrt(x.abs())


def square(x):
    return x**2


def sigmoid(x):
    return torch.sigmoid(x)


def cubic(x):
    return x**3


def relu(x):
    return torch.relu(x)


def rr_repeat(x, mirror=False):
    min_index = torch.randint(len(x), (1,)).squeeze()
    max_index = torch.randint(len(x), (1,)).squeeze()

    mini = x[min_index]
    maxi = x[max_index]
    lo = torch.minimum(mini, maxi)
    hi = torch.maximum(mini, maxi)
    width = hi - lo

    if width.item() == 0:
        return torch.full_like(x, lo)

    offset = torch.remainder(x - lo, width)

    if not mirror:
        return lo + offset

    q = torch.floor((x - lo) / width).to(torch.int64)
    flip = torch.remainder(q, 2) == 1
    return torch.where(flip, hi - offset, lo + offset)


# def relative_noise__random(x):
#     return x * torch.randn_like(x)

unary_ops = {
    "abs": absolute,
    "inv": inv,
    "sin": sin,
    "exp": exp,
    "log": log,
    "sqrt": sqrt,
    "square": square,
    "sigmoid": sigmoid,
    "cubic": cubic,
    "relu": relu,
    "rr_repeat": rr_repeat,
    "rr_repeat_m": partial(rr_repeat, mirror=True),
    # 'rr_rel_noise': relative_noise__random,
}

assert (
    MAX_OP_OR_INPUT_KEY_LENGTH >= len(max(unary_ops.keys(), key=len))
), "MAX_OP_OR_INPUT_KEY_LENGTH must be greater than the length of the longest unary operation"
