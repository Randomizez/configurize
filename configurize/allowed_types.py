import numpy as np
import torch
from loguru import logger

SEQUENCE_TYPES = (
    list,
    tuple,
    set,
)

MAPPING_TYPES = (dict,)

BASE_TYPES = (
    str,
    int,
    float,
    bool,
    type(None),
    torch.Tensor,
    torch.dtype,
    np.ndarray,
    np.dtype,
    slice,
    range,
)

ALLOWED_TYPES = SEQUENCE_TYPES + MAPPING_TYPES + BASE_TYPES


def recur_to_allowed_types(obj, extra_allowed=()):
    """Recursive convert the object into allowed types, objects that is not allowed
    will be replaced by repr(obj).
    NOTE: subclasses of allowed types may NOT be allowed

    Args:
        obj (Any): the object to be convert

    Returns:
        Any: the converted object
    """

    if not isinstance(obj, ALLOWED_TYPES + extra_allowed):
        logger.error(f"{obj} is NOT allowed in Config!!")
        obj = repr(obj)
    else:
        cls = type(obj)
        if cls in SEQUENCE_TYPES:
            obj = cls([recur_to_allowed_types(i, extra_allowed) for i in obj])
        elif cls in MAPPING_TYPES:
            obj = cls(
                {k: recur_to_allowed_types(v, extra_allowed) for k, v in obj.items()}
            )

    return obj
