from dataclasses import is_dataclass

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
    slice,
    range,
)

try:
    import numpy as np

    NUMPY_TYPES = (
        np.ndarray,
        np.dtype,
    )
except:
    NUMPY_TYPES = ()
try:
    import torch

    TORCH_TYPES = (
        torch.Tensor,
        torch.dtype,
    )
except:
    TORCH_TYPES = ()

ALLOWED_TYPES = SEQUENCE_TYPES + MAPPING_TYPES + BASE_TYPES + NUMPY_TYPES + TORCH_TYPES


def recur_to_allowed_types(obj, extra_allowed=()):
    """Recursive convert the object into allowed types, objects that is not allowed
    will be replaced by repr(obj).
    NOTE: subclasses of allowed types may NOT be allowed

    Args:
        obj (Any): the object to be convert

    Returns:
        Any: the converted object
    """

    if not isinstance(obj, ALLOWED_TYPES + extra_allowed) and not is_dataclass(obj):
        logger.error(f"{obj} is NOT allowed in Config!!")
        obj = repr(obj)
    else:
        cls = type(obj)
        if is_dataclass(obj):
            cls = dict
            obj = obj.__dict__

        if cls in SEQUENCE_TYPES:
            obj = cls([recur_to_allowed_types(i, extra_allowed) for i in obj])
        elif cls in MAPPING_TYPES:
            obj = cls(
                {k: recur_to_allowed_types(v, extra_allowed) for k, v in obj.items()}
            )

    return obj
