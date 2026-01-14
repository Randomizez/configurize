from __future__ import annotations

from functools import cached_property


class DataClass:
    """
    like dataclass, support inheritance & dict like

    class MyConfig(DataClass):
        a = 1
        b = 2

    func(**MyConfig())

    class YourConfig(MyConfig):
        a = 2
    """

    @classmethod
    def _get_class_attributes(cls):
        attributes = {}
        attributes.update(cls.__dict__)
        for base_cls in cls.__bases__:
            if issubclass(base_cls, DataClass):
                for k, v in base_cls._get_class_attributes().items():
                    attributes.setdefault(k, v)
        return attributes

    @classmethod
    def _get_class_annotations(cls):
        attributes = {}
        attributes.update(cls.__annotations__)
        for base_cls in cls.__bases__:
            if issubclass(base_cls, DataClass):
                for k, v in base_cls._get_class_annotations().items():
                    attributes.setdefault(k, v)
        return attributes

    @cached_property
    def _defined_attributes(self) -> set[str]:
        return set(
            self._get_class_attributes().keys() | self._get_class_annotations().keys()
        )

    def _merge_args(self, kwargs: dict):
        from copy import deepcopy
        from typing import Callable

        for k, v in self.__class__._get_class_attributes().items():
            if isinstance(v, DataClass):
                setattr(self, k, v.__class__(**{k: v for k, v in v.items(deref=False)}))
            elif not k.startswith("_") and not isinstance(
                v, (Callable, cached_property, property, classmethod)
            ):
                setattr(self, k, deepcopy(v))  # Config build: copy class attr to object
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __init__(self, **kwargs) -> None:
        self._merge_args(kwargs)

    def __repr__(self):
        def repr(x):
            return str(x)

        return (
            self.__class__.__qualname__
            + "("
            + ", ".join(f"{k}={repr(v)}" for k, v in self.__dict__.items())
            + ")"
        )

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def get(self, name, *args, **kwargs):
        return self.__dict__.get(name, *args, **kwargs)

    def keys(self):
        return self.__dict__.keys()

    def __eq__(self, other):
        if isinstance(other, DataClass):
            return self.__dict__ == other.__dict__
        return super().__eq__(other)

    def __hash__(self):
        return hash(tuple(self.__dict__))
