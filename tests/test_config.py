"""Basic tests for configurize.Config"""

from __future__ import annotations

from configurize import Config
from configurize.reference import Ref


def test_basic_config_creation():
    """Test creating a basic config"""

    class SimpleConfig(Config):
        name = "test"
        value = 42

    cfg = SimpleConfig()
    assert cfg.name == "test"
    assert cfg.value == 42


def test_config_merge():
    """Test merging configs"""

    class BaseConfig(Config):
        a = 1
        b = 2

    cfg = BaseConfig()
    cfg.merge({"a": 10})
    assert cfg.a == 10
    assert cfg.b == 2


class SubConfig(Config):
    value = 100
    self_ref = Ref(".value")
    parent_ref = Ref("..base_value")


class ParentConfig(Config):
    base_value = 42
    optional: int | None
    sub = SubConfig
    sub2: SubConfig


def test_config_references():
    """Test using Ref to reference other config values"""

    cfg = ParentConfig()
    cfg.sanity_check()
    # Test self-reference
    assert cfg.sub.self_ref == 100
    assert cfg.sub2.self_ref == 100
    # Test parent reference
    assert cfg.sub.parent_ref == 42
    assert cfg.sub2.parent_ref == 42
    # Verify references update when source changes
    cfg.base_value = 99
    assert cfg.sub.parent_ref == 99
    assert cfg.sub2.parent_ref == 99
