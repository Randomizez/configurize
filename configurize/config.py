from __future__ import annotations

import ast
import inspect
import weakref
from contextlib import contextmanager
from functools import partial
from typing import Any, Callable, Optional, TypedDict

from loguru import logger

from .allowed_types import recur_to_allowed_types
from .data_class import DataClass
from .reference import CfgReferenceError, Ref
from .utils import filter_traceback_frames, get_func_brief, writable_property


class TaskSpec(TypedDict, total=False):
    """Indicate resource & entrypoint for specific task.
    Below attributes DO NOT need default value if not set.
    """

    replica: int
    """Number of replica(nodes) this task should run on."""
    command: str
    """Entrypoint for this task. e.g. 'redis-server --port 12345'"""
    envs: dict[str, Any]
    """Environment variables for this task."""
    image: str
    """Docker image for this task. Optional."""


class Config(DataClass):
    """Base Config Type, like dataclass, extra support:
    - config.sub_config = SubConfig()
    - sub_config.root().xx
    - sub_config.a = Ref('..a') # use father.a

    - config.make_picklable() -> dict[BaseTypes]
    - config.merge(other_config_or dict)
    - config.sanity_check() # any assert
    """

    _allow_search = False
    """if can't get attribute, try to find in the tree"""

    _set_attribute_traces = None
    _tracing_set_attribute = False
    """if set to non-None, trace all set attr."""

    _flatten_args = None
    """the flatten tree (only on root)"""

    _sub_cfg_name = None
    """my name in father"""

    _is_tree_node = True
    """add this node to tree search"""

    _modify_stack = []
    """store modification in stack, enable push & pop"""

    _allow_set_new_attr = False
    """if set True, `cfg.aaa = 1` is allowed even if 'aaa' does not exists. None for allowed now."""

    _buildin_functions = [
        "father",
        "root",
        "sanity_check",
        "merge",
        "keys",
        "items",
        "get",
        "to_dict",
        "modify",
        "diff",
        "flatten_config",
        "assert_critical_attrs_expected",
    ]

    critical_keys: Optional[list[str]]
    """if set, mark some attributes as 'critical', critical attrs can be compared with other config for 
    consistency check(`self.assert_critical_attrs_expected(expected_cfg)`).
    e.g. ["key1", "key2?"] means:
    - key1 must in expected_cfg and must same as expected_cfg[key1].
    - key2 must same as in expected_cfg[key2] if it exists in expected_cfg.
    """

    task_specs: Optional[dict[str, TaskSpec]]
    """If set, this module requires somes individual entrypoints.
    """

    @writable_property
    def _class_name(self):
        return f"{self.__class__.__module__}.{self.__class__.__name__}"

    def __getstate__(self):
        return self.to_dict()

    def __setstate__(self, state):
        self._merge_args(state)

    def father(self) -> "Config":
        if hasattr(self, "_father"):
            return self._father()
        return None

    def root(self) -> "Config":
        """backtrace this config node to get root node

        Returns:
            Config: The Root Config.
        """
        root = self
        while root.father() is not None:
            root = root.father()
        return root

    def assert_critical_attrs_expected(
        self, expected_cfg: "dict | Config", cum_errs: list = None
    ):
        """Recursively compare self with another 'expected_cfg', raise error
        if any of 'critical_keys' mismatches.

        Args:
            expected_cfg (dict | Config): The reference config, NOTE: only critical_keys of
            THIS(self) config will be checked.

        Raises:
            Exception: raise all mismatch keys.
        """
        errs = []  # hold error and raise together
        # recursive check sub nodes
        for k, v in self.items():
            if isinstance(v, Config):
                try:
                    expected_sub_cfg = expected_cfg.get(k, None) or Config()
                    v.assert_critical_attrs_expected(expected_sub_cfg, errs)
                except Exception as e:
                    errs.append(e)

        if hasattr(self, "critical_keys"):
            for k in self.critical_keys:
                must_have = "?" not in k
                k = k.strip("?")

                try:
                    current = self[k]
                    if must_have:
                        assert (
                            k in expected_cfg
                        ), f"Cannot get expected value of {self._get_node_name()}.{k}: {current=}"
                    else:
                        logger.warning(
                            f"Cannot get expected value of {self._get_node_name()}.{k}: {current=}"
                        )
                    if k in expected_cfg:
                        expected = expected_cfg[k]
                        assert (
                            current == expected
                        ), f"Value different on {self._get_node_name()}.{k}: {current=}, {expected=}"
                except Exception as e:
                    errs.append(e)
        if cum_errs is not None:
            cum_errs.extend(errs)
        else:
            if errs:
                import traceback

                text = []
                for e in errs:
                    tb = "\n".join(traceback.format_exception(e)[-10:])
                    text.append(f">>> {tb}\n")
                text = "\n".join(text)
                raise Exception(f"{text}\nAttributes Check Fail!")

    def _deref(self, name, value, deref=True):
        if isinstance(value, Ref):
            try:
                cur = self
                for action in value.actions:
                    if action == ".":  # goto father
                        if cur.father() is None:
                            raise CfgReferenceError(
                                f"Unable to find father of {cur._class_name} requested by <{self._class_name}.{name} = Ref('{value.ref_str}')>"
                            )
                        cur = cur.father()
                    else:  # goto sub
                        if type(cur) is type:
                            cur = super(Config, cur).__getattribute__(cur, action)
                        else:
                            cur = super(Config, cur).__getattribute__(action)
                        if isinstance(cur, Ref):
                            ref_name = f"{self._get_node_name()}.{name}"
                            raise ValueError(
                                f"Cannot Refer to Ref: '{value.ref_str}' @ {ref_name} "
                                f"-> {cur}"
                            )
                if deref or (type(cur) is not type and isinstance(cur, Callable)):
                    # must deref Reference to Callable
                    value = cur
                else:
                    value.cur_value = cur
            except (AttributeError, CfgReferenceError) as e:
                if value.default is CfgReferenceError:
                    if deref:
                        raise value.default(
                            f"Unable to find reference of <{self._class_name}.{name} = Ref('{value.ref_str}')>"
                            + (
                                "\n\nReference Error took place when you try to access a Ref() but we can't find "
                                "the target in config tree. There might be 2 potential reason: \n"
                                "1. Typo in your ref string or wrong path.\n"
                                "2. The Ref() is accessed when the tree is not fully built. NOTE that "
                                "you should not access a Ref() during building (e.g. in __init__())!"
                            ),
                            e.with_traceback(
                                filter_traceback_frames("configurize", e.__traceback__)
                            ),
                        ) from None
                    else:
                        return value
                else:
                    return value.default

        return value

    def _get_node_name(self) -> str:
        """Get node name like 'Config.model.stem'

        Returns:
            str: node name
        """
        comp = []
        ptr = self
        while ptr:
            comp.insert(0, ptr._sub_cfg_name or ptr.__class__.__name__)
            ptr = ptr.father()
        return ".".join(comp)

    def _fix_ref(self):
        """After copy config, call this func to fix reference."""
        for k, v in self.items(deref=False):
            if isinstance(v, Config):
                v._fix_ref()
                v._father = weakref.ref(self)

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_"):
            return super().__setattr__(name, value)

        if self._allow_set_new_attr is False:
            if name not in self._defined_attributes:
                raise KeyError(
                    f"Undefined key '{name}' in {self._class_name}, raise to avoid typo!"
                )

        if isinstance(value, Config):
            value._father = weakref.ref(self)
            value._sub_cfg_name = name

        if self._tracing_set_attribute:
            import traceback
            from collections import defaultdict

            self._set_attribute_traces = self._set_attribute_traces or defaultdict(list)
            caller_stack = traceback.extract_stack(inspect.currentframe())
            caller_info = traceback.format_list(caller_stack[-2:-1])
            caller_info = "\n".join(caller_info)
            defined = name in self._defined_attributes or self._allow_set_new_attr
            self._set_attribute_traces[name].append((value, caller_info, defined))
        return super().__setattr__(name, value)

    def _all_set_history(self) -> dict[str, list[tuple[Any, str, bool]]]:
        history = {}
        if self._set_attribute_traces:
            for k, trace in self._set_attribute_traces.items():
                history[f"{self._get_node_name()}.{k}"] = trace
        for k, v in self.items(deref=False):
            if isinstance(v, Config):
                history.update(v._all_set_history())
        return history

    def __init__(self, **kwargs):
        """
        1. set kwargs on self. (deprecated, set in __init__.)
        2. build all my sub-configs.
        """
        if kwargs:
            logger.warning(
                "Setting config use Config(a=a) is not recommanded! These values might "
                "be override by __init__() and do not actually take effect."
            )
        self._merge_args(kwargs)

        for k, v in self.items(deref=False):
            if type(v) is type and issubclass(v, Config):
                v = v()
                setattr(self, k, v)
            if isinstance(v, Config):
                v._father = weakref.ref(self)
                v._sub_cfg_name = k

        if self._allow_search:
            self._flatten_args = self._flatten_config()

    def __copy__(self):
        new_self = self.__class__(**self)
        new_self._fix_ref()
        return new_self

    def sanity_check(self):
        # recursive check sub-configs
        for k, v in self.items(deref=False):
            if isinstance(v, Config):
                v.sanity_check()

        # check defined attrs all set
        missing_attrs = []
        expected_attrs = self._get_class_annotations()
        for k, t in expected_attrs.items():
            if "Optional[" in str(t):
                continue  # pass check for Optional annotation
            if not hasattr(self, k):
                missing_attrs.append(k)
        if missing_attrs:
            repr_str = "\n".join(f"- {k}: {expected_attrs[k]}" for k in missing_attrs)
            raise AttributeError(
                f"Attributes defined but not set for {self._get_node_name()}:\n{repr_str}"
            )

    def _flatten_config(self) -> dict:
        flatten_dict = {}
        this_class = self._get_node_name()
        for k, v in self.items(deref=False):
            if not self._is_tree_node:
                # not a tree node, use unique name
                k = this_class + "." + k
            if isinstance(v, Config):
                for sk, sv in v._flatten_config().items():
                    if sk in flatten_dict:
                        flatten_dict[sk][0].extend(sv[0])
                        flatten_dict[sk][1].extend(sv[1])
                    else:
                        flatten_dict[sk] = sv
            else:
                if k in flatten_dict:
                    flatten_dict[k][0].append(v)
                    flatten_dict[k][1].append(this_class)
                else:
                    flatten_dict[k] = ([v], [this_class])
        return flatten_dict

    def _find_in_tree(self, name):
        this_class = self._get_node_name()
        if self._flatten_args and name in self._flatten_args:
            values, classes = self._flatten_args[name]
            alters = [
                f"{'    ' if i else '--> '}{v} @ {c}"
                for i, (v, c) in enumerate(zip(values, classes))
            ]
            logger.opt(depth=2).warning(
                f"Cannot find '{name}' in {this_class}, using:\n" + "\n".join(alters)
            )
            return values[0]
        raise AttributeError(f"Can't find {name} anywhere!")

    def _get(self, name, deref=True):
        try:
            value = super().__getattribute__(name)
            value = self._deref(name, value, deref=deref)
            return value
        except AttributeError as e:
            if self._allow_search:
                return self._find_in_tree(name)
            else:
                raise e

    def __getattribute__(self, name: str):
        try:
            if not name.startswith("_"):
                return self._get(name)
        except AttributeError as e:
            e.__traceback__ = filter_traceback_frames("configurize", e.__traceback__)
            raise e from None
        return super().__getattribute__(name)

    def __repr__(self):
        from pprint import pformat

        text = [f"{self._class_name}("]

        for k, v in self.to_dict(rep=True).items():
            sub_texts = pformat(v, compact=True).splitlines()
            sub_texts[0] = f"{k} = {sub_texts[0]}"
            sub_texts[-1] = f"{sub_texts[-1]},"
            for f in sub_texts:
                text.append(f"    " + f)
        text.append(")")
        if self.root() is self and self.__class__.__name__ == "Exp":
            text.append(f"🗺️ TL;DR 🗺️\n{self._brief()}")
        return "\n".join(text)

    def _brief(self) -> str:
        self_repr = f"{self.__class__.__module__}.{self.__class__.__name__}"
        text = [f"{self_repr}("]
        for k, v in self.items(deref=False, rep=True):
            if isinstance(v, Config):
                sub_texts = v._brief().splitlines()
                sub_texts[0] = f"{k} = {sub_texts[0]}"
                sub_texts[-1] = f"{sub_texts[-1]},"
                for f in sub_texts:
                    text.append(f"    " + f)
        if len(text) == 1:
            text[0] += ")"
        else:
            text.append(")")
        return "\n".join(text)

    def _update_value(self, k, v):
        src_value = getattr(self, k, None)
        src_type = type(src_value)
        applied_diff = []
        if isinstance(src_value, Config) and isinstance(v, Config):
            applied_diff = src_value.merge(v)
            v = src_value
        elif src_value is not None and src_type != type(v):
            try:
                v = ast.literal_eval(v)
            except:
                pass
        if v != src_value:
            applied_diff.append((self._get_node_name() + "." + k, src_value, v))
        setattr(self, k, v)
        return applied_diff

    def __getitem__(self, key):
        return self._get(key)

    def __iter__(self):
        return iter(self.keys())

    def merge(self, args: dict, exists_only=False):
        applied_diff = []
        for k, v in args.items():
            if "." in k:
                comp = k.split(".")
                sub = getattr(self, comp[0], None)
                if isinstance(sub, Config):
                    applied_diff.extend(
                        sub.merge({".".join(comp[1:]): v}, exists_only=exists_only)
                    )
                    continue
                else:
                    raise KeyError(f"Arg: {k} does not exists or not a Config!")
            if hasattr(self, k):
                applied_diff.extend(self._update_value(k, v))
            else:
                if exists_only:
                    raise KeyError(f"Arg: {k} does not exists!")
                else:
                    logger.info(f"New Arg: {k} = {v}")
                    applied_diff.extend(self._update_value(k, v))
        return applied_diff

    def diff(self, other: "Config|dict", prefix=""):
        diff_self = {}
        sk, ok = list(self.keys()), list(other.keys())

        def _get(obj, k):
            if isinstance(obj, Config):
                return obj._get(k)
            return obj.get(k)

        for k in sorted(list(set(sk + ok))):
            v = _get(self, k) if k in sk else "<N/A>"
            ov = _get(other, k) if k in ok else "<N/A>"
            if callable(v):
                v = get_func_brief(v)
            if callable(ov):
                ov = get_func_brief(ov)
            if v != "<N/A>" and ov != "<N/A>":
                if isinstance(v, Config) and isinstance(ov, (Config, dict)):
                    diff_self[f"{prefix}{k}"] = v.diff(ov)
                elif str(v) != str(ov):
                    diff_self[f"{prefix}{k}"] = (v, ov)
            else:
                if isinstance(other, dict) and ov == "<N/A>":
                    continue
                diff_self[f"{prefix}{k}"] = (v, ov)
        return build_configdiff_from_flatten(diff_self)

    def flatten_config(self, prefix=None) -> dict[str, Any]:
        flatten = {}
        prefix = prefix or self.__class__.__name__
        for k, v in self.items():
            if isinstance(v, Config):
                flatten.update(v.flatten_config(f"{prefix}.{k}"))
            else:
                flatten[f"{prefix}.{k}"] = v
        return flatten

    def keys(self):
        my_keys = []
        for name in dir(self):
            if name.startswith("_"):
                continue

            if name not in self._buildin_functions:
                my_keys.append(name)
        return my_keys

    def items(self, deref=True, funcs=False, rep=False):
        props = {}
        for k in self.keys():
            if not isinstance(getattr(self.__class__, k, None), property):
                value = self._get(k, deref=deref)
                if (
                    not funcs
                    and type(value) is not type
                    and isinstance(value, Callable)
                ):
                    continue
                props[k] = value
            elif deref or rep:
                props[k] = getattr(self, k)
        return props.items()

    def to_dict(self, rep=False):
        out = {}
        for k, v in self.items(deref=not rep, funcs=rep, rep=rep):
            if isinstance(v, Config) and not rep:
                out[k] = v.to_dict(rep=rep)
            elif isinstance(v, Callable):
                if k not in self._buildin_functions:
                    out[k] = get_func_brief(v)
            else:
                if not rep:
                    if isinstance(v, Ref):
                        out[k] = str(v)  # allow PendingRef when dump
                    else:
                        out[k] = recur_to_allowed_types(v)
                else:
                    out[k] = recur_to_allowed_types(v, extra_allowed=(Ref, Config))
        return out

    def _push(self, **kwargs):
        storage = {}
        for key, value in kwargs.items():
            assert hasattr(
                self, key
            ), f"Invalid {key}! Can only use push for exist params!"
            my_value = self._get(key, deref=False)
            # assert not isinstance(
            #     my_value, Config
            # ), "Can not use push for Config params!"
            storage[key] = my_value
            setattr(self, key, value)
        self._modify_stack.append(storage)

    def _pop(self):
        assert self._modify_stack, "Empty modification stack!"
        storage = self._modify_stack.pop(-1)
        self.merge(storage, exists_only=True)

    @contextmanager
    def modify(self, **kwargs):
        trace_state = self._tracing_set_attribute
        self._tracing_set_attribute = False
        self._push(**kwargs)
        yield self
        self._pop()
        self._tracing_set_attribute = trace_state


class ConfigDiff(Config):
    def __repr__(self):
        from pprint import pformat

        text = [f"{self._class_name}("]

        for k, v in self.to_dict(rep=True).items():
            if isinstance(v, ConfigDiff):
                sub_texts = pformat(v, compact=True).splitlines()
            else:
                vo, vn = v
                t = (
                    f"\033[31m{pformat(vo, compact=True)}\033[0m -> "
                    f"\033[32m{pformat(vn, compact=True)}\033[0m"
                )
                sub_texts = [t]
            sub_texts[0] = f"{k} = {sub_texts[0]}"
            sub_texts[-1] = f"{sub_texts[-1]},"
            for f in sub_texts:
                text.append(f"    " + f)
        text.append(")")
        return "\n".join(text)

    def _A(self):
        root = Config()
        root._class_name = "ValueDiff"
        for k, v in self.to_dict(rep=True).items():
            if isinstance(v, ConfigDiff):
                setattr(root, k, v._A())
            else:
                setattr(root, k, v[0])
        return root

    def _B(self):
        root = Config()
        root._class_name = "ValueDiff"
        for k, v in self.to_dict(rep=True).items():
            if isinstance(v, ConfigDiff):
                setattr(root, k, v._B())
            else:
                setattr(root, k, v[1])
        return root


def build_configdiff_from_flatten(data: dict[str, object]):
    root = ConfigDiff()
    for k, v in data.items():
        _root = root
        keys = k.split(".")
        for key in keys[:-1]:
            if not hasattr(_root, key):
                setattr(_root, key, ConfigDiff())
            _root = _root.get(key)
        setattr(_root, keys[-1], v)
    return root


def inherit_diff(exp: Config):
    exp_module = inspect.getmodule(exp)
    diff_method, new_method = [], []

    def get_super_method(object: object, method_name):
        super_method, base = None, None
        for base in object.__class__.mro()[1:]:
            super_method = getattr(base, method_name, None)
            if super_method:
                base = (
                    inspect.getmodule(super_method).__name__
                    + f".{super_method.__qualname__.split('.')[0]}"
                )
                break
        if isinstance(base, str) and base.startswith("configurize.config"):
            super_method, base = None, None
        return super_method, base

    def _extract_method_diff(cfg: Config):
        keys = cfg.keys() + ["__init__"]
        for k in keys:
            v = super(Config, cfg).__getattribute__(k)
            if isinstance(v, Config):
                _extract_method_diff(v)
            elif callable(v):
                code_module = inspect.getmodule(v)
                if code_module == exp_module:  # defined in this exp
                    now_code = inspect.getsource(v)
                    old_code, father = get_super_method(cfg, k)
                    if old_code:
                        old_code = inspect.getsource(old_code)
                        diff_method.append(
                            (father, old_code, cfg._get_node_name(), now_code)
                        )
                    else:
                        new_method.append((cfg._get_node_name(), now_code))

    _extract_method_diff(exp)
    father_exp = exp.__class__.mro()[1]()
    diff_attr = father_exp.diff(exp)
    return new_method, diff_method, diff_attr


def config_diff(ref: Config, exp: Config):
    diff_method, new_method = [], []

    def _extract_method_diff(ref_cfg: Config, cfg: Config):
        keys = list(set(cfg.keys() + ref_cfg.keys())) + ["__init__"]
        for k in keys:
            try:
                vo = super(type(ref_cfg), ref_cfg).__getattribute__(k)
            except AttributeError:
                vo = None
            try:
                vn = super(type(cfg), cfg).__getattribute__(k)
            except AttributeError:
                vn = None

            if isinstance(vo, Config) and isinstance(vn, Config):
                _extract_method_diff(vo, vn)
            elif callable(vn):
                # 跳过 partial 对象和其他不支持 inspect.getsource 的可调用对象
                if isinstance(vn, partial):
                    continue

                try:
                    now_code = inspect.getsource(vn)
                except (TypeError, OSError):
                    # 如果无法获取源码，跳过
                    continue

                if callable(vo):
                    if isinstance(vo, partial):
                        continue
                    try:
                        old_code = inspect.getsource(vo)
                    except (TypeError, OSError):
                        continue
                    if now_code != old_code:
                        diff_method.append(
                            (
                                ref_cfg._get_node_name(),
                                old_code,
                                cfg._get_node_name(),
                                now_code,
                            )
                        )
                else:
                    new_method.append((cfg._get_node_name(), now_code))

    _extract_method_diff(ref, exp)
    # father_exp = exp.__class__.mro()[1]()
    diff_attr = ref.diff(exp)
    return new_method, diff_method, diff_attr
