from __future__ import annotations

import inspect
import os
import sys
import types
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from .config import Config


def is_log_rank():
    return int(os.getenv("RANK", "0")) == 0


def get_func_brief(func):
    try:
        signature = inspect.signature(func)
    except:
        signature = "(?) -> ?"
    return f"Func{signature}"


def filter_traceback_frames(
    module_pattern, tb: types.TracebackType = None
) -> types.TracebackType:
    """过滤特定模块的traceback帧"""
    if tb is None:
        _, _, tb = sys.exc_info()
    filtered: list[types.FrameType] = []
    while tb is not None:
        # 检查模块名是否匹配排除条件
        frame = tb.tb_frame
        module_name = frame.f_globals.get("__name__", "")
        if not module_name.startswith(module_pattern):
            # 保留非目标模块的帧
            filtered.append(frame)
        tb = tb.tb_next

    new_tb = None

    # 反向构建新traceback链（FILO）
    for frame in reversed(filtered):
        new_tb = types.TracebackType(
            new_tb,
            tb_frame=frame,
            tb_lasti=frame.f_lasti,
            tb_lineno=frame.f_lineno,
        )
    return new_tb


def get_object_from_file(file: str, name: str = "Exp") -> object:
    """
    get object by file.
    Args:
        file (str): file path.
        name (str): object name.
    """
    import importlib
    import sys

    module_name_without_ext = os.path.splitext(os.path.basename(file))[0]
    directory_path = os.path.dirname(file)
    sys.path.insert(0, directory_path)
    current_exp = importlib.import_module(module_name_without_ext)
    sys.path.pop(0)
    obj = getattr(current_exp, name)
    return obj


def writable_property(fget: Callable[[Any], Any]):
    """
    把普通的 getter 包装成“可写属性”。
    读取顺序：
        1. 若实例已经保存了覆盖值 → 返回该值；
        2. 否则调用原来的 getter。
    写入时直接把值存进实例（隐藏属性），以后读取都会走第 1 步。
    """

    class _WritableProperty(property):
        def __init__(self, func: Callable[[Any], Any]) -> None:
            self.raw_fget = func
            # 把文档字符串复制过去，保持和普通 property 一样的行为
            self.__doc__ = getattr(func, "__doc__", None)
            self.name: str | None = None  # 在 __set_name__ 时填充
            self.storage_name: str | None = None  # 实际保存值的实例属性名

        # Python 会在类创建阶段把属性名传进来，这里顺手记下来
        def __set_name__(self, owner, name: str) -> None:
            self.name = name
            # 为了不和用户可能自己使用的同名属性冲突，采用一个内部名字保存覆盖值
            self.storage_name = f"_{name}_override"

        # ----------- 读取 ----------
        def __get__(self, instance, owner=None):
            if instance is None:  # 通过类访问时返回 descriptor 本身（property 的惯例）
                return self
            # 已经被覆盖 → 直接返回保存的值
            if self.storage_name in instance.__dict__:
                return instance.__dict__[self.storage_name]
            # 否则走原来的 getter
            return self.raw_fget(instance)

        # ----------- 写入 ----------
        def __set__(self, instance, value):
            # 把覆盖值写进实例的隐藏属性
            instance.__dict__[self.storage_name] = value

        # 可选：del obj.x 让属性恢复为“未覆盖”状态
        def __delete__(self, instance):
            if self.storage_name in instance.__dict__:
                del instance.__dict__[self.storage_name]

    return _WritableProperty(fget)


def cdiff(a, b):
    def colored(text):
        if text.startswith("+"):
            text = f"\033[32m{text}\033[0m"
        elif text.startswith("-"):
            text = f"\033[31m{text}\033[0m"
        elif text.startswith("?"):
            text = f"\033[33m{text}\033[0m"
        return text

    from difflib import ndiff

    ans = "\n".join(
        [colored(x) for x in ndiff([str(x) for x in a], [str(x) for x in b])]
    )
    return ans


def sanity_check(exp: Config) -> str:
    """Check an Exp without initialize anything."""
    import traceback

    try:
        exp.sanity_check()
    except Exception as e:
        tb = traceback.format_exception(e)[-2:]
        return "\n".join(tb)
    return None


def show_or_compare(ref: Config, exp: Config = None):

    text = ""

    def prt(x):
        nonlocal text
        text += str(x) + "\n"

    if exp is None:
        prt(ref)

        err = sanity_check(ref)
        if err is None:
            err = f"\033[32mOK\033[0m"
        else:
            err = f"\033[31m{err}\033[0m"
        prt(f"sanity_check: {err}")
    else:
        from configurize import config_diff

        diff = cdiff(str(ref).splitlines(), str(exp).splitlines())

        prt(diff)

        new_method, diff_method, diff_attr = config_diff(ref, exp)
        prt(f"\033[33m<< Diff Attributes >>\033[0m")
        prt(diff_attr)

        prt(f"\033[36m<< New-Defined Methods >>\033[0m")
        for new, new_code in new_method:
            prt(f"\n\033[36m{new}\033[0m")
            prt(new_code)

        prt(f"\033[33m<< Diff Methods >>\033[0m")
        for old, old_code, new, new_code in diff_method:
            prt(f"\n\033[33m{new}\033[0m")
            prt(cdiff(old_code.splitlines(), new_code.splitlines()))

    return text


def compare_in_vscode(ref: Config, exp: Config = None):
    from .config import config_diff, inherit_diff

    textA = ""

    def prtA(x):
        nonlocal textA
        textA += str(x) + "\n"

    textB = ""

    def prtB(x):
        nonlocal textB
        textB += str(x) + "\n"

    def prt(x):
        nonlocal textA, textB
        textA += str(x) + "\n"
        textB += str(x) + "\n"

    if exp is None:
        new_method, diff_method, diff_attr = inherit_diff(ref)
        print(ref)
    else:
        new_method, diff_method, diff_attr = config_diff(ref, exp)

    prt(f"# << Diff Attributes >>")
    prtA(diff_attr._A())
    prtB(diff_attr._B())

    prt(f"# << New-Defined Methods >>")
    for new, new_code in new_method:
        prt(f"\nclass {new.replace('.', '__')}:")
        prtB(new_code)

    prt(f"# << Diff Methods >>")
    for old, old_code, new, new_code in diff_method:
        prt(f"\nclass {new.replace('.', '__')}:")
        prtA(old_code)
        prtB(new_code)

    with open("/tmp/A.py", "w") as f:
        f.write(textA)
    with open("/tmp/B.py", "w") as f:
        f.write(textB)
    import subprocess

    try:
        subprocess.run(["code", "-d", "/tmp/A.py", "/tmp/B.py"], check=True)
    except FileNotFoundError:
        subprocess.run(["cursor", "-d", "/tmp/A.py", "/tmp/B.py"])
    except subprocess.CalledProcessError as e:
        print(
            f"An error occurred while trying to open the diff: {e}\n\nTry run 'diff /tmp/A.py /tmp/B.py' directly."
        )
