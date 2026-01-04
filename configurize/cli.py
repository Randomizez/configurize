#!/usr/bin/env python3
import os
import warnings

warnings.filterwarnings("ignore", ".*")

from configurize import Config
from configurize.utils import compare_in_vscode, get_object_from_file, show_or_compare


def cfshow(ref: str | Config, exp: str | Config = None, key=None, query=None):
    """Show an Exp or compare two Exps

    Usage:
    cfshow my_exp.py
    cfshow base_exp.py new_exp.py
    cfshow my_exp.py --key=model.stem
    """
    from configurize.mock_imports import mock_imports

    Config._tracing_set_attribute = True

    with mock_imports():
        if isinstance(ref, str):
            ref = get_object_from_file(ref, "Exp")()

        if isinstance(exp, str) and os.path.exists(exp):
            exp = get_object_from_file(exp, "Exp")()

    _ref = ref  # hold the reference
    if exp:
        _exp = exp

    key = key.split(".") if key else []
    for k in key:
        ref = getattr(ref, k)
        if exp:
            exp = getattr(exp, k)

    if exp and os.getenv("TERM_PROGRAM") == "vscode":
        compare_in_vscode(ref=ref, exp=exp)
    else:
        print(show_or_compare(ref=ref, exp=exp))

    all_history = ref._all_set_history()
    for k, trace in all_history.items():
        warn = False
        _, _, defined = trace[0]
        if not defined:
            print(f"\033[31m[Undefined] \033[33m{k}\033[0m\n")
            warn = True

        def has_repetition(sequence):
            last_occurrence = {}
            for idx, record in enumerate(sequence):
                if not isinstance(
                    record[0],
                    (str, int, float, bool, type(None), slice, range, tuple),
                ):
                    return False
                if (
                    record[0] in last_occurrence
                    and last_occurrence[record[0]] != idx - 1
                ):
                    return True
                last_occurrence[record[0]] = idx
            return False

        if has_repetition(trace):
            print(f"\033[31m[Confusing] \033[33m{k}\033[0m\n")
            warn = True

        if query and query in k:
            print(f"\033[32m[Diagnose] \033[33m{k}\033[0m\n")
            warn = True
        if warn:
            for set_i, tb in enumerate(trace):
                v, tb, defined = tb
                if set_i == 0:
                    print(f">>> \033[32mINIT\033[0m @ {tb}    = {v}\n")
                else:
                    print(f">>> \033[32mSET \033[0m @ {tb}    = {v}\n")


def main():
    import fire

    fire.Fire(cfshow)


if __name__ == "__main__":
    main()
