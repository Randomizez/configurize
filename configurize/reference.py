class CfgReferenceError(LookupError):
    def __str__(self):
        import traceback

        error_hint = ""
        for frame, lineno in traceback.walk_tb(self.__traceback__):
            from configurize import Config

            if isinstance(frame.f_locals.get("self", None), Config):
                if frame.f_code.co_name == "__init__":
                    _self = frame.f_locals["self"]
                    error_hint = f"\n\nHint: Find you access this in {_self._class_name}.__init__() during build!"
        return (
            super().__str__()
            + (
                "\n\nReference Error took place when you try to access a Ref() but we can't find "
                "the target in config tree. There might be 2 potential reason: \n"
                "1. Typo in your ref string or wrong path.\n"
                "2. The Ref() is accessed when the tree is not fully built. NOTE that "
                "you should not access a Ref() during bulding (e.g. in __init__())!"
            )
            + error_hint
        )


class Ref:
    """Create a Reference to Another arg, use:
      ".a.b" for self.a.b
      "..a.b" for father.a.b

    e.g.

    class Model:
        arg_from_father = Ref('..the_arg')

    class Exp:
        model = Model
        the_arg = 'arg from father'

    print(Exp())

    {
        "model":{"arg_from_father": "arg from father"},
        "the_arg": "arg from father"
    }
    """

    def __init__(self, ref_str: str, default=CfgReferenceError):
        assert ref_str.startswith(".")
        self.ref_str = ref_str
        self.actions = self._parse_level(ref_str)
        self.default = default
        self.cur_value = default

    @staticmethod
    def _parse_level(ref_str):
        levels = []
        i = 1
        while i < len(ref_str):
            if ref_str[i] == ".":
                levels.append(".")
            else:
                if (j := ref_str[i:].find(".")) == -1:
                    levels.append(ref_str[i:])
                    break
                else:
                    levels.append(ref_str[i : i + j])
                i += j
            i += 1
        return levels

    def __repr__(self) -> str:
        if self.cur_value is KeyError:
            return f"❓ PendingRef({self.ref_str})"
        else:
            return f"{repr(self.cur_value)} 🈯 Ref({self.ref_str})"
