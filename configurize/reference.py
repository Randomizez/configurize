class CfgReferenceError(Exception):
    def __init__(self, err_str, raw_exec=None):
        super().__init__()
        self.err_str = err_str
        self.raw_exec = raw_exec

    def __str__(self):
        import traceback

        error_hint = ""
        for frame, lineno in traceback.walk_tb(self.__traceback__):
            from configurize import Config

            if isinstance(frame.f_locals.get("self", None), Config):
                if frame.f_code.co_name == "__init__":
                    _self = frame.f_locals["self"]
                    error_hint = f"\n\nHint: Find you access this in {_self._class_name}.__init__() during build!"
        if self.raw_exec:
            raw_tb = traceback.format_exception(self.raw_exec)
            raw_tb = "\n".join(raw_tb)
            error_hint += f"\n\nException during de-ref:\n\n{raw_tb}"
        return self.err_str + error_hint


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
        if self.cur_value is CfgReferenceError:
            return f"❓ PendingRef({self.ref_str})"
        else:
            from .config import Config

            if isinstance(self.cur_value, Config):
                return f"{self.cur_value._get_node_name()} 🈯 Ref({self.ref_str})"
            return f"{repr(self.cur_value)} 🈯 Ref({self.ref_str})"
