"""Minimal numba stub for importing old outlines in non-guided vLLM serving.

The regular generation path does not use outlines' numba-compiled regex FSMs.
This stub lets vLLM 0.5.1 import its guided-decoding modules without pulling a
heavy numba/llvmlite stack into the shared conda environment.
"""


class _NumbaScalar:
    def __call__(self, value=0):
        return int(value)

    def __getitem__(self, _item):
        return self


class _Types:
    unicode_type = str

    @staticmethod
    def ListType(_item_type):
        return list

    @staticmethod
    def UniTuple(_item_type, _count):
        return tuple

    @staticmethod
    def Tuple(_item_types):
        return tuple


class _TypedDict(dict):
    @classmethod
    def empty(cls, *_args, **_kwargs):
        return cls()


class _TypedList(list):
    @classmethod
    def empty_list(cls, *_args, **_kwargs):
        return cls()


class _Typed:
    Dict = _TypedDict
    List = _TypedList


def njit(*_args, **_kwargs):
    if _args and callable(_args[0]) and len(_args) == 1 and not _kwargs:
        return _args[0]

    def decorator(func):
        return func

    return decorator


int64 = _NumbaScalar()
uint64 = _NumbaScalar()
types = _Types()
typed = _Typed()
