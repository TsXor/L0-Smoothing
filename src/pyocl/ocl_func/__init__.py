import pathlib
from importlib.machinery import SourceFileLoader
CODEDIR = pathlib.Path(__file__).parent

from .cl_function import CLFunction

def _import_func(fname):
    mod = SourceFileLoader(fname, str(CODEDIR / ('%s.py'%fname))).load_module()
    pyfunc = getattr(mod, fname)
    with open(CODEDIR / ('%s.cl'%fname)) as clcodefp:
        clcode = clcodefp.read()
        clfunc = CLFunction(fname, clcode, pyfunc)
    globals()[fname] = clfunc

funcs = [
    'circshift2D',
    'fancyindex2D',
    'pad2D_constant',
]

for fn in funcs:
    _import_func(fn)


__all__ = []
__all__ += funcs