import numpy as np
import pathlib


self_dir = pathlib.Path(__file__).parent
blen = 32; quicker = False
save_fn = 'sizetable_%d_%squick.npy'
cache_path = self_dir / (save_fn%(blen, '' if quicker else 'no'))

available_sizes = None


def get_cache(func):
    def with_cache(*args, **kwargs):
        global available_sizes
        if available_sizes is None:
            if not cache_path.is_file():
                from .gen_size_table import calculate
                calculate(blen, cache_path, quicker)
                print('This will only be cauculated once and result will be cached.')
            available_sizes = np.load(cache_path, allow_pickle=False)
        return func(*args, **kwargs)
    return with_cache

def cut_trailing_zero(x):
    '''Cuts the trailing zero of binary and return number of zeros.'''
    cutted = 0
    while (~x&1):
        x = x>>1
        cutted += 1
    return x, cutted

@get_cache
def get_nearest_bigger(x, quicker=False):
    if quicker:
        x, offset = cut_trailing_zero(x)
    idx = np.searchsorted(available_sizes, x, side='left')
    ret = available_sizes[idx]
    if quicker:
        ret = ret >> offset
    return ret