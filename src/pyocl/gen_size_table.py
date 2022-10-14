import math, pathlib
import numpy as np


def split_int_2(aint):
    return tuple((x, aint-x) for x in range(0, aint+1))

def split_int(aint, nints):
    assert nints > 0
    if nints==1:
        return ((aint,),)
    split = split_int_2(aint)
    nints -= 2
    while nints > 0:
        split = tuple((*p[:-1], *s) for p in split for s in split_int_2(p[-1]))
        nints -= 1
    return split

def sumul(iterable):
    s = 1
    for i in iterable:
        s *= i
    return s

# Hope no user will use size bigger than 2^32
def calculate(blen=32, save_path=None, quicker=False):
    print('Calculating legal sizes for clFFT...')
    if quicker:
        print('Using quick mode.')
        available_primes = (3, 5, 7, 11, 13)
        maxpow = math.floor(math.log(2**blen, 3))
    else:
        available_primes = (2, 3, 5, 7, 11, 13)
        maxpow = blen

    pow_distros = sum([list(split_int(i+1, len(available_primes))) for i in range(maxpow)], [])

    available_sizes = [sumul(p**d for p, d in zip(available_primes, dist)) for dist in pow_distros]
    available_sizes.sort()
    available_sizes = [x for x in available_sizes if x.bit_length() <= blen]

    available_sizes = np.array(available_sizes)

    print('Got %d available sizes in range of %d.'%(len(available_sizes), 2**blen))

    if save_path:
        np.save(save_path, available_sizes, allow_pickle=False)
        print('Saved in %s.'%str(save_path))
    else:
        print('Not saved because path to save is not given')

if __name__=='__main__':
    print('This is a tester.')
    calculate()