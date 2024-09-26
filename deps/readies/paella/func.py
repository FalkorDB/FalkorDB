
from functools import reduce

foldl = lambda func, seq, z: reduce(func, z, seq)
foldr = lambda func, seq, xs: reduce(lambda x, y: func(y, x), xs[::-1], seq)
