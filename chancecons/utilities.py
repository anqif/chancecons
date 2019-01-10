import numpy as np

def max_elems(x, axis = None):
    if axis is None:
        return x.size
    else:
        return np.prod([x.shape[i] for i in axis])

def count_to_frac(x, k, axis = None):
    n = max_elems(x, axis)
    k = np.array(k) if isinstance(k, list) else k
    if not np.all(k >= 1 and k <= n):
        raise ValueError("Second argument can only contain integers in [1,%d]" % n)
    return k/n, n
