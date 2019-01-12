import numpy as np
from chancecons.order import kth_smallest, kth_largest
from chancecons.utilities import max_elems

def quantile(a, q, axis = None, interpolation = "higher", keepdims = False):
	if not np.isscalar(q):
		raise NotImplementedError
	if q < 0 or q > 1:
		raise ValueError("q must lie in [0,1]")
	n = max_elems(a, axis)
	h = (n-1)*q + 1   # Add 1 since k = 1,...,n.

	if interpolation == "linear":
		if q < 1/n:
			return kth_smallest(a, 1, axis, keepdims)
		elif q == 1:
			return kth_largest(a, 1, axis, keepdims)
		else:
			lower = kth_smallest(a, np.floor(h), axis, keepdims)
			upper = kth_smallest(a, np.floor(h) + 1, axis, keepdims)
			frac = h - np.floor(h)
			return lower + frac*(upper - lower)
	elif interpolation == "lower":
		return kth_smallest(a, np.floor(h), axis, keepdims)
	elif interpolation == "higher":
		return kth_smallest(a, np.ceil(h), axis, keepdims)
	elif interpolation == "nearest":
		return kth_smallest(a, np.round(h), axis, keepdims)
	elif interpolation == "midpoint":
		if q == 0:
			return kth_smallest(a, 1, axis, keepdims)
		elif q == 1:
			return kth_largest(a, 1, axis, keepdims)
		else:
			lower = kth_smallest(a, np.floor(h), axis, keepdims)
			upper = kth_smallest(a, np.floor(h) + 1, axis, keepdims)
			return (lower + upper)/2
	else:
		raise ValueError("interpolation can only be 'linear', 'lower', 'higher', 'midpoint', or 'nearest'")

def median(x, axis = None, keepdims = False):
	return quantile(x, 0.5, axis = axis, keepdims = keepdims)

def percentile(x, q, axis = None, keepdims = False):
	return quantile(x, q/100.0, axis = axis, keepdims = keepdims)
