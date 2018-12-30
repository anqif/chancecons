import numpy as np
import cvxpy.lin_ops.lin_utils as lu
from chancecons.constraint import OrderConstraint
from cvxpy.atoms.atom import Atom
from cvxpy.atoms.axis_atom import AxisAtom

def max_elems(x, axis = None):
    if axis is None:
        return x.size
    else:
        return np.prod([x.shape[i] for i in axis])

class smallest(AxisAtom):
    """The k-th smallest element of x along the specified axis.
    """

    def __init__(self, x, k, axis = None, keepdims = False):
        self.id = lu.get_id()
        self.k = [int(k)] if np.isscalar(k) else [int(k_elem) for k_elem in k]
        self._k_max = max_elems(x, axis)
        super(smallest, self).__init__(x, axis = axis, keepdims = keepdims)

    def validate_arguments(self):
        """Verify that 1 <= k <= n where n is the total number of elements in x along the axis.
        For multiple axes, we take the product of the corresponding dimensions.
        """
        if not (self.axis is None and len(self.k) == 1):
            raise NotImplementedError
        if not np.all([np.isscalar(k_elem) and k_elem >= 1 and k_elem <= self._k_max for k_elem in self.k]):
            raise ValueError("Second argument can only contain integers in [1,%d]" % int(self._k_max))

    def name(self):
        return "%s(%s, %s)" % (self.__class__.__name__,
                               self.args[0].name(),
                               self.k)

    @Atom.numpy_numeric
    def numeric(self, values):
        """
        Returns the k-th smallest element of x along the specified axis.
        """
        p = 100*(self.k - 1)/(self._k_max - 1.0)
        return np.percentile(values[0], p, axis = self.axis, interpolation = "lower")

    def sign_from_args(self):
        """Returns sign (is positive, is negative) of the expression.
        """
        return (self.args[0].is_nonneg(), self.args[0].is_nonpos())

    def is_atom_convex(self):
        """Is the atom convex?
        """
        return False

    def is_atom_concave(self):
        """Is the atom concave?
        """
        return False

    def is_incr(self, idx):
        """Is the composition non-decreasing in argument idx?
        """
        return True

    def is_decr(self, idx):
        """Is the composition non-increasing in argument idx?
        """
        return False

    def is_pwl(self):
        """Is the atom piecewise linear?
        """
        return False

    def get_data(self):
        return [self.k, self.axis]

    def _domain(self):
        """Returns constraints describing the domain of the node.
        """
        return []

    def _grad(self, values):
        return self._axis_grad(values)

    # Convex restrictions
    def __le__(self, other):
        """OrderConstraint : Creates an upper order constraint.
        """
        return [OrderConstraint([self.args[0] >= other], self._k_max-k) for k in self.k]

    def __lt__(self, other):
        """Unsupported.
        """
        raise NotImplementedError("Strict inequalities are not allowed.")

    def __ge__(self, other):
        """OrderConstraint : Creates a lower order constraint.
        """
        return [OrderConstraint([self.args[0] <= other], k-1) for k in self.k]

    def __gt__(self, other):
        """Unsupported.
        """
        raise NotImplementedError("Strict inequalities are not allowed.")

def largest(x, k, axis = None, keepdims = False):
    """The k-th largest element of x along the specified axis.
    """
    n = max_elems(x, axis)
    return smallest(x, n-k+1, axis, keepdims)
