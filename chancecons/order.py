import numpy as np
import cvxpy.lin_ops.lin_utils as lu
from chancecons.constraint import OrderConstraint
from chancecons.utilities import count_to_frac
from cvxpy.atoms.atom import Atom
from cvxpy.atoms.axis_atom import AxisAtom

class order(AxisAtom):
    """The f-th smallest element (by fraction) of x along the specified axis.
    """
    def __init__(self, x, f, axis = None, keepdims = False):
        self.id = lu.get_id()
        self.f = np.array(f) if isinstance(f, list) else f
        super(order, self).__init__(x, axis = axis, keepdims = keepdims)

    def validate_arguments(self):
        """Verify that 0 <= f <= 1.
        """
        if not (self.axis is None and np.isscalar(self.f)):   # TODO: Remove when axis constraints are implemented.
            raise NotImplementedError
        if not np.all(self.f >= 0 and self.f <= 1):
            raise ValueError("Second argument can only contain numbers in [0,1]")

    def name(self):
        return "%s(%s, %s)" % (self.__class__.__name__,
                               self.args[0].name(),
                               self.f)

    @Atom.numpy_numeric
    def numeric(self, values):
        """Returns the f-th smallest element (by fraction) of x along the specified axis.
        """
        return np.percentile(values[0], 100*self.f, axis = self.axis, interpolation = "lower")

    def sign_from_args(self):
        """Returns sign (is positive, is negative) of the expression.
        """
        return (self.args[0].is_nonneg(), self.args[0].is_nonpos())

    def is_atom_convex(self):
        """Is the atom convex?
        """
        return self.f == 1

    def is_atom_concave(self):
        """Is the atom concave?
        """
        return self.f == 0

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
        return [self.f, self.axis]

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
        if np.isscalar(self.f):
            return OrderConstraint([self.args[0] >= other], 1.0 - self.f)
        else:
            return [OrderConstraint([self.args[0] >= other], 1.0 - f) for f in self.f]

    def __lt__(self, other):
        """Unsupported.
        """
        raise NotImplementedError("Strict inequalities are not allowed.")

    def __ge__(self, other):
        """OrderConstraint : Creates a lower order constraint.
        """
        if np.isscalar(self.f):
            return OrderConstraint([self.args[0] <= other], self.f)
        else:
            return [OrderConstraint([self.args[0] <= other], f) for f in self.f]

    def __gt__(self, other):
        """Unsupported.
        """
        raise NotImplementedError("Strict inequalities are not allowed.")

def kth_smallest(x, k, axis = None, keepdims = False):
    """The k-th smallest element of x along the specified axis.
    """
    p, n = count_to_frac(x, k, axis)
    return order(x, p, axis, keepdims)

def kth_largest(x, k, axis = None, keepdims = False):
    """The k-th largest element of x along the specified axis.
    """
    p, n = count_to_frac(x, k, axis)
    return order(x, 1 - p + 1.0/n, axis, keepdims)
