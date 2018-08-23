import numpy as np
from cvxpy.atoms.atom import AxisAtom

class quantile(AxisAtom):
	"""The q-th quantile of x along the specified axis.
	"""
	def __init__(self, x, q, axis = None, keepdims = False):
		self.q = q
		super(percentile, self).__init__(x, axis = axis, keepdims = keepdims)
	
	def validate_arguments(self):
		"""Verify that 0 <= q <= 1.
		"""
		if self.axis is not None or len(self.q) > 1:
			raise NotImplementedError
		if any(self.q < 0 or self.q > 1):
			raise ValueError("Second argument must lie in [0,1]")
		super(percentile, self).validate_arguments()
	
	def name(self):
        return "%s(%s, %s)" % (self.__class__.__name__,
							   self.args[0].name(), 
                               self.q)
	
	def numeric(self, values):
		"""Returns the q-th percentile of x along the specified axis.
		"""
		return np.percentile(values[0], self.q, axis = self.axis)
	
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
		return [self.q, self.axis]
	
	def _domain(self):
        """Returns constraints describing the domain of the node.
        """
        return []
    
    # Convex restrictions
    def __eq__(self, other):
        """Unsupported.
        """
        raise NotImplementedError("Strict equalities are not allowed.")

    def __le__(self, other):
        """ChanceConstraint : Creates an upper percentile constraint.
        """
        return ChanceConstraint([self.args[0] - other], 1.0-self.q)

    def __lt__(self, other):
        """Unsupported.
        """
        raise NotImplementedError("Strict inequalities are not allowed.")

    def __ge__(self, other):
        """ChanceConstraint : Creates a lower percentile constraint.
        """
        return ChanceConstraint([other - self.args[0]], self.q)

    def __gt__(self, other):
        """Unsupported.
        """
        raise NotImplementedError("Strict inequalities are not allowed.")
