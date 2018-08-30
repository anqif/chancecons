import numpy as np
import cvxpy.lin_ops.lin_utils as lu
from chancecons.constraint import ChanceConstraint
from cvxpy.atoms.atom import Atom
from cvxpy.atoms.axis_atom import AxisAtom

class quantile(AxisAtom):
	"""The q-th quantile of x along the specified axis.
	"""
	def __init__(self, x, q, axis = None, keepdims = False):
		self.q = q
		self.id = lu.get_id()
		super(quantile, self).__init__(x, axis = axis, keepdims = keepdims)
	
	def validate_arguments(self):
		"""Verify that 0 <= q <= 1.
		"""
		if not (self.axis is None and np.isscalar(self.q)):
			raise NotImplementedError
		if self.q < 0 or self.q > 1:
			raise ValueError("Second argument must lie in [0,1]")
		super(quantile, self).validate_arguments()
	
	def name(self):
		return "%s(%s, %s)" % (self.__class__.__name__,
							   self.args[0].name(), 
							   self.q)
	
	@Atom.numpy_numeric
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
	
	def _grad(self, values):
		return self._axis_grad(values)
	
	# Convex restrictions
	def __le__(self, other):
		"""ChanceConstraint : Creates an upper quantile constraint.
		"""
		return ChanceConstraint([self.args[0] >= other], 1.0 - self.q)
	
	def __lt__(self, other):
		"""Unsupported.
		"""
		raise NotImplementedError("Strict inequalities are not allowed.")
	
	def __ge__(self, other):
		"""ChanceConstraint : Creates a lower quantile constraint.
		"""
		return ChanceConstraint([self.args[0] <= other], self.q)
	
	def __gt__(self, other):
		"""Unsupported.
		"""
		raise NotImplementedError("Strict inequalities are not allowed.")

def median(x, axis = None, keepdims = False):
	return quantile(x, 0.5, axis = axis, keepdims = keepdims)

def percentile(x, q, axis = None, keepdims = False):
	return quantile(x, q/100.0, axis = axis, keepdims = keepdims)
