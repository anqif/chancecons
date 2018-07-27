import numpy as np
from cvxpy import Variable, Zero, NonPos
from cvxpy.constraints.constraint import Constraint
from cvxpy.atoms import *

# Add expression: Quantile([A*x <= b, ...], fraction)
# 1. Replace with variable t.
# 2. Add chance constraint a_i^Tx - b <= t for fraction i's
# Only works for convex inequality constraints f_i(x) <= 0.

# Look up references, this is a special case of majorization-minimization
# on constraints with only 2 steps.
class ChanceConstraint(object):
	def __init__(self, constraints = None, fraction = 1.0):
		if constraints is None:
			constraints = []
		elif isinstance(constraints, Constraint):
			constraints = [constraints]
		
		for constr in constraints:
			if not isinstance(constr, (NonPos, Zero)):   # NonPos: expr <= 0, Zero: expr == 0
				raise ValueError("Only (<=, ==, >=) constraints supported")
		if fraction < 0 or fraction > 1:
			raise ValueError("fraction must be in [0,1]")
		self.constraints = constraints
		self.fraction = fraction
		self.slope = Variable(nonneg = True)
	
	@property
	def id(self):
		return self.restriction.id
	
	@property
	def size(self):
		return sum([constr.size for constr in self.constraints])
	
	@property
	def residual(self):
		# return self.restriction.residual
		raise NotImplementedError   # TODO: What is the correct residual?
	
	@property
	def dual_value(self):
		raise NotImplementedError   # TODO: Save value of dual variable (on restriction)
	
	@property
	def max_violations(self):
		return (1.0 - self.fraction)*self.size
	
	def is_real(self):
		return not self.is_complex()
	
	def is_imag(self):
		return all(constr.is_imag() for constr in self.constraints)
	
	def is_complex(self):
		return any(constr.is_complex() for constr in self.constraints)
	
	def is_dcp(self):
		return all(constr.is_dcp() and isinstance(constr, (NonPos, Zero)) for constr in self.constraints)
	
	def violation(self):
		residual = self.residual
		if residual is None:
			raise ValueError("Cannot compute the violation of a chance "
							 "constraint whose expression is None-valued.")
		return residual
    
	def value(self, tolerance = 1e-8):
		residual = self.residual
		if residual is None:
			raise ValueError("Cannot compute the value of a chance "
							 "constraint whose expression is None-valued.")
		return np.all(residual <= tolerance)
    
	def get_data(self):
		return [self.id]
	
	def save_dual_value(self, dual_value):
		# self.dual_variables[0].save_value(dual_value)
		raise NotImplementedError

	def variables(self):
		"""Returns all the variables present in the constraints including
		the chance constraint slope.
		"""
		# Remove duplicates.
		return [self.slope] + list(set(var for arg in self.constraints for var in arg.variables()))

	def parameters(self):
		"""Returns all the parameters present in the constraints.
		"""
		# Remove duplicates.
		return list(set(param for cons in self.constraints for param in arg.parameters()))

	def constants(self):
		"""Returns all the constants present in the constraints.
		"""
		const_list = (const for arg in self.constraints for const in arg.constants())
		# Remove duplicates:
		const_dict = {id(constant): constant for constant in const_list}
		return list(const_dict.values())
	
	def atoms(self):
		"""Returns all the atoms present in the constraints.
		"""
		# Remove duplicates.
		return list(set(atom for arg in self.constraints for atom in arg.atoms()))
	
	def margins(self):
		margins = []
		for constr in self.constraints:
			value = constr.expr.value
			if value is not None and isinstance(constr, Zero):
				value = np.abs(value)
			margins += [value]
		return margins
	
	@property
	def restriction(self):
		restricted = []
		for constr in self.constraints:
			# Convert expr == 0 to |expr| <= 0 and apply hinge approximation
			expr = abs(constr.expr) if isinstance(constr, Zero) else constr.expr
			restricted += [sum(pos(self.slope + expr))]
		return sum(restricted) <= self.slope*self.max_violations
