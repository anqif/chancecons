import numpy as np
from cvxpy.atoms import pos
from cvxpy.variables import Variable
from cvxpy.constraints import Zero, NonPos

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
		for constr in constraints:
			if not isinstance(constr, (NonPos, Zero)):   # NonPos: expr <= 0, Zero: expr == 0
				raise ValueError("Only (<=, ==, >=) constraints supported")
		if fraction < 0 or fraction > 1:
			raise ValueError("fraction must be in [0,1]")
		self.constraints = constraints
		self.fraction = fraction
		self.restriction = self.restrict(constraints, self.max_violations)
	
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
		return (1 - self.fraction)*self.size
	
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
	
	def margins(self):
		margins = []
		for constr in self.constraints:
			value = constr.expr.value
			if value is not None and isinstance(constr, Zero):
				value = np.abs(value)
			margins += [value]
		return margins
	
	@staticmethod
	def restrict(constraints, max_violations):
		alpha = Variable(nonneg = True)
		restricted = []
		for constr in constraints:
			# Convert expr == 0 to |expr| <= 0 and apply hinge approximation
			expr = abs(constr.expr) if isinstance(constr, Zero) else constr.expr
			restricted += [sum(pos(alpha + expr))]
		return sum(restricted) <= alpha*max_violations
