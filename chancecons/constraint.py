import numpy as np
import matplotlib.pyplot as plt
from cvxpy import Variable, Zero, NonPos
from cvxpy.constraints.constraint import Constraint
from cvxpy.atoms import *

class ChanceConstraint(object):
	"""A chance constraint requires at least a given fraction of the specified
	sub-constraints to hold. For instance, if x  is a variable and f(x) has
	dimension n, the constraint Prob(f(x) <= 0) >= p is interpreted as f_i(x) <= 0
	for at least np indices i = 1,...,n.
	
	Multiple sub-constraints will be treated as if their expressions are vectorized
	and stacked in the form f(x) <= 0.
	"""
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
	
	def name(self):
		return "({0}) for {1} of total elements" \
					.format(", ".join(constr.name() for constr in self.constraints), self.fraction)
	
	def __str__(self):
		"""Returns a string showing the mathematical constraint.
		"""
		return self.name()
	
	def __repr__(self):
		"""Returns a string with information about the constraint.
		"""
		return "%s(%s, %s)" % (self.__class__.__name__,
						       repr(self.constraints),
						       self.fraction)
	
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
		return list(set(param for arg in self.constraints for param in arg.parameters()))

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
	
	def plot_cdf(self, *args, **kwargs):
		margins = self.margins()
		if any(margin is None for margin in margins):
			raise Exception("One or more margins is None.")
		margin_vec = [margin.flatten("C") for margin in margins]
		margin_vec = np.concatenate(margin_vec)
		
		x = np.sort(margin_vec)
		y = np.arange(x.size)/float(x.size)
		plt.plot(x, y, *args, **kwargs)
		plt.axvline(0, color = 'grey', linestyle = '--')

		# Trace fraction horizontally to curve, then vertically to x-intercept.
		# plt.axhline(self.fraction, linestyle = '--')
		idx = np.argmin(np.abs(y - self.fraction))
		plt.plot(x[:(idx + 1)], np.full(idx + 1, self.fraction), color = 'grey', linestyle = '--')
		plt.plot(np.full(idx + 1, x[idx]), y[:(idx + 1)], color = 'grey', linestyle = '--')
		
		plt.ylim(0, 1)
		plt.xlim(x[0], x[-1])
		plt.show()
	
	@property
	def restriction(self):
		restricted = []
		for constr in self.constraints:
			# Convert expr == 0 to |expr| <= 0 and apply hinge approximation
			expr = abs(constr.expr) if isinstance(constr, Zero) else constr.expr
			restricted += [sum(pos(self.slope + expr))]
		return sum(restricted) <= self.slope*self.max_violations

class prob(object):
	"""Syntatic sugar for constructing chance constraints.
	"""
	def __init__(self, *args):
		self.constraints = list(args)
	
	def __eq__(self, other):
        """Unsupported.
        """
        raise NotImplementedError("Strict equalities are not allowed.")

    def __le__(self, other):
        """ChanceConstraint : Creates an upper chance constraint.
        """
        flipped = []
        for constr in self.constraints:
			c = constr.copy()
			c.args[0] = -c.args[0]
			flipped += [c]
        return ChanceConstraint(flipped, 1.0-other)

    def __lt__(self, other):
        """Unsupported.
        """
        raise NotImplementedError("Strict inequalities are not allowed.")

    def __ge__(self, other):
        """ChanceConstraint : Creates a lower chance constraint.
        """
        return ChanceConstraint(self.constraints, other)

    def __gt__(self, other):
        """Unsupported.
        """
        raise NotImplementedError("Strict inequalities are not allowed.")
