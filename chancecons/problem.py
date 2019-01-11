import numpy as np
import cvxpy.settings as s
import cvxpy.problems.problem as cvxprob
import cvxpy.constraints.constraint as cvxcons
from cvxpy import Variable
from cvxpy.error import DCPError, SolverError
from cvxpy.problems.objective import Minimize, Maximize
from cvxpy.constraints.nonpos import Inequality
from cvxpy.constraints.zero import Equality
from chancecons.constraint import OrderConstraint
from chancecons.order2chance import Order2Chance

class Problem(object):
	def __init__(self, objective, constraints = None):
		if constraints is None:
			constraints = []

		# Check that objective is Minimize or Maximize.
		if not isinstance(objective, (Minimize, Maximize)):
			raise ValueError("Problem objective must be Minimize or Maximize.")
			
        # Constraints and objective are immutable.
		self._objective = objective
		self._regular_constraints = []
		self._order_constraints = []
		for constr in constraints:
			if isinstance(constr, OrderConstraint):
				self._order_constraints += [constr]
			elif isinstance(constr, cvxcons.Constraint):
				self._regular_constraints += [constr]
			else:
				raise ValueError("Problem constraints must be of type Constraint or ChanceConstraint")
		
		self._vars = self._variables()
		self._value = None
		self._status = None
		self._solver_stats = None
		self._size_metrics = OrderSizeMetrics(self)
	
	@property
	def value(self):
		return self._value

	@property
	def status(self):
		return self._status
	
	@property
	def objective(self):
		return self._objective
	
	@property
	def constraints(self):
		return self._regular_constraints + self._order_constraints
	
	@property
	def regular_constraints(self):
		return self._regular_constraints[:]
	
	@property
	def order_constraints(self):
		return self._order_constraints[:]
	
	def is_dcp(self):
		return all(exp.is_dcp() for exp in [self.objective] + self.constraints)
	
	def is_mixed_integer(self):
		return any(v.attributes['boolean'] or v.attributes['integer']
					for v in self.variables())
	
	def variables(self):
		"""Accessor method for variables.

		Returns
		-------
		list of :class:`~cvxpy.expressions.variable.Variable`
			A list of the variables in the problem.
		"""
		return self._vars

	def _variables(self):
		vars_ = self.objective.variables()
		for constr in self.constraints:
			vars_ += constr.variables()
		seen = set()
		# never use list as a variable name
		return [seen.add(obj.id) or obj for obj in vars_ if obj.id not in seen]

	def parameters(self):
		"""Accessor method for parameters.

		Returns
		-------
		list of :class:`~cvxpy.expressions.constants.parameter.Parameter`
			A list of the parameters in the problem.
		"""
		params = self.objective.parameters()
		for constr in self.constraints:
			params += constr.parameters()
		return list(set(params))

	def constants(self):
		"""Accessor method for parameters.

		Returns
		-------
		list of :class:`~cvxpy.expressions.constants.constant.Constant`
			A list of the constants in the problem.
		"""
		const_dict = {}
		constants_ = self.objective.constants()
		for constr in self.constraints:
			constants_ += constr.constants()
		# Note that numpy matrices are not hashable, so we use the built-in
		# function "id"
		const_dict = {id(constant): constant for constant in constants_}
		return list(const_dict.values())

	def atoms(self):
		"""Accessor method for atoms.

		Returns
		-------
		list of :class:`~cvxpy.atoms.Atom`
			A list of the atom types in the problem; note that this list
			contains classes, not instances.
		"""
		atoms = self.objective.atoms()
		for constr in self.constraints:
			atoms += constr.atoms()
		return list(set(atoms))
	
	@property
	def size_metrics(self):
		""":class:`~cvxpy.problems.problem.SizeMetrics` : Information about the problem's size.
		"""
		return self._size_metrics

	@property
	def solver_stats(self):
		""":class:`~cvxpy.problems.problem.SolverStats` : Information returned by the solver.
		"""
		return self._solver_stats

	@staticmethod
	def best_subset(margins, max_violations):
		# Convert list of margins into single vector.
		K = int(np.round(max_violations))   # TODO: Should be np.floor, but floating point error causes issues.
		margin_vec = [margin.flatten("C") for margin in margins]
		margin_vec = np.concatenate(margin_vec, axis = 0)
		
		# Form subset vector with False everywhere but the K largest margins.
		idx = np.argsort(margin_vec)[-K:]   # Select K largest margins.
		subset_vec = np.full(margin_vec.shape, False)
		subset_vec[idx] = True
		
		# Reshape subset vector into same shape as list of margins.
		subset = []
		offset = 0
		for margin in margins:
			vec = subset_vec[offset:(offset + margin.size)]
			subset += [np.reshape(vec, margin.shape, order = "C")]
			offset = offset + margin.size
		return subset

	def solve(self, *args, **kwargs):
		use_2step = kwargs.pop("two_step", True)
		use_slack = kwargs.pop("slack", False)
		
		# Reduce quantile atoms in objective.
		original = Problem(self._objective, self.constraints)
		if not Order2Chance().accepts(original):
			raise DCPError("Cannot convert quantiles to order constraints")
		reduced, inv_data = Order2Chance().apply(original)
		
		# First pass with convex restrictions.
		order_constraints = [cc for cc in reduced._order_constraints if cc.fraction != 0]
		for cc in order_constraints:   # Define slack variable for each order constraint.
			cc.slack = Variable(nonneg = True) if use_slack else 0
		restrictions = [cc.restriction for cc in order_constraints]
		constrs1 = reduced._regular_constraints + restrictions
		prob1 = cvxprob.Problem(reduced.objective, constrs1)
		prob1.solve(*args, **kwargs)
		
		# Terminate if first pass does not produce solution.
		if prob1.status not in s.SOLUTION_PRESENT:
			self.save_results(prob1, [Order2Chance()], inv_data)
			raise SolverError("First pass failed with status {0}".format(self.status))
		
		if not use_2step:
			self.save_results(prob1, [Order2Chance()], inv_data)
			return self.value
		
		# Replace chance constraints with exact bounds where solution of
		# first pass yields a relatively low constraint violation.
		constrs2 = reduced._regular_constraints
		for cc in order_constraints:
			subsets = self.best_subset(cc.margins(), (1.0 - cc.fraction)*cc.size)
			for constr, subset in zip(cc.constraints, subsets):
				# if not np.any(subset):
				#	continue
				if isinstance(constr, Inequality):
					constrs2 += [constr.expr[subset] >= 0]   # Flip direction of inequality.
				elif isinstance(constr, Equality):
					constrs2 += [constr.expr[subset] == 0]
				else:
					raise ValueError("Only (<=, ==, >=) constraints supported")
		
		# Second pass with exact bounds.
		prob2 = cvxprob.Problem(reduced.objective, constrs2)
		prob2.solve(*args, **kwargs)
		self.save_results(prob2, [Order2Chance()], inv_data)
		return self.value

	def save_results(self, problem, solving_chain, inv_data):
		# TODO: We don't use inv_data right now because Quantile2Chance
		# just adds epigraph variables, so the reduced problem contains all
		# the variables in the original problem (as well as the same objective).
		self._status = problem.status
		self._value = problem.value
		self._solver_stats = problem.solver_stats
	
	def __str__(self):
		if len(self.constraints) == 0:
			return str(self.objective)
		else:
			subject_to = "subject to "
			lines = [str(self.objective),
					 subject_to + str(self.constraints[0])]
			for constr in self.constraints[1:]:
				lines += [len(subject_to) * " " + str(constr)]
			return "\n".join(lines)
	
	def __repr__(self):
		return "Problem({0}, {1}, {2})".format(repr(self.objective), \
											   repr(self.regular_constraints), repr(self.order_constraints))
	
	def __neg__(self):
		return Problem(-self.objective, self.constraints)

	def __add__(self, other):
		if other == 0:
			return self
		elif not isinstance(other, Problem):
			return NotImplemented
		return Problem(self.objective + other.objective,
					   list(set(self.constraints + other.constraints)))

	def __radd__(self, other):
		if other == 0:
			return self
		else:
			return NotImplemented

	def __sub__(self, other):
		if not isinstance(other, Problem):
			return NotImplemented
		return Problem(self.objective - other.objective,
					   list(set(self.constraints + other.constraints)))

	def __rsub__(self, other):
		if other == 0:
			return -self
		else:
			return NotImplemented

	def __mul__(self, other):
		if not isinstance(other, (int, float)):
			return NotImplemented
		return Problem(self.objective * other, self.constraints)

	__rmul__ = __mul__

	def __div__(self, other):
		if not isinstance(other, (int, float)):
			return NotImplemented
		return Problem(self.objective * (1.0 / other), self.constraints)

	__truediv__ = __div__

class OrderSizeMetrics(cvxprob.SizeMetrics):
	"""Reports various metrics regarding the problem.
	
	Attributes
	----------
	num_scalar_cc_constr : integer
	    The number of scalar order constraints in the problem.
	"""
	def __init__(self, problem):
		self.num_scalar_cc_constr = 0
		for cc in problem.order_constraints:
			self.num_scalar_cc_constr += cc.size
		super(OrderSizeMetrics, self).__init__(problem)
