import numpy as np
import cvxpy.settings as s
import cvxpy.problems.problem as cvxprob
from cvxpy.problems.objective import Minimize, Maximize
from cvxpy.constraints import Zero, NonPos
from chancecons.constraint import ChanceConstraint

class Problem(object):
	def __init__(self, objective, constraints = None):
		if constraints is None:
			constraints = []

		# Check that objective is Minimize or Maximize.
		if not isinstance(objective, (Minimize, Maximize)):
			raise TypeError("Problem objective must be Minimize or Maximize.")
			
        # Constraints and objective are immutable.
		self._objective = objective
		self._regular_constraints = []
		self._chance_constraints = []
		for constr in constraints:
			if isinstance(constr, ChanceConstraint):
				self._chance_constraints += [constr]
			else:
				self._regular_constraints += [constr]
		
		self._vars = self._variables()
		self._value = None
		self._status = None
	
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
		return self._regular_constraints + self._chance_constraints
	
	@property
	def regular_constraints(self):
		return self._regular_constraints[:]
	
	@property
	def chance_constraints(self):
		return self._chance_constraints[:]
	
	def is_dcp(self):
		constrs = self.constraints + self.chance_constraints
		return all(exp.is_dcp() for exp in [self.objective] + constrs)
	
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

	@staticmethod
	def best_subset(margins, max_violations):
		# Convert list of margins into single vector
		K = int(np.round(max_violations))   # TODO: Should be np.floor, but floating point error causes issues
		margin_vec = [margin.flatten("C") for margin in margins]
		margin_vec = np.concatenate(margin_vec, axis = 0)
		
		# Form subset vector with True everywhere but the K largest margins
		idx = np.argsort(margin_vec)[-K:]   # Select K largest margins
		subset_vec = np.full(margin_vec.shape, True)
		subset_vec[idx] = False
		
		# Reshape subset vector into same shape as list of margins
		subset = []
		offset = 0
		for margin in margins:
			vec = subset_vec[offset:(offset + margin.size)]
			subset += [np.reshape(vec, margin.shape, order = "C")]
			offset = offset + margin.size
		return subset

	def solve(self, *args, **kwargs):
		# First pass with convex restrictions
		restrictions = [cc.restriction for cc in self._chance_constraints]
		constrs1 = self._regular_constraints + restrictions
		prob1 = cvxprob.Problem(self._objective, constrs1)
		prob1.solve(*args, **kwargs)
		
		# TODO: Handle statuses (e.g., infeasible first pass)
		if prob1.status in s.INF_OR_UNB:
			self._status = prob1.status
			raise Exception("First pass failed with status", self.status)
		
		# Second pass with exact bounds where solution of first pass
		# yields a relatively low constraint violation
		constrs2 = self._regular_constraints
		for cc in self._chance_constraints:
			if cc.fraction == 0:   # Drop chance constraints that are not enforced
				continue
			
			subsets = self.best_subset(cc.margins(), cc.max_violations)
			for constr, subset in zip(cc.constraints, subsets):
				# if not np.any(subset):
				#	continue
				if isinstance(constr, NonPos):
					constrs2 += [constr.expr[subset] <= 0]
				elif isinstance(constr, Zero):
					constrs2 += [constr.expr[subset] == 0]
				else:
					raise ValueError("Only (<=, ==, >=) constraints supported")
		
		prob2 = cvxprob.Problem(self._objective, constrs2)
		prob2.solve(*args, **kwargs)   # TODO: Use warm start
		
		# TODO: Handle statuses and save results
		self._status = prob2.status
		self._value = prob2.value
		return self.value
