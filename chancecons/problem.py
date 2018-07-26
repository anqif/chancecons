import numpy as np
import cvxpy.setttings as s
import cvxpy.problems.problem as cvxprob
from cvxpy.problems.objective import Minimize, Maximize
from cvxpy.constraints import Zero, NonPos

class Problem(object):
	def __init__(self, objective, constraints = None, chance_constraints = None):
		if constraints is None:
			constraints = []
		if chance_constraints is None:
			chance_constraints = []
		
		# Check that objective is Minimize or Maximize.
        if not isinstance(objective, (Minimize, Maximize)):
            raise TypeError("Problem objective must be Minimize or Maximize.")
        
        # Constraints and objective are immutable.
        self._objective = objective
        self._constraints = [c for c in constraints]
        self._chance_constraints = [c for c in chance_constraints]
        
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
		return self._constraints[:]
	
	@property
	def chance_constraints(self):
		return self._chance_constraints[:]
	
	def is_dcp(self):
		constrs = self.constraints + self.chance_constraints
		return all(exp.is_dcp() for exp in [self.objective] + constrs]
	
	@staticmethod
	def best_subset(self, margins, max_violations):
		# Convert list of margins into single vector
		K = np.floor(max_violations)
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
			subset += [np.reshape(vec, margin.shape, order = "C")
			offset += margin.size
		return subset

    def solve(self, *args, **kwargs):
		# First pass with convex restrictions
		restrictions = [cc.restriction for cc in self._chance_constraints]
		constrs1 = self._constraints + restrictions
		prob1 = cvxprob.Problem(self._objective, constrs1)
		prob1.solve(*args, **kwargs)
		
		# TODO: Handle statuses (e.g., infeasible first pass)
		if prob1.status in s.INF_OR_UNB:
			self._status = prob1.status
			raise Exception("First pass failed with status", self.status)
		
		# Second pass with exact bounds where solution of first pass
		# yields a relatively low constraint violation
		constrs2 = self._constraints
		for cc in self._chance_constraints:
			subsets = self.best_subset(cc.margins, cc.max_violations)
			for constr, subset in zip(cc.constraints, subsets):
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
