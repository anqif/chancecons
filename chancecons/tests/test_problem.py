import numpy as np
import cvxpy.problems.problem as cvxprob
from cvxpy import Variable, Minimize
from cvxpy.atoms import *
from cvxpy.error import SolverError
import chancecons.problem as ccprob
from chancecons import OrderConstraint
from chancecons.tests.base_test import BaseTest

class TestProblem(BaseTest):
	"""Unit test for problems with chance constraints"""
	
	def setUp(self):
		# np.random.seed(1)
		self.A = np.random.randn(200,10)
		self.x = Variable(10, name = "x")
		self.y = Variable(30, name = "y")
		self.tolerance = 1e-8
	
	def test_solve(self):
		b = np.abs(np.random.randn(*self.x.shape))
		obj = sum_squares(self.x - b)
		constr = [OrderConstraint(self.x <= 0, 0.8)]
		p = ccprob.Problem(Minimize(obj), constr)
		p.solve()
		self.assertTrue(np.sum(self.x.value <= self.tolerance) <= 0.8*self.x.size)
		
		b = np.random.randn(self.A.shape[0])
		obj = sum_squares(self.A*self.x - b)
		constr = [OrderConstraint(self.x >= 0, 0.8)]
		p = ccprob.Problem(Minimize(obj), constr)
		p.solve()
		self.assertTrue(np.sum(self.x.value >= -self.tolerance) <= 0.8*self.x.size)
	
	def test_2step(self):
		b = np.abs(np.random.randn(*self.x.shape))
		obj = sum_squares(self.x - b)
		constr = [OrderConstraint(self.x <= 0, 0.8)]
		p = ccprob.Problem(Minimize(obj), constr)
		p.solve(two_step = False)
		val_1step = p.value
		
		p.solve(two_step = True)
		self.assertTrue(p.value <= val_1step)
	
	def test_slack(self):
		b = np.abs(np.random.randn(*self.x.shape))
		obj = sum_squares(self.x - b)
		constr = [OrderConstraint(self.x >= 0, 0.8)]
		p = ccprob.Problem(Minimize(obj), constr)
		
		p.solve(slack = False)
		val_noslack = p.value
		slacks = [cc.slack_value for cc in p.chance_constraints]
		self.assertItemsAlmostEqual(slacks, len(p.chance_constraints)*[0])
		
		p.solve(slack = True)
		self.assertTrue(p.value <= val_noslack)
		slacks = [cc.slack_value for cc in p.chance_constraints]
		for slack in slacks:
			self.assertAlmostGeq(slack, 0)
		
		# Conflicting constraints: Prob(x <= 0) >= 0.2 and x >= 1.
		constr += [self.x >= 1]
		p = ccprob.Problem(Minimize(obj), constr)
		with self.assertRaises(SolverError) as cm:
			p.solve(slack = False)
		
		# Solvable with slackened chance constraint.
		p.solve(slack = True)
		slacks = [cc.slack_value for cc in p.chance_constraints]
		for slack in slacks:
			self.assertAlmostGeq(slack, 0)
