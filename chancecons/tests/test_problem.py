import numpy as np
import cvxpy.problems.problem as cvxprob
from cvxpy import Variable, Parameter, Minimize
from cvxpy.atoms import *
import chancecons.problem as ccprob
from chancecons import ChanceConstraint
from chancecons.tests.base_test import BaseTest

class TestProblem(BaseTest):
	"""Unit test for problems with chance constraints"""
	
	def setUp(self):
		# np.random.seed(1)
		self.A = np.random.randn(200,10)
		self.x = Variable(10, name = "x")
		self.y = Variable(30, name = "y")
	
	def test_solve(self):
		b = np.random.randn(self.A.shape[0])
		obj = sum_squares(self.A*self.x - b)
		constr = [ChanceConstraint(self.x >= 0, 0.8)]
		prob = ccprob.Problem(Minimize(obj), constr)
		prob.solve(solver = "MOSEK")
		
		self.assertTrue(np.sum(self.x.value >= 0) >= 0.8*self.x.size)
