import numpy as np
from cvxpy import Variable, Minimize, Maximize
from cvxpy.atoms import *
import chancecons.problem as ccprob
from chancecons import quantile, ChanceConstraint
from chancecons.tests.base_test import BaseTest

class TestQuantile(BaseTest):
	""" Unit tests for the quantile atom """
	
	def setUp(self):
		# np.random.seed(1)
		self.A = np.random.randn(200,50)
		self.x = Variable(50, name = "x")
		self.tolerance = 1e-3
	
	def test_constraints(self):
		b = np.random.randn(self.A.shape[0])
		obj = sum_squares(self.A*self.x - b)
		constr = [quantile(self.x, 0.5) <= -5]
		prob = ccprob.Problem(Minimize(obj), constr)
		prob.solve()
		print(np.median(self.x.value))
		self.assertAlmostLeq(np.median(self.x.value), -5, self.tolerance)
		
		constr = [quantile(self.x, 0.8) >= 5]
		prob = ccprob.Problem(Minimize(obj), constr)
		prob.solve()
		self.assertAlmostGeq(np.percentile(self.x.value, 80), 5, self.tolerance)
