import numpy as np
from cvxpy import Variable, Minimize, Maximize
from cvxpy.atoms import *
import chancecons.problem as ccprob
from chancecons import quantile
from chancecons.tests.base_test import BaseTest

class TestQuantile(BaseTest):
	""" Unit tests for the quantile atom """
	
	def setUp(self):
		# np.random.seed(1)
		self.A = np.random.randn(200,10)
		self.x = Variable(10, name = "x")
		self.tolerance = 1e-8
	
	def test_constraints(self):
		b = np.random.randn(self.A.shape[0])
		obj = sum_squares(self.A*self.x - b)
		constr = [quantile(self.x, 0.5) <= 0]
		prob = ccprob.Problem(obj, constr)
		prob.solve()
		self.assertTrue(np.abs(np.median(self.x.value)) <= self.tolerance)
