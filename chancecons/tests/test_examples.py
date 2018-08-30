import numpy as np
from cvxpy import Variable, Minimize, Maximize
from cvxpy.atoms import *
from chancecons import Problem, quantile, prob
from chancecons.tests.base_test import BaseTest

class TestExamples(BaseTest):
	"""Unit tests for chance constraint examples."""
	
	def setUp(self):
		self.c = np.random.uniform(1,3,100)
		self.tolerance = 1e-4
		self.x = Variable(100, name = "x")

	def test_lp(self):
		obj = 100*sum(self.x)
		constr = [prob(self.c.T*self.x >= 1) >= 0.95, self.x >= self.tolerance]
		p = Problem(Minimize(obj), constr)
		p.solve(slack = False)
