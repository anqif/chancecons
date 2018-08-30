import numpy as np
from cvxpy import Variable, Parameter, Minimize
from cvxpy.atoms import *
import chancecons.problem as ccprob
from chancecons import quantile, ChanceConstraint
from chancecons.tests.base_test import BaseTest

class TestQuantile(BaseTest):
	""" Unit tests for the quantile atom """
	
	def setUp(self):
		np.random.seed(1)
		self.A = np.random.randn(200,10)
		self.x = Variable(10, name = "x")
		self.f = np.random.uniform(0,1)
		self.tolerance = 1e-3
	
	def test_properties(self):
		q = quantile(self.x, self.f)
		self.assertFalse(q.is_atom_convex())
		self.assertFalse(q.is_atom_concave())
		self.assertTrue(q.is_incr(0))
		self.assertFalse(q.is_decr(0))
		self.assertFalse(q.is_pwl())
		
		self.assertEqual(quantile(0, self.f).sign_from_args(), (True, True))
		self.assertEqual(quantile(1, self.f).sign_from_args(), (True, False))
		self.assertEqual(quantile(-1, self.f).sign_from_args(), (False, True))
		
		self.x.value = np.random.randn(10)
		self.assertEqual(q.sign_from_args(), (False, False))
		self.assertItemsAlmostEqual(q.value, np.percentile(self.x.value, self.f))
	
	def test_constraints(self):
		obj = norm(self.x)
		constr = [quantile(self.x, 0.6) <= -1]
		prob = ccprob.Problem(Minimize(obj), constr)
		prob.solve()
		self.assertTrue(np.sum(self.x.value <= -1 + self.tolerance) >= 0.6*self.x.size)
		
		b = np.abs(np.random.randn(self.x.size))
		obj = norm(self.x - b)
		constr = [quantile(self.x, 0.7) >= 1]
		prob = ccprob.Problem(Minimize(obj), constr)
		prob.solve()
		self.assertTrue(np.sum(self.x.value >= 1 - self.tolerance) >= np.round(1-0.7)*self.x.size)
	
	def test_reduction(self):
		obj = quantile(self.x, 0.5)
		prob = ccprob.Problem(Minimize(obj))
		prob.solve()
