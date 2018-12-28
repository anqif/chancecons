import numpy as np
from cvxpy import Variable, Minimize, Maximize
from cvxpy.atoms import *
import chancecons.problem as ccprob
from chancecons import quantile
from chancecons.tests.base_test import BaseTest

class TestQuantile(BaseTest):
	""" Unit tests for the quantile atom """
	
	def setUp(self):
		np.random.seed(1)
		self.A = np.random.randn(200,10)
		self.x = Variable(10, name = "x")
		self.y = Variable(50, name = "y")
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
		self.assertItemsAlmostEqual(q.value, np.percentile(self.x.value, 100*self.f))
	
	def test_constraints(self):
		obj = norm(self.x)
		constr = [quantile(self.x, 0.6) <= -1]
		p = ccprob.Problem(Minimize(obj), constr)
		p.solve()
		self.assertTrue(np.sum(self.x.value <= -1 + self.tolerance) >= 0.6*self.x.size)
		
		b = np.abs(np.random.randn(self.x.size))
		obj = norm(self.x - b)
		constr = [quantile(self.x, 0.7) >= 1]
		p = ccprob.Problem(Minimize(obj), constr)
		p.solve()
		self.assertTrue(np.sum(self.x.value >= 1 - self.tolerance) >= np.round(1-0.7)*self.x.size)
	
	def test_reduction_basic(self):
		# Minimize quantile(y, 0.5) subject to y >= 0.
		t = Variable()
		constr = [quantile(self.y, 0.5) <= t, self.y >= 0]
		p0 = ccprob.Problem(Minimize(t), constr)
		p0.solve()
		y_epi = self.y.value
		
		obj = quantile(self.y, 0.5)
		constr = [self.y >= 0]
		p1 = ccprob.Problem(Minimize(obj), constr)
		p1.solve()
		self.assertAlmostEqual(p1.value, p0.value)
		self.assertItemsAlmostEqual(self.y.value, y_epi)
		
		# Maximize quantile(y, 0.75) subject to y <= 5.
		constr = [quantile(self.y, 0.75) >= t, self.y <= 5]
		p0 = ccprob.Problem(Maximize(t), constr)
		p0.solve()
		y_epi = self.y.value
		
		obj = quantile(self.y, 0.75)
		constr = [self.y <= 5]
		p1 = ccprob.Problem(Maximize(obj), constr)
		p1.solve()
		self.assertAlmostEqual(p1.value, p0.value)
		self.assertItemsAlmostEqual(self.y.value, y_epi)
		
		# Minimize quantile(abs(A*x - b), 0.5)
		#   subject to 0 <= x <= 1,
		#              quantile(x, 0.25) >= 0.1
		b = np.random.randn(self.A.shape[0])
		constr = [quantile(abs(self.A*self.x - b), 0.5) <= t,
				  self.x >= 0, self.x <= 1, quantile(self.x, 0.25) >= 0.1]
		p0 = ccprob.Problem(Minimize(t), constr)
		p0.solve()
		x_epi = self.x.value
		
		obj = quantile(abs(self.A*self.x - b), 0.5)
		constr = [self.x >= 0, self.x <= 1, quantile(self.x, 0.25) >= 0.1]
		p1 = ccprob.Problem(Minimize(obj), constr)
		p1.solve()
		self.assertAlmostEqual(p1.value, p0.value)
		self.assertItemsAlmostEqual(self.x.value, x_epi)
	
	def test_reduction_nested(self):
		# Minimize 2*quantile(y, 0.5)) + 1
		#   subject to y >= 0.
		t = Variable()
		obj = 2*t + 1
		constr = [quantile(self.y, 0.5) <= t, self.y >= 0]
		p0 = ccprob.Problem(Minimize(obj), constr)
		p0.solve()
		y_epi = self.y.value
		
		obj = 2*quantile(self.y, 0.5) + 1
		constr = [self.y >= 0]
		p1 = ccprob.Problem(Minimize(obj), constr)
		p1.solve()
		self.assertAlmostEqual(p1.value, p0.value)
		self.assertItemsAlmostEqual(self.y.value, y_epi)
		
		# Minimize -2*quantile(y, 0.75) + 3
		#   subject to y <= 5.
		obj = -2*t + 3
		constr = [quantile(self.y, 0.75) >= t, self.y <= 5]
		p0 = ccprob.Problem(Minimize(obj), constr)
		p0.solve()
		y_epi = self.y.value
		
		obj = -2*quantile(self.y, 0.75) + 3
		constr = [self.y <= 5]
		p1 = ccprob.Problem(Minimize(obj), constr)
		p1.solve()
		self.assertAlmostEqual(p1.value, p0.value)
		self.assertItemsAlmostEqual(self.y.value, y_epi)
		
		# Minimize quantile(y, 0.75) - quantile(y, 0.5)
		u = Variable()
		obj = t - u
		constr = [quantile(self.y, 0.75) <= t, quantile(self.y, 0.5) >= u,
				  self.y >= 0, self.y <= 1]
		p0 = ccprob.Problem(Minimize(obj), constr)
		p0.solve()
		y_epi = self.y.value
		
		obj = quantile(self.y, 0.75) - quantile(self.y, 0.5)
		constr = [self.y >= 0, self.y <= 1]
		p1 = ccprob.Problem(Minimize(obj), constr)
		p1.solve()
		self.assertAlmostEqual(p1.value, p0.value)
		self.assertItemsAlmostEqual(self.y.value, y_epi)
