import numpy as np
from cvxpy import Variable, Minimize, Maximize
from cvxpy.atoms import *
from cvxpy.error import DCPError
import chancecons.problem as ccprob
from chancecons import order
from chancecons.tests.base_test import BaseTest

class TestOrder(BaseTest):
	""" Unit tests for the order atom """

	def setUp(self):
		np.random.seed(1)
		self.A = np.random.randn(200,10)
		self.x = Variable(10, name = "x")
		self.y = Variable(50, name = "y")
		self.f = np.random.uniform(0,1)
		self.tolerance = 1e-3

	def test_properties(self):
		q = order(self.x, self.f)
		self.assertFalse(q.is_atom_convex())
		self.assertFalse(q.is_atom_concave())
		self.assertTrue(q.is_incr(0))
		self.assertFalse(q.is_decr(0))
		self.assertFalse(q.is_pwl())

		self.assertEqual(order(0, self.f).sign_from_args(), (True, True))
		self.assertEqual(order(1, self.f).sign_from_args(), (True, False))
		self.assertEqual(order(-1, self.f).sign_from_args(), (False, True))

		self.x.value = np.random.randn(10)
		self.assertEqual(q.sign_from_args(), (False, False))
		self.assertItemsAlmostEqual(q.value, np.percentile(self.x.value, 100*self.f, interpolation = "lower"))

	def test_constraints(self):
		obj = norm(self.x)
		constr = [order(self.x, 0.6) <= -1]
		p = ccprob.Problem(Minimize(obj), constr)
		p.solve()
		self.assertTrue(np.sum(self.x.value <= -1 + self.tolerance) >= 0.6*self.x.size)

		b = np.abs(np.random.randn(self.x.size))
		obj = norm(self.x - b)
		constr = [order(self.x, 0.7) >= 1]
		p = ccprob.Problem(Minimize(obj), constr)
		p.solve()
		self.assertTrue(np.sum(self.x.value >= 1 - self.tolerance) >= np.round(1-0.7)*self.x.size)

	def test_reduction_dcp(self):
		obj = norm(self.x)
		constr = [order(self.x, 0.8) == 0]
		p = ccprob.Problem(Minimize(obj), constr)
		with self.assertRaises(DCPError) as cm:
			p.solve()

	def test_reduction_basic(self):
		# Minimize order(y, 0.5) subject to y >= 0.
		t = Variable()
		constr = [order(self.y, 0.5) <= t, self.y >= 0]
		p0 = ccprob.Problem(Minimize(t), constr)
		p0.solve()
		y_epi = self.y.value

		obj = order(self.y, 0.5)
		constr = [self.y >= 0]
		p1 = ccprob.Problem(Minimize(obj), constr)
		p1.solve()
		self.assertAlmostEqual(p1.value, p0.value)
		self.assertItemsAlmostEqual(self.y.value, y_epi)

		# Maximize order(y, 0.75) subject to y <= 5.
		constr = [order(self.y, 0.75) >= t, self.y <= 5]
		p0 = ccprob.Problem(Maximize(t), constr)
		p0.solve()
		y_epi = self.y.value

		obj = order(self.y, 0.75)
		constr = [self.y <= 5]
		p1 = ccprob.Problem(Maximize(obj), constr)
		p1.solve()
		self.assertAlmostEqual(p1.value, p0.value)
		self.assertItemsAlmostEqual(self.y.value, y_epi)

		# Minimize order(abs(A*x - b), 0.5)
		#   subject to 0 <= x <= 1,
		#              order(x, 0.25) >= 0.1
		b = np.random.randn(self.A.shape[0])
		constr = [order(abs(self.A*self.x - b), 0.5) <= t,
				  self.x >= 0, self.x <= 1, order(self.x, 0.25) >= 0.1]
		p0 = ccprob.Problem(Minimize(t), constr)
		p0.solve()
		x_epi = self.x.value

		obj = order(abs(self.A*self.x - b), 0.5)
		constr = [self.x >= 0, self.x <= 1, order(self.x, 0.25) >= 0.1]
		p1 = ccprob.Problem(Minimize(obj), constr)
		p1.solve()
		self.assertAlmostEqual(p1.value, p0.value)
		self.assertItemsAlmostEqual(self.x.value, x_epi)

	def test_reduction_nested(self):
		# Minimize 2*order(y, 0.5)) + 1
		#   subject to y >= 0.
		t = Variable()
		obj = 2*t + 1
		constr = [order(self.y, 0.5) <= t, self.y >= 0]
		p0 = ccprob.Problem(Minimize(obj), constr)
		p0.solve()
		y_epi = self.y.value

		obj = 2*order(self.y, 0.5) + 1
		constr = [self.y >= 0]
		p1 = ccprob.Problem(Minimize(obj), constr)
		p1.solve()
		self.assertAlmostEqual(p1.value, p0.value)
		self.assertItemsAlmostEqual(self.y.value, y_epi)

		# Minimize -2*order(y, 0.75) + 3
		#   subject to y <= 5.
		obj = -2*t + 3
		constr = [order(self.y, 0.75) >= t, self.y <= 5]
		p0 = ccprob.Problem(Minimize(obj), constr)
		p0.solve()
		y_epi = self.y.value

		obj = -2*order(self.y, 0.75) + 3
		constr = [self.y <= 5]
		p1 = ccprob.Problem(Minimize(obj), constr)
		p1.solve()
		self.assertAlmostEqual(p1.value, p0.value)
		self.assertItemsAlmostEqual(self.y.value, y_epi)

		# Minimize order(y, 0.75) - order(y, 0.5)
		u = Variable()
		obj = t - u
		constr = [order(self.y, 0.75) <= t, order(self.y, 0.5) >= u,
				  self.y >= 0, self.y <= 1]
		p0 = ccprob.Problem(Minimize(obj), constr)
		p0.solve()
		y_epi = self.y.value

		obj = order(self.y, 0.75) - order(self.y, 0.5)
		constr = [self.y >= 0, self.y <= 1]
		p1 = ccprob.Problem(Minimize(obj), constr)
		p1.solve()
		self.assertAlmostEqual(p1.value, p0.value)
		self.assertItemsAlmostEqual(self.y.value, y_epi)