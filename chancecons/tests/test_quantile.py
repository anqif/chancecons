import numpy as np
from cvxpy import Variable, Minimize
from cvxpy.atoms import *
import chancecons.problem as ccprob
from chancecons import quantile, order
from chancecons.tests.base_test import BaseTest

class TestQuantile(BaseTest):
	""" Unit tests for the quantile atom """
	
	def setUp(self):
		# np.random.seed(1)
		self.x = Variable(10, name = "x")
		self.f = np.random.uniform(0,1)
		self.tolerance = 1e-3
		self.interpolation = ["linear", "lower", "higher", "nearest", "midpoint"]
	
	def test_corner(self):
		self.x.value = np.random.randn(10)
		self.assertItemsAlmostEqual(quantile(self.x, 0).value, np.min(self.x.value))
		self.assertItemsAlmostEqual(quantile(self.x, 1).value, np.max(self.x.value))

	def test_interpolation(self):
		a = np.array(np.arange(11))
		self.assertAlmostEqual(quantile(a, 0.175, interpolation = "linear").value, 1.75)
		self.assertAlmostEqual(quantile(a, 0.175, interpolation = "lower").value, 1.0)
		self.assertAlmostEqual(quantile(a, 0.175, interpolation = "higher").value, 2.0)
		self.assertAlmostEqual(quantile(a, 0.175, interpolation = "nearest").value, 2.0)
		self.assertAlmostEqual(quantile(a, 0.175, interpolation = "midpoint").value, 1.5)

		self.x.value = np.random.randn(10)
		for intp in self.interpolation:
			q = quantile(self.x, self.f, interpolation = intp)
			self.assertItemsAlmostEqual(q.value, np.percentile(self.x.value, 100*self.f, interpolation = intp))

	def test_constraints_linear(self):
		# Minimize norm(x) subject to quantile(x, 0.65) <= -1.
		obj = norm(self.x)
		constr = [quantile(self.x, 0.65, interpolation = "linear") <= -1]
		p0 = ccprob.Problem(Minimize(obj), constr)
		p0.solve()
		x_quant = self.x.value

		h = (self.x.size - 1)*0.65
		lower = order(self.x, 0.6)
		upper = order(self.x, 0.7)
		quant_expr = lower + (h - np.floor(h))*(upper - lower)
		constr = [quant_expr <= -1]
		p1 = ccprob.Problem(Minimize(obj), constr)
		p1.solve()

		self.assertAlmostEqual(p1.value, p0.value)
		self.assertItemsAlmostEqual(self.x.value, x_quant)
		self.assertAlmostEqual(quant_expr.value, np.percentile(self.x.value, 100*0.65, interpolation = "linear"))

		# Minimize norm(x) subject to quantile(x, 0.8) >= 1.
		constr = [quantile(self.x, 0.825, interpolation = "linear") >= 1]
		p0 = ccprob.Problem(Minimize(obj), constr)
		p0.solve()
		x_quant = self.x.value

		h = (self.x.size - 1)*0.825
		lower = order(self.x, 0.8)
		upper = order(self.x, 0.9)
		quant_expr = lower + (h - np.floor(h))*(upper - lower)
		constr = [quant_expr >= 1]
		p1 = ccprob.Problem(Minimize(obj), constr)
		p1.solve()

		self.assertAlmostEqual(p1.value, p0.value)
		self.assertItemsAlmostEqual(self.x.value, x_quant)
		self.assertAlmostEqual(quant_expr.value, np.percentile(self.x.value, 100*0.825, interpolation = "linear"))
