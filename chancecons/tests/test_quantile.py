import numpy as np
from cvxpy import Variable
from chancecons import quantile
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
		self.assertItemsAlmostEqual(quantile(a, 0.175, interpolation = "linear").value, 1.75)
		self.assertItemsAlmostEqual(quantile(a, 0.175, interpolation = "lower").value, 1.0)
		self.assertItemsAlmostEqual(quantile(a, 0.175, interpolation = "higher").value, 2.0)
		self.assertItemsAlmostEqual(quantile(a, 0.175, interpolation = "nearest").value, 2.0)
		self.assertItemsAlmostEqual(quantile(a, 0.175, interpolation = "midpoint").value, 1.5)

		self.x.value = np.random.randn(10)
		for intp in self.interpolation:
			q = quantile(self.x, self.f, interpolation = intp)
			self.assertItemsAlmostEqual(q.value, np.percentile(self.x.value, 100*self.f, interpolation = intp))
