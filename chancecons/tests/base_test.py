"""
Copyright 2018 Anqi Fu

This file is part of CVXConsensus.

CVXConsensus is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

CVXConsensus is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with CVXPY.  If not, see <http://www.gnu.org/licenses/>.
"""

# Base class for unit tests.
from unittest import TestCase
import numpy as np
import matplotlib.pyplot as plt

class BaseTest(TestCase):
	# AssertAlmostEqual for lists.
	def assertItemsAlmostEqual(self, a, b, places=4):
		if np.isscalar(a):
			a = [a]
		else:
			a = self.mat_to_list(a)
		if np.isscalar(b):
			b = [b]
		else:
			b = self.mat_to_list(b)
		for i in range(len(a)):
			self.assertAlmostEqual(a[i], b[i], places)
	
	# Overriden method to assume lower accuracy.
	def assertAlmostEqual(self, a, b, places=4):
		super(BaseTest, self).assertAlmostEqual(a, b, places=places)
	
	def assertAlmostLeq(self, a, b, tol=1e-4):
		self.assertTrue(a - b <= tol)
	
	def assertAlmostGeq(self, a, b, tol=1e-4):
		self.assertTrue(a - b >= -tol)
	
	def mat_to_list(self, mat):
		"""Convert a numpy matrix to a list.
		"""
		if isinstance(mat, (np.matrix, np.ndarray)):
			return np.asarray(mat).flatten('F').tolist()
		else:
			return mat
	
	def plot_cdf(self, x, *args, **kwargs):
		x_sort = np.sort(x)
		prob = np.arange(1,len(x)+1)/len(x)
		handle = plt.plot(x_sort, prob, *args, **kwargs)
		plt.ylim(0,1)
		return handle
	
	def plot_abline(self, slope, intercept, *args, **kwargs):
		"""Plot a line from slope and intercept"""
		axes = plt.gca()
		x_vals = np.array(axes.get_xlim())
		y_vals = intercept + slope * x_vals
		plt.plot(x_vals, y_vals, *args, **kwargs)
