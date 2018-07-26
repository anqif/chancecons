import numpy as np
from cvxpy import Variable, Parameter
from cvxpy.atoms import *
from chancecons import ChanceConstraint
from chancecons.tests.base_test import BaseTest

class TestConstraint(BaseTest):
	""" Unit tests for chance constraints"""
	
	def setUp(self):
		np.random.seed(1)
		self.x = Variable(2, name = "x")
		self.y = Variable(3, name = "y")
	
	def test_properties(self):
		self.assertEqual(ChanceConstraints([x >= 0]).size, 2)
		self.assertEqual(ChanceConstraints([x >= 0, y >= 0]).size, 2+3)
		
		self.assertEqual(ChanceConstraints([x >= 0]).max_violations, 0)
		self.assertEqual(ChanceConstraints([x >= 0], 0.8).max_violations, (1-0.8)*2)
		self.assertEqual(ChanceConstraints([x >= 0, y >= 0], 0.8).max_violations, (1-0.8)*(2+3))
