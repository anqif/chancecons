import numpy as np
from cvxpy import Variable, Minimize
from cvxpy.atoms import *
from cvxpy.error import DCPError
import chancecons.problem as ccprob
from chancecons import ChanceConstraint, prob
from chancecons.tests.base_test import BaseTest

class TestConstraint(BaseTest):
	""" Unit tests for chance constraints"""
	
	def setUp(self):
		# np.random.seed(1)
		self.A = np.random.randn(10,2)
		self.B = np.random.randn(15,3)
		
		self.x = Variable(2, name = "x")
		self.y = Variable(3, name = "y")
		self.U = Variable((2,2), name = "U")
		self.V = Variable((3,2), name = "V")
	
	def test_properties(self):
		self.assertEqual(ChanceConstraint([self.x >= 0]).size, 2)
		self.assertEqual(ChanceConstraint([self.x >= 0, self.y >= 0]).size, 2+3)
		self.assertEqual(ChanceConstraint([self.U >= 0, self.V >= 0]).size, 2*2+3*2)
		
		self.assertEqual(ChanceConstraint([self.x >= 0]).max_violations, 2)
		self.assertEqual(ChanceConstraint([self.x >= 0], 0.8).max_violations, 0.8*2)
		self.assertEqual(ChanceConstraint([self.x >= 0, self.y >= 0], 0.8).max_violations, 0.8*(2+3))
		self.assertEqual(ChanceConstraint([self.U >= 0, self.V >= 0], 0.8).max_violations, 0.8*(2*2+3*2))

	def test_margins(self):
		cc = ChanceConstraint([self.x <= 0])
		self.assertItemsAlmostEqual(cc.margins(), [None])
		self.x.value = [-1,1]
		margins = cc.margins()
		self.assertEqual(len(margins), 1)
		self.assertItemsAlmostEqual(margins[0], self.x.value, places = 8)
		
		cc.constraints = [self.x >= 0]
		self.assertItemsAlmostEqual(cc.margins()[0], -self.x.value, places = 8)
		
		cc.constraints = [self.x >= 0, self.y <= 0]
		margins = cc.margins()
		self.assertEqual(len(margins), 2)
		self.assertItemsAlmostEqual(margins[0], -self.x.value, places = 8)
		self.assertEqual(margins[1], None)
		self.y.value = [-5,0,10]
		self.assertItemsAlmostEqual(cc.margins()[1], self.y.value)
		
		b = np.random.randn(self.A.shape[0])
		cc = ChanceConstraint([self.A*self.x == b])
		self.x.value = [-1,1]
		margins = cc.margins()
		self.assertItemsAlmostEqual(margins[0], np.abs(self.A.dot(self.x.value) - b))
	
	def test_margins_2d(self):
		cc = ChanceConstraint([self.U <= 0, min(self.V) >= 5])
		self.U.value = np.array([[1,2], [3,4]])
		self.V.value = np.array([[-2,3], [-1,2], [0,1]])
		margins = cc.margins()
		self.assertEqual(len(margins), 2)
		self.assertItemsAlmostEqual(margins[0], self.U.value, places = 8)
		self.assertItemsAlmostEqual(margins[1], 5-np.min(self.V.value), places = 8)

	def test_restriction(self):
		cc = ChanceConstraint([self.x <= 0], 0.8)
		self.assertEqual(cc.restriction.shape, ())
		self.assertEqual(cc.restriction.size, 1)
		self.assertEqual(cc.restriction.expr.value, None)
		
		self.x.value = [-1,1]
		cc.slope.value = 0.5
		val = np.sum(np.maximum(0.5 - self.x.value, 0)) - 0.5*0.8*2
		self.assertItemsAlmostEqual(cc.restriction.expr.value, val)
		
		cc.constraints = [self.x >= 0]
		val = np.sum(np.maximum(0.5 - self.x.value, 0)) - 0.5*0.8*2
		self.assertItemsAlmostEqual(cc.restriction.expr.value, val)
		
		cc.fraction = 0.4
		cc.slope.value = 1.5
		val = np.sum(np.maximum(1.5 - self.x.value, 0)) - 1.5*0.4*2
		self.assertItemsAlmostEqual(cc.restriction.expr.value, val)
		
		b = np.random.randn(self.A.shape[0])
		cc =  ChanceConstraint([self.A*self.x == b], 0.8)
		self.x.value = [-1,1]
		cc.slope.value = 0.5
		expr = np.abs(self.A.dot(self.x.value) - b)
		val = np.sum(np.maximum(0.5 - expr, 0)) - 0.5*0.8*self.A.shape[0]
		self.assertItemsAlmostEqual(cc.restriction.expr.value, val)
	
	def test_restriction_2d(self):
		cc = ChanceConstraint([self.U <= 0], 0.8)
		self.U.value = np.array([[-1,-2], [3,4]])
		cc.slope.value = 0.5
		val = np.sum(np.maximum(0.5 - self.U.value, 0)) - 0.5*0.8*(2*2)
		self.assertItemsAlmostEqual(cc.restriction.expr.value, val)
	
	def test_dcp(self):
		# Prob(g(x) >= 0) <= p is DCP iff g is convex.
		constr = [ChanceConstraint(log(self.x) >= 0, 0.5)]
		p = ccprob.Problem(Minimize(norm(self.x)), constr)
		with self.assertRaises(DCPError) as cm:
			p.solve()
		
		# Prob(h(x) <= 0) <= p is DCP iff h is concave.
		constr = [ChanceConstraint(exp(self.x) <= 5, 0.8)]
		p = ccprob.Problem(Minimize(norm(self.x)), constr)
		with self.assertRaises(DCPError) as cm:
			p.solve()
	
	def test_probability(self):
		b = np.random.randn(self.A.shape[0])
		obj = sum_squares(self.A*self.x - b)
		
		cc = ChanceConstraint([self.x <= 0], 0.8)
		pc = prob(self.x <= 0) <= 0.8
		self.assertEqual(cc.size, pc.size)
		self.assertEqual(cc.fraction, pc.fraction)
		self.assertEqual(cc.max_violations, pc.max_violations)
		self.assertAlmostEqual(ccprob.Problem(Minimize(obj), [cc]).solve(), 
							   ccprob.Problem(Minimize(obj), [pc]).solve())
		
		cc = ChanceConstraint([self.x >= 0], 0.8)
		pc = prob(self.x <= 0) >= 0.2
		self.assertEqual(cc.size, pc.size)
		self.assertEqual(cc.fraction, pc.fraction)
		self.assertEqual(cc.max_violations, pc.max_violations)
		self.assertAlmostEqual(ccprob.Problem(Minimize(obj), [cc]).solve(), 
					   ccprob.Problem(Minimize(obj), [pc]).solve())
