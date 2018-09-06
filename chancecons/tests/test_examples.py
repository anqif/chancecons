import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from cvxpy import Variable, Minimize, Maximize
from cvxpy.atoms import *
from chancecons import Problem, quantile, prob
from chancecons.tests.base_test import BaseTest

class TestExamples(BaseTest):
	"""Unit tests for chance constraint examples."""
	
	def setUp(self):
		np.random.seed(1)
		self.N = 1000
		self.tolerance = 1e-4

	def test_lp(self):
		n = 10
		m = 100
		c = np.random.randn(n)
		A = np.random.randn(m,n)
		b = np.random.randn(m)
		
		x = Variable(n)
		obj = c.T*x
		constr = [prob(A*x <= b) >= 0.8]
		p = Problem(Maximize(obj), constr)
		p.solve()
		
		print("Objective:", p.value)
		print("Chance constraint fraction:", np.mean((A*x - b).value <= 0))
	
	def test_portfolio(self):
		n = 10
		beta = 0.05
		p_bar = 1.5*np.random.randn(n) + 4
		sigma = 2.5*np.random.randn(n,n)
		sigma = sigma.T.dot(sigma)
		price = np.random.multivariate_normal(p_bar, sigma, self.N)
		
		# Optimal portfolio.
		x = Variable(n)
		ret = sum(price*x)/self.N
		constr = [sum(x) == 1, x >= -0.1, prob(price*x <= 0) <= beta]
		p = Problem(Maximize(ret), constr)
		p.solve(solver = "MOSEK")
		ret_opt = price.dot(x.value)
		print("Optimal portfolio")
		print("Expected return:", ret.value)
		print("Fraction nonpositive:", np.mean(price.dot(x.value) <= 0))
		
		# Optimal portfolio without loss risk constraint.
		constr = [sum(x) == 1, x >= -0.1]
		p = Problem(Maximize(ret), constr)
		p.solve(solver = "MOSEK")
		ret_nocc = price.dot(x.value)
		print("\nOptimal portfolio without loss risk constraint")
		print("Expected return", ret.value)
		print("Fraction nonpositive", np.mean(price.dot(x.value) <= 0))
		
		# Uniform portfolio.
		x_unif = np.ones(n)/n
		ret_unif = price.dot(x_unif)
		print("\nUniform portfolio")
		print("Expected return", np.sum(ret_unif)/self.N)
		print("Fraction nonpositive", np.mean(ret_unif <= 0))
		
		# Plot return distributions.
		rets = [ret_opt, ret_nocc, ret_unif]
		titles = ["Optimal", "Without Loss Constraint", "Uniform"]
		for i in range(len(rets)):
			plt.subplot(3,1,i+1)
			sns.distplot(rets[i], hist = False, kde = True, color = "blue")
			plt.axvline(np.mean(rets[i]), color = "red")
			plt.xlim(-10,20)
			plt.ylim(0,0.225)
			plt.title(titles[i])
			plt.gca().axes.get_yaxis().set_visible(False)
			plt.gca().axes.get_xaxis().set_ticks_position("both")
			plt.gca().axes.tick_params(axis = "x", direction = "in")
		plt.subplots_adjust(hspace = 0.6)
		# plt.savefig("portfolio.pdf", bbox_inches = "tight", pad_inches = 0)
		# plt.show()
	
	def test_treatment(self):
		m = 1000
		n = 200
		labels = (np.random.rand(m) > 0.2).astype(int)
		A = np.random.rand(m,n)
		dose = 1.0
		
		# Dose matrix with target voxels receive ~3x radiation of non-target voxels.
		# Label 0 = tumor, label 1 = organ-at-risk.
		FACTOR = 3
		for i, label in enumerate(labels):
			if label == 0:
				A[i,:] *= FACTOR

		x = Variable(n)
		y = Variable(m)
		y_ptv = y[labels == 0]
		y_oar = y[labels == 1]
		obj = sum(neg(y_ptv - dose) + 2*pos(y_ptv - dose)) + sum(1.6*pos(y_oar))
		constr = [A*x == y, x >= 0, 
				  quantile(y_ptv,0.15) <= 1.05, quantile(y_ptv,0.85) >= 0.9,
				  quantile(y_oar,0.5) <= 0.4]
		p = Problem(Minimize(obj), constr)
		p.solve()
		
		# TODO: Plot cumulative distribution function for 1st and 2nd step.
		# TODO: Add more constraints and enable slack.
