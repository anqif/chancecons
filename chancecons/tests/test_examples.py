import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from cvxpy import Variable, Minimize, Maximize
from cvxpy.atoms import *
from cvxpy.error import SolverError
from chancecons import ChanceConstraint, Problem, quantile, prob
from chancecons.tests.base_test import BaseTest

def plot_cdf(x, *args, **kwargs):
	x_sort = np.sort(x)
	prob = np.arange(1,len(x)+1)/len(x)
	plt.plot(x_sort, prob, *args, **kwargs)
	plt.ylim(0,1)

class TestExamples(BaseTest):
	"""Unit tests for chance constraint examples."""
	
	def setUp(self):
		np.random.seed(1)
		self.N = 1000
		self.markersize = 10
		self.tolerance = 1e-4

	def test_lp(self):
		n = 10
		c = np.random.randn(n)
		sigma = np.eye(n)
		A = np.random.multivariate_normal(c, sigma, self.N)
		
		x = Variable(n)
		obj = c.T*x
		constr = [prob(A*x >= 0) <= 0.75, x >= 0, x <= 1]
		p = Problem(Maximize(obj), constr)
		p.solve()
		
		print("Objective:", p.value)
		print("Chance constraint fraction:", np.mean(A.dot(x.value) <= 0))
	
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
	
	def test_radiation(self):
		m = 1000   # voxels
		n = 200    # beams
		dose = 1.0
		A = np.random.rand(m,n)
		labels = (np.random.rand(m) > 0.2).astype(int)
		
		# Dose matrix with target voxels receive ~3x radiation of non-target voxels.
		# Label 0 = tumor, label 1 = organ-at-risk.
		FACTOR = 3
		for i, label in enumerate(labels):
			if label == 0:
				A[i,:] *= FACTOR
		
		# Solve radiation treatment planning problem.
		x = Variable(n)
		y = Variable(m)
		y_ptv = y[labels == 0]
		y_oar = y[labels == 1]
		obj = sum(neg(y_ptv - dose) + 2*pos(y_ptv - dose)) + sum(1.6*pos(y_oar))
		constr = [A*x == y, x >= 0, 
				  quantile(y_ptv,0.15) <= 1.0, quantile(y_ptv,0.85) >= 0.95,
				  quantile(y_oar,0.5) <= 0.35]
		p = Problem(Minimize(obj), constr)
		
		# Plot CDF for 1st and 2nd step.
		p.solve(two_step = False, slack = False)
		plot_cdf(y_oar.value, "b--")
		plot_cdf(y_ptv.value, "r--")
		
		p.solve(two_step = True, slack = False)
		plot_cdf(y_oar.value, "b-")
		plot_cdf(y_ptv.value, "r-")
		
		# Label constraints on plot.
		plt.plot(1.0, 0.15, marker = "<", markersize = self.markersize, color = "red")
		plt.plot(0.95, 0.85, marker = ">", markersize = self.markersize, color = "red")
		plt.plot(0.35, 0.5, marker = "<", markersize = self.markersize, color = "blue")
		plt.axvline(dose, color = "red", linestyle = ":", linewidth = 1.0)
		plt.xlim(0,1.2)
		# plt.savefig("radiation_2step.pdf", bbox_inches = "tight", pad_inches = 0)
		# plt.show()
	
	def test_radiation_slack(self):
		m = 1000   # voxels
		n = 200    # beams
		dose = 1.0
		A = np.random.rand(m,n)
		labels = (np.random.rand(m) > 0.2).astype(int)
		
		# Dose matrix with target voxels receive ~3x radiation of non-target voxels.
		# Label 0 = tumor, label 1 = organ-at-risk.
		FACTOR = 3
		for i, label in enumerate(labels):
			if label == 0:
				A[i,:] *= FACTOR
		
		# Solve radiation treatment planning problem.
		x = Variable(n)
		y = Variable(m)
		y_ptv = y[labels == 0]
		y_oar = y[labels == 1]
		obj = 2.5*sum(neg(y_ptv - dose) + 2*pos(y_ptv - dose)) + sum(1.6*pos(y_oar))
		constr = [A*x == y, x >= 0, quantile(y_ptv,0.85) >= 0.95]
		p = Problem(Minimize(obj), constr)
		p.solve(two_step = True, slack = False)
		
		# Plot CDF of 2-step solution.
		# plot_cdf(y_oar.value, "b-")
		# plot_cdf(y_ptv.value, "r-")
		# plt.plot(0.95, 0.85, marker = ">", markersize = self.markersize, color = "red")
		# plt.show()
		
		# First pass infeasible with additional constraint.
		constr += [quantile(y_oar,0.3) <= 0.15]
		p = Problem(Minimize(obj), constr)
		with self.assertRaises(SolverError) as cm:
			p.solve(slack = False)
		
		# Plot CDF of slack solution with constraint.
		p.solve(two_step = False, slack = True)
		plot_cdf(y_oar.value, "b--")
		plot_cdf(y_ptv.value, "r--")
		q1 = quantile(y_oar,0.3).value
		
		p.solve(two_step = True, slack = True)
		plot_cdf(y_oar.value, "b-")
		plot_cdf(y_ptv.value, "r-")
		
		# Label original and slack bound.
		plt.plot(0.95, 0.85, marker = ">", markersize = self.markersize, color = "red")
		plt.plot(q1, 0.3, marker = "<", markersize = self.markersize, color = "blue")
		plt.plot(0.15, 0.3, marker = "<", markersize = self.markersize, color = "blue")
		plt.plot([0.15, q1], [0.3, 0.3], "b-")
		plt.axvline(dose, color = "red", linestyle = ":", linewidth = 1.0)
		plt.xlim(0,1.2)
		# plt.savefig("radiation_slack.pdf", bbox_inches = "tight", pad_inches = 0)
		# plt.show()
