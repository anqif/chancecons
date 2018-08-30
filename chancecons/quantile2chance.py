import chancecons as cc
from cvxpy import Variable
from cvxpy.reductions import InverseData, Reduction, Solution
from cvxpy.problems.objective import Minimize, Maximize
from cvxpy.error import DCPError

class Quantile2Chance(Reduction):
	"""Replace quantile atoms with the appropriate chance constraints.
	"""
	
	def accepts(self, problem):
		return True
	
	def apply(self, problem):
		inverse_data = InverseData(problem)
		is_minimize = type(problem.objective) == Minimize
		chance_expr, chance_constraints = self.chance_tree(problem.objective.args[0], is_minimize, True, False)
		chance_objective = Minimize(chance_expr) if is_minimize else Maximize(chance_expr)
		
		if any([type(atom) == cc.quantile for con in problem.constraints for atom in con.atoms()]):
			raise DCPError("Quantile atom may not be nested in constraints.")
		new_problem = cc.problem.Problem(chance_objective, problem.constraints + chance_constraints)
		return new_problem, inverse_data
	
	def invert(self, solution, inverse_data):
		pvars = {vid: solution.primal_vars[vid] for vid in inverse_data.id_map
				 if vid in solution.primal_vars}
		dvars = {orig_id: solution.dual_vars[vid]
				 for orig_id, vid in inverse_data.cons_id_map.items()
				 if vid in solution.dual_vars}
		return Solution(solution.status, solution.opt_val, pvars, dvars,
						solution.attr)

	def chance_tree(self, expr, is_minimize, is_incr, is_decr):
		chance_args = []
		constrs = []
		for i, arg in enumerate(expr.args):
			is_obj_incr = (is_incr and expr.is_incr(i)) or (is_decr and expr.is_decr(i))
			is_obj_decr = (is_incr and expr.is_decr(i)) or (is_decr and expr.is_incr(i))
			chance_arg, c = self.chance_tree(arg, is_minimize, is_obj_incr, is_obj_decr)
			chance_args += [chance_arg]
			constrs += c
		chance_expr, c = self.chance_expr(expr, chance_args, is_minimize, is_incr, is_decr)
		constrs += c
		return chance_expr, constrs

	def chance_expr(self, expr, args, is_minimize, is_incr, is_decr):
		if type(expr) == cc.quantile:
			t = Variable(expr.shape)
			if (is_incr and is_minimize) or (is_decr and not is_minimize):
				constr = expr <= t   # TODO: Do we need expr.copy(args) here?
			elif (is_incr and not is_minimize) or (is_decr and is_minimize):
				constr = expr >= t
			else:
				raise DCPError("Objective must be non-decreasing or non-increasing in each quantile argument.")
			return t, [constr]
		else:
			return expr.copy(args), []
