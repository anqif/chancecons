import chancecons as cc
from cvxpy import Variable, Minimize, Maximize
from cvxpy.constraints import NonPos, Inequality
from cvxpy.reductions import InverseData, Reduction, Solution
from cvxpy.error import DCPError

class Order2Chance(Reduction):
	"""Replace order atoms with the appropriate chance constraints.
	"""
	
	def accepts(self, problem):
		return True
	
	def apply(self, problem):
		inverse_data = InverseData(problem)
		is_minimize = type(problem.objective) == Minimize
		order_expr, order_constraints = self.order_tree(problem.objective.args[0], is_minimize, True, False)
		order_objective = Minimize(order_expr) if is_minimize else Maximize(order_expr)
		
		# if any([type(atom) == cc.order for con in problem.constraints for atom in con.atoms()]):
		#	raise DCPError("Order atom may not be nested in constraints.")
		for constraint in problem.constraints:
			if isinstance(constraint, (NonPos, Inequality)):
				order_expr, aux_constr = self.order_tree(constraint.expr, is_minimize, True, False)
				order_constr = order_expr <= 0
				order_constraints += [order_constr] + aux_constr
				inverse_data.cons_id_map.update({constraint.id: order_constr.id})
			else:
				order_constraints += [constraint]

		new_problem = cc.problem.Problem(order_objective, order_constraints)
		return new_problem, inverse_data
	
	def invert(self, solution, inverse_data):
		pvars = {vid: solution.primal_vars[vid] for vid in inverse_data.id_map
				 if vid in solution.primal_vars}
		dvars = {orig_id: solution.dual_vars[vid]
				 for orig_id, vid in inverse_data.cons_id_map.items()
				 if vid in solution.dual_vars}
		return Solution(solution.status, solution.opt_val, pvars, dvars,
						solution.attr)

	def order_tree(self, expr, is_minimize, is_incr, is_decr):
		order_args = []
		constrs = []
		for i, arg in enumerate(expr.args):
			is_obj_incr = (is_incr and expr.is_incr(i)) or (is_decr and expr.is_decr(i))
			is_obj_decr = (is_incr and expr.is_decr(i)) or (is_decr and expr.is_incr(i))
			order_arg, c = self.order_tree(arg, is_minimize, is_obj_incr, is_obj_decr)
			order_args += [order_arg]
			constrs += c
		order_expr, c = self.order_expr(expr, order_args, is_minimize, is_incr, is_decr)
		constrs += c
		return order_expr, constrs

	def order_expr(self, expr, args, is_minimize, is_incr, is_decr):
		if type(expr) == cc.order:
			t = Variable(expr.shape)
			if (is_incr and is_minimize) or (is_decr and not is_minimize):
				constr = expr <= t   # TODO: Do we need expr.copy(args) here?
			elif (is_incr and not is_minimize) or (is_decr and is_minimize):
				constr = expr >= t
			else:
				raise DCPError("Objective must be non-decreasing or non-increasing in each order argument.")
			return t, [constr]
		else:
			return expr.copy(args), []
