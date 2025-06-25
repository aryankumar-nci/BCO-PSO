from mealpy.system_based import AEO
from mealpy import FloatVar
import numpy as np

def objective_function(solution):
    return np.sum(solution**2)

problem = {
    "obj_func": objective_function,
    "bounds": FloatVar(lb=(-100,) * 3, ub=(10,) * 3),
    "minmax": "min",
    "verbose": True,
}

model = AEO.ModifiedAEO(epoch=100, pop_size=50)
best_solution = model.solve(problem)

print(f"Best solution: {best_solution.solution}, Fitness: {best_solution.target.fitness}")
