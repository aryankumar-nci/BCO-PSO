from mealpy import FloatVar
from mealpy.swarm_based.PSO import OriginalPSO
import numpy as np
import matplotlib.pyplot as plt


def sphere(solution):
    return np.sum(np.square(solution))


problem = {
    "obj_func": sphere,
    "bounds": FloatVar(lb=(-10.0,) * 30, ub=(10.0,) * 30),
    "minmax": "min",
    "log_to": None
}


model = OriginalPSO(epoch=100, pop_size=50)
g_best = model.solve(problem)


print(f"Best solution: {g_best.solution}")
print(f"Best fitness: {g_best.target.fitness}")


fitness_over_time = model.history.list_global_best_fit

plt.figure(figsize=(10, 5))
plt.plot(fitness_over_time, label='Global Best Fitness', linewidth=2)
plt.title("PSO Convergence Curve")
plt.xlabel("Epoch")
plt.ylabel("Fitness Value")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
