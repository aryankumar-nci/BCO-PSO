from mealpy import FloatVar
from mealpy.evolutionary_based import GA
import numpy as np
import matplotlib.pyplot as plt

def objective_func(solution):
    return np.sum(solution**2)

problem = {
    "obj_func": objective_func,
    "bounds": FloatVar(lb=[-100] * 30, ub=[100] * 30),
    "minmax": "min",
    "log_to": "console",
    "save_population": False
}

model = GA.BaseGA(epoch=100, pop_size=50, pc=0.9, pm=0.1)
g_best = model.solve(problem, seed=42)

print("Best Solution:", g_best.solution)
print("Best Fitness:", g_best.target.fitness)

history = model.history

plt.figure()
plt.plot(history.list_global_best_fit)
plt.title("Global Best Fitness over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Fitness")
plt.grid(True)
plt.show()

plt.figure()
plt.plot(history.list_epoch_time)
plt.title("Runtime per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Time (s)")
plt.grid(True)
plt.show()

if hasattr(history, 'list_diversity') and history.list_diversity:
    plt.figure()
    plt.plot(history.list_diversity)
    plt.title("Population Diversity")
    plt.xlabel("Epoch")
    plt.ylabel("Diversity")
    plt.grid(True)
    plt.show()
