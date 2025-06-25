from mealpy import FloatVar
from mealpy.swarm_based import PSO
from opfunu.cec_based.cec2017 import F52017
import matplotlib.pyplot as plt
f1 = F52017(30, f_bias=0) 
# Problem definition
problem = {
    "bounds": FloatVar(lb=f1.lb, ub=f1.ub),
    "obj_func": f1.evaluate,
    "minmax": "min",
    "name": "F5",
    "log_to": "console",
    "save_population": True
}
# PSO algorithm
model = PSO.OriginalPSO(epoch=100, pop_size=30)
g_best = model.solve(problem, seed=10)

print("Best solution:", g_best.solution)
print("Best fitness:", g_best.target.fitness)

# Show charts manually
history = model.history

# Plot trajectory chart (Best fitness over time)
plt.figure()
plt.plot(history.list_global_best_fit)
plt.title("Trajectory of Global Best Fitness")
plt.xlabel("Epoch")
plt.ylabel("Fitness")
plt.grid(True)
plt.show()
# Plot runtime chart
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
