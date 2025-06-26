from PSO_main import OriginalPSO  
import numpy as np
import matplotlib.pyplot as plt
from mealpy import FloatVar

# Define the Sphere function
def sphere(x):
    return np.sum(x**2)

# PSO settings
num_dimensions = 30
bounds = (-5.12, 5.12)
problem = {
    "obj_func": sphere,
    "bounds": FloatVar(lb=(bounds[0],) * num_dimensions, ub=(bounds[1],) * num_dimensions, name="x"),
    "minmax": "min",
}

# Create and run the PSO model
model = OriginalPSO(epoch=100, pop_size=50, c1=2.05, c2=2.05, w=0.4)

# Solve and collect results
best = model.solve(problem)
curve = model.history.list_global_best_fit

# Output
print("\nBest solution:", best.solution)
print("Best fitness score:", best.target.fitness)

# Plot convergence
plt.plot(curve)
plt.title("PSO Convergence on Sphere Function")
plt.xlabel("Iteration")
plt.ylabel("Best Fitness")
plt.grid(True)
plt.show()
