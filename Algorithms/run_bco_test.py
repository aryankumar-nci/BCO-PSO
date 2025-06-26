from BCO_main import BacterialColonyOptimization
import numpy as np
import matplotlib.pyplot as plt

# benchmark fitness function- Sphere
def sphere(x):
    return np.sum(x**2)

# parameters
bounds = (-5.12, 5.12)  
num_dimensions = 30

# BCO optimizer
bco = BacterialColonyOptimization(
    fitness_function=sphere,
    bounds=bounds,
    num_bacteria=700,
    num_iterations=100
)

# optimization
best_sol, best_score, curve = bco.optimize(num_dimensions)

# Output
print("\nBest solution:", best_sol)
print("Best fitness score:", best_score)

# Plot convergence curve
plt.plot(curve)
plt.title("BCO Convergence on Sphere Function")
plt.xlabel("Iteration")
plt.ylabel("Best Fitness")
plt.grid(True)
plt.show()
