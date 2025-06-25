import numpy as np
import matplotlib.pyplot as plt
import time

# ===============================
# Bacterial Colony Optimization (BCO)
# Based on the original 2012 research paper
# Includes: Chemotaxis, Reproduction, Migration, Communication (simplified)
# ===============================

# --------- Benchmark Function: Sphere Function (f(x) = sum(x_i^2)) ---------
def sphere_function(position):
    return sum(x**2 for x in position)

# --------- Adaptive Step Size Function (Eq. 3.6 from paper) ---------
def adaptive_step(iteration, max_iter, C_min, C_max, n):
    return C_min + ((max_iter - iteration) / max_iter) ** n * (C_max - C_min)

# --------- Tumbling + Swimming Movement (Eq. 3.4 from paper) ---------
def tumble_swim(position, C, G_best, P_best, turbulent_strength=0.001):
    f = np.random.rand()
    turbulence = np.random.uniform(-turbulent_strength, turbulent_strength, size=len(position))
    direction = f * (G_best - position) + (1 - f) * (P_best - position) + turbulence
    return position + C * direction

# --------- BCO Parameters ---------
num_dimensions = 30              # Problem dimensionality
lower_bound = -5.12              # Search space lower limit
upper_bound = 5.12               # Search space upper limit
num_bacteria = 50                # Population size
num_iterations = 100             # Max iterations
C_min, C_max = 0.01, 0.1         # Chemotaxis step range
n = 2                            # Chemotaxis step decay control
reproduction_interval = 20       # Reproduction every N iterations
migration_interval = 30          # Migration every N iterations
migration_probability = 0.1      # 10% migration chance

# --------- Initialization ---------
population = np.random.uniform(lower_bound, upper_bound, (num_bacteria, num_dimensions))
fitness = np.array([sphere_function(b) for b in population])
G_best = population[np.argmin(fitness)]  # Global best
best_fitness = []                        # Track best fitness per iteration

# --------- Start Execution Timer ---------
start_time = time.time()

# --------- Main BCO Loop ---------
for iteration in range(num_iterations):
    C = adaptive_step(iteration, num_iterations, C_min, C_max, n)

    for i in range(num_bacteria):
        P_best = population[i]  # Personal best = current position
        new_pos = tumble_swim(population[i], C, G_best, P_best)
        new_pos = np.clip(new_pos, lower_bound, upper_bound)
        new_fit = sphere_function(new_pos)

        # Accept movement if fitness improves
        if new_fit < fitness[i]:
            population[i] = new_pos
            fitness[i] = new_fit

    # Update global best
    G_best = population[np.argmin(fitness)]
    best_fitness.append(sphere_function(G_best))

    # --------- Reproduction (Every N iterations) ---------
    if (iteration + 1) % reproduction_interval == 0:
        sorted_idx = np.argsort(fitness)
        survivors = population[sorted_idx[:num_bacteria // 2]]
        population = np.concatenate([survivors, survivors])
        fitness = np.array([sphere_function(p) for p in population])

    # --------- Migration (Every M iterations) ---------
    if (iteration + 1) % migration_interval == 0:
        for i in range(num_bacteria):
            if np.random.rand() < migration_probability:
                population[i] = np.random.uniform(lower_bound, upper_bound, num_dimensions)
                fitness[i] = sphere_function(population[i])

    print(f"Iteration {iteration + 1} | Best Fitness: {sphere_function(G_best):.6f}")

# --------- End Timer ---------
end_time = time.time()
execution_duration = end_time - start_time
print(f"\n Total Execution Time: {execution_duration:.4f} seconds")

# --------- Plot Fitness Convergence ---------
plt.plot(best_fitness)
plt.title("BCO Convergence on Sphere Function")
plt.xlabel("Iteration")
plt.ylabel("Best Fitness")
plt.grid(True)
plt.show()
