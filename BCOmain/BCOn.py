import numpy as np
import matplotlib.pyplot as plt

# Sphere benchmark function
def sphere_function(position):
    return sum(x**2 for x in position)

# Adaptive chemotaxis step
def adaptive_step(iteration, max_iter, C_min, C_max, n):
    return C_min + ((max_iter - iteration) / max_iter) ** n * (C_max - C_min)

# Tumbling and swimming movement
def tumble_swim(position, C, G_best, P_best, turbulent_strength=0.001):
    f = np.random.rand()
    turbulence = np.random.uniform(-turbulent_strength, turbulent_strength, size=len(position))
    direction = f * (G_best - position) + (1 - f) * (P_best - position) + turbulence
    return position + C * direction

# Parameters
num_dimensions = 30
lower_bound = -5.12
upper_bound = 5.12
num_bacteria = 50
num_iterations = 100
C_min, C_max = 0.01, 0.1
n = 2  # Non-linear decay
reproduction_interval = 20
migration_interval = 30

# Initialization
population = np.random.uniform(lower_bound, upper_bound, (num_bacteria, num_dimensions))
fitness = np.array([sphere_function(b) for b in population])
G_best = population[np.argmin(fitness)]

# Track best fitness over time
best_fitness = []

# Main BCO loop
for iteration in range(num_iterations):
    C = adaptive_step(iteration, num_iterations, C_min, C_max, n)

    for i in range(num_bacteria):
        P_best = population[i]
        new_pos = tumble_swim(population[i], C, G_best, P_best)
        new_pos = np.clip(new_pos, lower_bound, upper_bound)
        new_fit = sphere_function(new_pos)

        if new_fit < fitness[i]:
            population[i] = new_pos
            fitness[i] = new_fit

    # Update global best
    G_best = population[np.argmin(fitness)]
    best_fitness.append(sphere_function(G_best))

    # Reproduction
    if (iteration + 1) % reproduction_interval == 0:
        sorted_idx = np.argsort(fitness)
        population = np.concatenate([population[sorted_idx[:num_bacteria // 2]]] * 2)
        fitness = np.array([sphere_function(p) for p in population])

    # Migration
    if (iteration + 1) % migration_interval == 0:
        for i in range(num_bacteria):
            if np.random.rand() < 0.1:  # 10% chance
                population[i] = np.random.uniform(lower_bound, upper_bound, num_dimensions)
                fitness[i] = sphere_function(population[i])

    print(f"Iteration {iteration + 1} | Best Fitness: {sphere_function(G_best):.6f}")

# Plot convergence
plt.plot(best_fitness)
plt.title("BCO Convergence on Sphere Function")
plt.xlabel("Iteration")
plt.ylabel("Best Fitness")
plt.grid(True)
plt.show()
