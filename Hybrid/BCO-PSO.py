import numpy as np
import time
import matplotlib.pyplot as plt

class BCO_PSO_Hybrid:
    def __init__(self, fitness_function, bounds, num_bacteria=30, num_iterations=100,
                 C_min=0.01, C_max=0.1, n=2,
                 reproduction_interval=20, pso_update_interval=30, pso_update_prob=0.1):
        self.fitness_function = fitness_function
        self.bounds = bounds
        self.num_bacteria = num_bacteria
        self.num_iterations = num_iterations
        self.C_min = C_min
        self.C_max = C_max
        self.n = n
        self.reproduction_interval = reproduction_interval
        self.pso_update_interval = pso_update_interval
        self.pso_update_prob = pso_update_prob

    def _adaptive_step(self, iteration):
        return self.C_min + ((self.num_iterations - iteration) / self.num_iterations) ** self.n * (self.C_max - self.C_min)

    def _tumble_swim(self, position, C, G_best, P_best, turbulent_strength=0.001):
        f = np.random.rand()
        turbulence = np.random.uniform(-turbulent_strength, turbulent_strength, size=position.shape)
        direction = f * (G_best - position) + (1 - f) * (P_best - position) + turbulence
        return position + C * direction

    def optimize(self, num_dimensions):
        lower, upper = self.bounds if isinstance(self.bounds, tuple) else (np.array([b[0] for b in self.bounds]), 
                                                                            np.array([b[1] for b in self.bounds]))

        population = np.random.uniform(lower, upper, (self.num_bacteria, num_dimensions))
        velocity = np.zeros_like(population)

        fitness = np.array([self.fitness_function(ind) for ind in population])
        P_best = population.copy()
        P_best_fitness = fitness.copy()
        G_best = population[np.argmin(fitness)]
        G_best_score = np.min(fitness)
        best_curve = []

        start_time = time.time()

        for iteration in range(self.num_iterations):
            C = self._adaptive_step(iteration)

            for i in range(self.num_bacteria):
                new_pos = self._tumble_swim(population[i], C, G_best, P_best[i])
                new_pos = np.clip(new_pos, lower, upper)
                new_fit = self.fitness_function(new_pos)

                if new_fit < fitness[i]:
                    population[i] = new_pos
                    fitness[i] = new_fit

            for i in range(self.num_bacteria):
                if fitness[i] < P_best_fitness[i]:
                    P_best[i] = population[i].copy()
                    P_best_fitness[i] = fitness[i]

            G_best_idx = np.argmin(fitness)
            G_best = population[G_best_idx]
            G_best_score = fitness[G_best_idx]
            best_curve.append(G_best_score)

            if (iteration + 1) % self.reproduction_interval == 0:
                sorted_idx = np.argsort(fitness)
                top_half = population[sorted_idx[:self.num_bacteria // 2]]
                population = np.concatenate([top_half, top_half])
                fitness = np.array([self.fitness_function(p) for p in population])

            if (iteration + 1) % self.pso_update_interval == 0:
                w, c1, c2 = 0.5, 1.5, 1.5
                for i in range(self.num_bacteria):
                    if np.random.rand() < self.pso_update_prob:
                        r1 = np.random.rand(num_dimensions)
                        r2 = np.random.rand(num_dimensions)
                        velocity[i] = (
                            w * velocity[i]
                            + c1 * r1 * (P_best[i] - population[i])
                            + c2 * r2 * (G_best - population[i])
                        )
                        population[i] = np.clip(population[i] + velocity[i], lower, upper)
                        fitness[i] = self.fitness_function(population[i])

            print(f"Iteration {iteration+1}/{self.num_iterations}, Best Fitness = {G_best_score:.6f}")

        end_time = time.time()
        print(f"\nTotal Execution Time: {end_time - start_time:.4f} seconds")

        return G_best, G_best_score, best_curve


# Example benchmark: Sphere function
def sphere(x):
    return np.sum(x ** 2)

# Run optimizer
optimizer = BCO_PSO_Hybrid(
    fitness_function=sphere,
    bounds=(-5.12, 5.12),
    num_bacteria=30,
    num_iterations=100
)

best_solution, best_score, convergence = optimizer.optimize(num_dimensions=30)

# Plot convergence
plt.plot(convergence, color='orange')
plt.xlabel("Iteration")
plt.ylabel("Best Fitness")
plt.title("Convergence Curve - BCO-PSO Hybrid")
plt.grid(True)
plt.show()
