import numpy as np
import time

class BacterialColonyOptimization:
    def __init__(self, fitness_function, bounds, num_bacteria=300, num_iterations=100,
                 C_min=0.01, C_max=0.1, n=2,
                 reproduction_interval=20, migration_interval=30, migration_prob=0.1):
        self.fitness_function = fitness_function
        self.bounds = bounds
        self.num_bacteria = num_bacteria
        self.num_iterations = num_iterations
        self.C_min = C_min
        self.C_max = C_max
        self.n = n
        self.reproduction_interval = reproduction_interval
        self.migration_interval = migration_interval
        self.migration_prob = migration_prob

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
        fitness = np.array([self.fitness_function(ind) for ind in population])
        G_best = population[np.argmin(fitness)]
        G_best_score = np.min(fitness)
        best_curve = []

        start_time = time.time()

        for iteration in range(self.num_iterations):
            C = self._adaptive_step(iteration)

            for i in range(self.num_bacteria):
                P_best = population[i].copy()
                new_pos = self._tumble_swim(P_best, C, G_best, P_best)
                new_pos = np.clip(new_pos, lower, upper)
                new_fit = self.fitness_function(new_pos)

                if new_fit < fitness[i]:
                    population[i] = new_pos
                    fitness[i] = new_fit

            G_best_idx = np.argmin(fitness)
            G_best = population[G_best_idx]
            G_best_score = fitness[G_best_idx]
            best_curve.append(G_best_score)

            if (iteration + 1) % self.reproduction_interval == 0:
                sorted_idx = np.argsort(fitness)
                top_half = population[sorted_idx[:self.num_bacteria // 2]]
                population = np.concatenate([top_half, top_half])
                fitness = np.array([self.fitness_function(p) for p in population])

            if (iteration + 1) % self.migration_interval == 0:
                for i in range(self.num_bacteria):
                    if np.random.rand() < self.migration_prob:
                        population[i] = np.random.uniform(lower, upper, num_dimensions)
                        fitness[i] = self.fitness_function(population[i])

            print(f"Iteration {iteration+1}/{self.num_iterations}, Best Fitness = {G_best_score:.6f}")

        end_time = time.time()
        print(f"\n Total Execution Time: {end_time - start_time:.4f} seconds")

        return G_best, G_best_score, best_curve
