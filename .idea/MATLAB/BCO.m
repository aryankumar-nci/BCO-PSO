function bco()
    % Parameters
    num_bacteria = 30;
    num_iterations = 50;
    num_dimensions = 30;
    lower = -5.12;
    upper = 5.12;
    C_min = 0.01;
    C_max = 0.1;
    n = 2;
    reproduction_interval = 10;
    migration_interval = 20;
    migration_prob = 0.1;

    % Initialization
    population = lower + (upper - lower) * rand(num_bacteria, num_dimensions);
    fitness = arrayfun(@(i) sphere(population(i, :)), 1:num_bacteria)';
    [G_best_score, g_idx] = min(fitness);
    G_best = population(g_idx, :);
    best_curve = zeros(num_iterations, 1);

    for iter = 1:num_iterations
        C = C_min + ((num_iterations - iter) / num_iterations)^n * (C_max - C_min);

        for i = 1:num_bacteria
            P_best = population(i, :);
            f = rand();
            turbulence = -0.001 + 0.002 * rand(1, num_dimensions);
            direction = f * (G_best - P_best) + (1 - f) * (P_best - P_best) + turbulence;
            new_pos = P_best + C * direction;
            new_pos = max(min(new_pos, upper), lower);
            new_fit = sphere(new_pos);

            if new_fit < fitness(i)
                population(i, :) = new_pos;
                fitness(i) = new_fit;
            end
        end

        [G_best_score, g_idx] = min(fitness);
        G_best = population(g_idx, :);
        best_curve(iter) = G_best_score;
        fprintf("Iteration %d: Best Fitness = %.6f\n", iter, G_best_score);

        % Reproduction
        if mod(iter, reproduction_interval) == 0
            [~, idx] = sort(fitness);
            top_half = population(idx(1:num_bacteria/2), :);
            population = [top_half; top_half];
            fitness = arrayfun(@(i) sphere(population(i, :)), 1:num_bacteria)';
        end

        % Migration
        if mod(iter, migration_interval) == 0
            for i = 1:num_bacteria
                if rand() < migration_prob
                    population(i, :) = lower + (upper - lower) * rand(1, num_dimensions);
                    fitness(i) = sphere(population(i, :));
                end
            end
        end
    end

    % Final output
    fprintf("\nBest solution:\n");
    disp(G_best)
    fprintf("Best fitness score: %.6f\n", G_best_score)

    % Plot convergence
    figure;
    plot(1:num_iterations, best_curve, 'b-', 'LineWidth', 2);
    xlabel('Iteration');
    ylabel('Best Fitness');
    title('BCO Convergence Curve');
    grid on;
end

% Objective Function: Sphere
function f = sphere(x)
    f = sum(x .^ 2);
end
