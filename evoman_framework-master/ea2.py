import sys
import numpy as np
import os
import random
import pandas as pd
from evoman.environment import Environment
from demo_controller import player_controller
import time

os.environ["SDL_VIDEODRIVER"] = "dummy"

# Initialize the environment with a single enemy
n_hidden_neurons = 10
enemies = [3]  #Enemy number
experiment_name = f'custom_evolution_{enemies}'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

env = Environment(experiment_name=experiment_name,
                  enemies=enemies,
                  playermode="ai",
                  player_controller=player_controller(n_hidden_neurons),
                  enemymode="static",
                  level=2,
                  speed="fastest",
                  visuals=False)

# Initialize parameters
pop_size = 100
n_gens = 30
n_runs = 10
n_weights = (env.get_num_sensors() + 1) * n_hidden_neurons + (n_hidden_neurons + 1) * 5

# Get fitness
def fitness_function(fit, p_energy, e_energy, time):
    return fit

# Calculate gain
def calculate_gain(p_energy, e_energy):
    return p_energy - e_energy

# Simulation function
def simulate(env, x):
    fit, p_energy, e_energy, time = env.play(pcont=x)
    gain = calculate_gain(p_energy, e_energy)  
    return fitness_function(fit, p_energy, e_energy, time), gain

# Initialize population
def initialize_population(pop_size, n_weights):
    return np.random.uniform(-1, 1, (pop_size, n_weights))

# Evaluate population fitness and gain
def evaluate_population(population):
    fitness_scores = []
    gain_scores = []
    for x in population:
        fitness, gain = simulate(env, x)
        fitness_scores.append(fitness)
        gain_scores.append(gain)
    return np.array(fitness_scores), np.array(gain_scores)

# Mutation and crossover with adaptive F and CR
def mutation_and_crossover(population, target_idx, gen, n_gens):
    # Adaptive F and CR based on the current gen
    F = 0.4 + 0.5 * (gen / n_gens)  
    CR = 0.9 - 0.6 * (gen / n_gens)  
    
    # Choose three random indices that are different from the target index
    indices = list(range(pop_size))
    indices.remove(target_idx)
    a, b, c = random.sample(indices, 3)
    
    # Create mutant vector
    mutant = population[a] + F * (population[b] - population[c])
    mutant = np.clip(mutant, -1, 1)  
    
    # Create trial vector (crossover)
    trial = np.copy(population[target_idx])
    for i in range(n_weights):
        if random.random() < CR or i == random.randint(0, n_weights - 1):  
            trial[i] = mutant[i]
    
    return trial

# Main DE loop
def run_differential_evolution():
    indices_run = []
    indices_gen = []
    best_gain = []
    best_fit = []
    mean_fitness = []
    std_fitness = []
    best_solutions = []
    result_matrix_max = np.zeros((n_runs, n_gens))
    result_matrix_mean = np.zeros((n_runs, n_gens))

    for r in range(n_runs):
        population = initialize_population(pop_size, n_weights)
        fitness_scores, gain_scores = evaluate_population(population)
        
        for gen in range(n_gens):
            new_population = np.copy(population)
            
            # Perform DE for each individual
            for i in range(pop_size):
                trial = mutation_and_crossover(population, i, gen, n_gens)
                trial_fitness, trial_gain = simulate(env, trial)
                
                # Selection
                if trial_fitness > fitness_scores[i]:
                    new_population[i] = trial
                    fitness_scores[i] = trial_fitness
                    gain_scores[i] = trial_gain  

            population = new_population

            # Log the results
            best_idx = np.argmax(fitness_scores)
            best_solution = population[best_idx].tolist()
            best_fit_val = fitness_scores[best_idx]
            best_gain_val = gain_scores[best_idx]
            mean_val = np.mean(fitness_scores)
            std_val = np.std(fitness_scores)

            print(f'RUN {r}, GENERATION {gen} - Best Fitness: {round(best_fit_val, 6)}, Gain: {round(best_gain_val, 6)}, Mean: {round(mean_val, 6)}, Std: {round(std_val, 6)}')

            indices_run.append(r)
            indices_gen.append(gen)
            best_gain.append(best_gain_val)
            best_fit.append(best_fit_val)
            mean_fitness.append(mean_val)
            std_fitness.append(std_val)
            best_solutions.append(best_solution)

            result_matrix_max[r, gen] = best_fit_val
            result_matrix_mean[r, gen] = mean_val
    
    # Store the results in a Dataframe and CSV
    results_df = pd.DataFrame({
        'Run': indices_run,
        'Generation': indices_gen,
        'Best Gain': best_gain,
        'Best Fitness': best_fit,
        'Mean Fitness': mean_fitness,
        'Std Fitness': std_fitness,
        'Best Solution': best_solutions
    })
    results_df.to_csv(f'{experiment_name}/results.csv', index=False)

    print("Differential Evolution completed. Results saved.")

if __name__ == "__main__":
    run_differential_evolution()
