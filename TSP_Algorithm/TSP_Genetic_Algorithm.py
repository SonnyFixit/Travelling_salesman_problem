import random
import time
import os

from line_profiler import LineProfiler
from itertools import tee, islice, chain

def load_triangular_matrix(file_name):
    # Construct the full file path by including the folder name
    file_path = os.path.join('DistanceMatrices', file_name)
    with open(file_path, 'r') as file:
        # Skip the first line
        lines = file.readlines()[1:]
        matrix = [list(map(int, line.split())) for line in lines]
    return matrix

# Function to create a symmetric matrix from a given matrix
def make_symmetric(matrix):
    n = len(matrix)
    symmetric_matrix = [[0] * n for _ in range(n)]

    # Iterate only up to the diagonal (inclusive)
    for i in range(n):
        for j in range(i + 1):  
            symmetric_matrix[i][j] = symmetric_matrix[j][i] = matrix[i][j]

    return symmetric_matrix

# Function to create a dictionary with distances between cities
def create_distance_lookup(distance_matrix):
    num_cities = len(distance_matrix)
    distance_lookup = {}

    for i in range(num_cities):
        for j in range(i, num_cities): 
            distance_lookup[(i, j)] = distance_lookup[(j, i)] = distance_matrix[i][j]

    return distance_lookup

# Function to calculate the total distance of a route
def total_distance(route, distance_lookup):
    total_dist = 0.0
    num_cities = len(route)

    for i in range(num_cities):
        total_dist += distance_lookup[(route[i], route[(i + 1) % num_cities])]

    return total_dist

# Function to initialize a population with random routes
def initialize_population(pop_size, num_cities):
    population = []
    for _ in range(pop_size):
        route = list(range(num_cities))
        random.shuffle(route)
        population.append(route)
    return population

# Tournament selection function
def tournament_selection(population, distances, k):
    selected = random.sample(population, k)
    return min(selected, key=lambda x: total_distance(x, distances))

# PMX crossover function
def pmx_crossover(parent1, parent2):
    size = len(parent1)
    a, b = random.sample(range(size), 2)
    if a > b:
        a, b = b, a

    child = parent1[a:b+1]
    child_set = set(child)

    for i in range(size):
        if i < a or i > b:
            gene = parent2[i]
            while gene in child_set:
                idx = parent2.index(gene)
                gene = parent2[(idx + 1) % size]
            child.append(gene)
            child_set.add(gene)

    return child

# Inversion mutation function
def inversion_mutation(route):
    a, b = random.sample(range(len(route)), 2)
    if a > b:
        a, b = b, a
    route[a:b+1] = reversed(route[a:b+1])
    return route

# Exchange mutation function
def exchange_mutation(route):
    a, b = random.sample(range(len(route)), 2)
    route[a], route[b] = route[b], route[a]
    return route

# Function to generate a new population and evaluate their routes
def generate_population_and_evaluate(population, distance_lookup, tournament_size):
    new_population = []

    for _ in range(len(population) // 2):
        parent1 = tournament_selection(population, distance_lookup, tournament_size)
        parent2 = tournament_selection(population, distance_lookup, tournament_size)

        if random.random() < crossover_prob:
            child1 = pmx_crossover(parent1, parent2)
            child2 = pmx_crossover(parent2, parent1)
        else:
            child1, child2 = parent1[:], parent2[:]

        if random.random() < inversion_prob:
            child1 = inversion_mutation(child1)
        if random.random() < inversion_prob:
            child2 = inversion_mutation(child2)
        if random.random() < exchange_prob:
            child1 = exchange_mutation(child1)
        if random.random() < exchange_prob:
            child2 = exchange_mutation(child2)

        fitness_child1 = total_distance(child1, distance_lookup)
        fitness_child2 = total_distance(child2, distance_lookup)

        if fitness_child1 < fitness_child2:
            new_population.append(child1)
            new_population.append(parent2)
        else:
            new_population.append(child2)
            new_population.append(parent1)

    return new_population

# Genetic algorithm with improved calculation
def genetic_algorithm_with_elitism(distance_matrix, pop_size, tournament_size, crossover_prob, inversion_prob, exchange_prob, num_generations, elitism_ratio, distance_lookup):
    population = initialize_population(pop_size, len(distance_matrix))
    elitism_count = int(elitism_ratio * pop_size)

    for generation in range(num_generations):
        new_population = generate_population_and_evaluate(population, distance_lookup, tournament_size)
        new_population.sort(key=lambda x: total_distance(x, distance_lookup))
        
        # Preserve the best individuals from the current population
        elite_individuals = new_population[:elitism_count]
        
        # Generate the rest of the population through genetic operations
        non_elite_population = new_population[elitism_count:]
        offspring_population = generate_population_and_evaluate(non_elite_population, distance_lookup, tournament_size)
        
        # Combine elite and offspring populations to form the next generation
        population = elite_individuals + offspring_population

    best_route = min(population, key=lambda x: total_distance(x, distance_lookup))
    best_distance = total_distance(best_route, distance_lookup)

    return best_route, best_distance

# Parameters
file_name = 'berlin52.txt'
symmetric_matrix = make_symmetric(load_triangular_matrix(file_name))
distance_lookup = create_distance_lookup(symmetric_matrix)

pop_size = 100
tournament_size = 3
crossover_prob = 0.85
inversion_prob = 0.15
exchange_prob = 0.15
num_generations = 100000
elitism_ratio = 0.05

# Measure execution time of the algorithm
start_time = time.time()
best_route, best_distance = genetic_algorithm_with_elitism(symmetric_matrix, pop_size, tournament_size, crossover_prob, inversion_prob, exchange_prob, num_generations, elitism_ratio, distance_lookup=distance_lookup)
end_time = time.time()

# Display the results
print("\nBest Route:")
print(best_route + [best_route[0]])
print("\nBest Distance:", best_distance)
print("\nExecution Time:", end_time - start_time, "seconds")


""" cprofiler.disable()
cprofiler.print_stats(sort='cumulative')

# Profiling with line_profiler
profiler = LineProfiler()

# Adding functions to profile
profiler.add_function(total_distance)
profiler.add_function(initialize_population)
profiler.add_function(tournament_selection)
profiler.add_function(pmx_crossover)
profiler.add_function(inversion_mutation)
profiler.add_function(create_distance_lookup)
profiler.add_function(exchange_mutation)
profiler.add_function(genetic_algorithm)
profiler.add_function(generate_population_and_evaluate)

# Start profiling algorithm
profiler_wrapper = profiler(genetic_algorithm)
best_route, best_distance = genetic_algorithm(symmetric_matrix, pop_size, tournament_size, crossover_prob, inversion_prob, exchange_prob, num_generations, distance_lookup=distance_lookup)

# Printing profiling results
profiler.print_stats() """