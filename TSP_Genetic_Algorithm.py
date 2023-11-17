import random
""" import cProfile
from line_profiler import LineProfiler """

def load_triangular_matrix(file_path):
    with open(file_path, 'r') as file:
        # Skip the first row
        lines = file.readlines()[1:]
        matrix = [list(map(int, line.split())) for line in lines]
    return matrix

def make_symmetric(matrix):
    n = len(matrix)
    symmetric_matrix = [[0] * n for _ in range(n)]

    for i in range(n):
        for j in range(i + 1):  # Only iterate up to the diagonal (inclusive)
            symmetric_matrix[i][j] = symmetric_matrix[j][i] = matrix[i][j]

    return symmetric_matrix

def create_distance_lookup(distance_matrix):
    num_cities = len(distance_matrix)
    distance_lookup = {}

    for i in range(num_cities):
        for j in range(num_cities):
            distance_lookup[(i, j)] = distance_matrix[i][j]

    return distance_lookup

def total_distance(route, distance_lookup):
    total_dist = 0.0
    num_cities = len(route)

    for i in range(num_cities):
        # Look up the distance in the precalculated table
        total_dist += distance_lookup[(route[i], route[(i + 1) % num_cities])]

    return total_dist

def initialize_population(pop_size, num_cities):
    population = []
    for _ in range(pop_size):
        route = list(range(num_cities))
        random.shuffle(route)
        population.append(route)
    return population


def tournament_selection(population, distances, k, fitness_values):
    selected = random.sample(population, k)
    return min(selected, key=lambda x: fitness_values[population.index(x)])

def pmx_crossover(parent1, parent2):
    size = len(parent1)
    a, b = random.sample(range(size), 2)
    if a > b:
        a, b = b, a

    child = [None] * size
    child_set = set()

    # Copy the selected portion from parent1
    child[a:b+1] = parent1[a:b+1]
    child_set.update(child[a:b+1])

    # Map genes in the selected portion from parent2 to their indices
    gene_to_index = {gene: idx for idx, gene in enumerate(parent2)}

    # Fill the rest of the child using parent2
    for i in range(size):
        if i < a or i > b:
            gene = parent2[i]
            while gene in child_set:
                idx = gene_to_index[gene]
                gene = parent2[(idx + 1) % size]
            child[i] = gene
            child_set.add(gene)

    return child


def inversion_mutation(route):
    a, b = sorted(random.sample(range(len(route)), 2))
    route[a:b+1] = reversed(route[a:b+1])
    return route


def exchange_mutation(route):
    a, b = random.sample(range(len(route)), 2)
    route[a], route[b] = route[b], route[a]
    return route


def generate_population_and_evaluate(population, distance_lookup, tournament_size, fitness_values):
    new_population = []

    for _ in range(len(population) // 2):
        parent1 = tournament_selection(population, distance_lookup, tournament_size, fitness_values)
        parent2 = tournament_selection(population, distance_lookup, tournament_size, fitness_values)

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

def genetic_algorithm(distance_matrix, pop_size, tournament_size, crossover_prob, inversion_prob, exchange_prob, num_generations, distance_lookup):
    population = initialize_population(pop_size, len(distance_matrix))
    
    for generation in range(num_generations):
        fitness_values = [total_distance(individual, distance_lookup) for individual in population]
        new_population = generate_population_and_evaluate(population, distance_lookup, tournament_size, fitness_values)
        population = new_population

    best_route = min(population, key=lambda x: total_distance(x, distance_lookup))
    best_distance = total_distance(best_route, distance_lookup)

    return best_route, best_distance

# Example usage:
file_path = 'berlin52.txt'
symmetric_matrix = make_symmetric(load_triangular_matrix(file_path))

# Create the distance lookup table
distance_lookup = create_distance_lookup(symmetric_matrix)

# Adjusted parameters
pop_size = 1500
tournament_size = 10
crossover_prob = 0.8
inversion_prob = 0.15
exchange_prob = 0.15
num_generations = 250

""" # Profiling with cProfile
cprofiler = cProfile.Profile()
cprofiler.enable() """

best_route, best_distance = genetic_algorithm(symmetric_matrix, pop_size, tournament_size, crossover_prob, inversion_prob, exchange_prob, num_generations, distance_lookup=distance_lookup)

print("\nBest Route:")
print(best_route + [best_route[0]])
print("\nBest Distance:", best_distance)

""" cprofiler.disable()
cprofiler.print_stats(sort='cumulative')

# Profiling with line_profiler
profiler = LineProfiler()

# Add functions to be profiled
profiler.add_function(total_distance)
profiler.add_function(initialize_population)
profiler.add_function(tournament_selection)
profiler.add_function(pmx_crossover)
profiler.add_function(inversion_mutation)
profiler.add_function(create_distance_lookup)
profiler.add_function(exchange_mutation)
profiler.add_function(genetic_algorithm)
profiler.add_function(generate_population_and_evaluate)

# Run the code while profiling
profiler_wrapper = profiler(genetic_algorithm)
best_route, best_distance = genetic_algorithm(symmetric_matrix, pop_size, tournament_size, crossover_prob, inversion_prob, exchange_prob, num_generations, distance_lookup=distance_lookup)

# Print the results
profiler.print_stats() """
