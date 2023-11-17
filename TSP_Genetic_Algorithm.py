import numpy as np
import random

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

def total_distance(route, distance_matrix):
    total = 0
    for i in range(len(route) - 1):
        total += distance_matrix[route[i]][route[i+1]]
    total += distance_matrix[route[-1]][route[0]]  # Return to the starting city
    return total

def initialize_population(pop_size, num_cities):
    population = []
    for _ in range(pop_size):
        route = list(range(num_cities))
        random.shuffle(route)
        population.append(route)
    return population

def tournament_selection(population, distances, k):
    selected = random.sample(population, k)
    return min(selected, key=lambda x: total_distance(x, distances))

def pmx_crossover(parent1, parent2):
    size = len(parent1)
    a, b = random.sample(range(size), 2)
    if a > b:
        a, b = b, a

    # Copy the selected part from parent1 to the child
    child = parent1[a:b+1]
    child_set = set(child)

    # Fill the remaining elements from parent2
    for i in range(size):
        if i < a or i > b:
            gene = parent2[i]
            while gene in child_set:
                idx = parent2.index(gene)
                gene = parent2[(idx + 1) % size]
            child.append(gene)
            child_set.add(gene)

    return child

def inversion_mutation(route):
    a, b = random.sample(range(len(route)), 2)
    if a > b:
        a, b = b, a
    route[a:b+1] = reversed(route[a:b+1])
    return route

def exchange_mutation(route):
    a, b = random.sample(range(len(route)), 2)
    route[a], route[b] = route[b], route[a]
    return route

def genetic_algorithm(distance_matrix, pop_size, tournament_size, crossover_prob, inversion_prob, exchange_prob, num_generations):
    population = initialize_population(pop_size, len(distance_matrix))
    
    for generation in range(num_generations):
        new_population = []

        for _ in range(pop_size // 2):
            parent1 = tournament_selection(population, distance_matrix, tournament_size)
            parent2 = tournament_selection(population, distance_matrix, tournament_size)

            # Crossover
            if random.random() < crossover_prob:
                child1 = pmx_crossover(parent1, parent2)
                child2 = pmx_crossover(parent2, parent1)
            else:
                child1, child2 = parent1[:], parent2[:]

            # Mutation
            if random.random() < inversion_prob:
                child1 = inversion_mutation(child1)
            if random.random() < inversion_prob:
                child2 = inversion_mutation(child2)
            if random.random() < exchange_prob:
                child1 = exchange_mutation(child1)
            if random.random() < exchange_prob:
                child2 = exchange_mutation(child2)

            new_population.extend([child1, child2])

        # Replace the old population with the new one
        population = new_population

    # Find the best individual in the final population
    best_route = min(population, key=lambda x: total_distance(x, distance_matrix))
    best_distance = total_distance(best_route, distance_matrix)

    return best_route, best_distance

# Example usage
file_path = 'berlin52.txt'
symmetric_matrix_np = np.array(make_symmetric(load_triangular_matrix(file_path)))

# Adjusted parameters
pop_size = 10
tournament_size = 3
crossover_prob = 0.7
inversion_prob = 0.05
exchange_prob = 0.05
num_generations = 10

best_route, best_distance = genetic_algorithm(symmetric_matrix_np, pop_size, tournament_size, crossover_prob, inversion_prob, exchange_prob, num_generations)

print("\nBest Route:")
print(best_route + [best_route[0]])
print("\nBest Distance:", best_distance)


print("\nDistance matrix: ")
print(symmetric_matrix_np)
