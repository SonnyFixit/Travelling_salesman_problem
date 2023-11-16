import random
import numpy as np

import numpy as np

def read_distance_matrix(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()[1:] 
        distance_matrix = [list(map(int, line.split())) for line in lines]
    
    num_cities = len(distance_matrix) + 1
    full_distance_matrix = np.zeros((num_cities, num_cities), dtype=int)
    
    for i in range(1, num_cities):
        for j in range(i):
            full_distance_matrix[i][j] = distance_matrix[i-1][j]
            full_distance_matrix[j][i] = distance_matrix[i-1][j]
    
    return full_distance_matrix


def initialize_population(population_size, num_cities):
    population = []
    for _ in range(population_size):
        individual = list(range(num_cities))
        random.shuffle(individual)
        population.append(individual)
    return population

def calculate_fitness(individual, distance_matrix):
    fitness = sum(distance_matrix[individual[i-1], individual[i]] for i in range(len(individual)))
    return fitness

def tournament_selection(population, k, distance_matrix):
    selected_parents = []
    for _ in range(len(population)):
        tournament = random.sample(population, k)
        winner = min(tournament, key=lambda ind: calculate_fitness(ind, distance_matrix))
        selected_parents.append(winner)
    return selected_parents

def partially_mapped_crossover(parent1, parent2):

    length = len(parent1)
    child1 = [None] * length
    child2 = [None] * length
    start, end = sorted(random.sample(range(length), 2))


    child1[start:end] = parent1[start:end]
    child2[start:end] = parent2[start:end]


    mapping = {parent1[i]: parent2[i] for i in range(start, end)}


    for i in range(length):
        if child1[i] is None:
            city = parent1[i]
            while city in mapping:
                city = mapping[city]
            child1[i] = city

    for i in range(length):
        if child2[i] is None:
            city = parent2[i]
            while city in mapping:
                city = mapping[city]
            child2[i] = city

    return child1, child2


def inversion_mutation(individual):

    start, end = sorted(random.sample(range(len(individual)), 2))
    mutated_individual = individual[:start] + list(reversed(individual[start:end])) + individual[end:]
    return mutated_individual

def swap_mutation(individual):

    pos1, pos2 = random.sample(range(len(individual)), 2)
    mutated_individual = individual.copy()
    mutated_individual[pos1], mutated_individual[pos2] = mutated_individual[pos2], mutated_individual[pos1]
    return mutated_individual

def genetic_algorithm(distance_matrix, population_size, tournament_size, crossover_prob, inversion_mut_prob, swap_mut_prob, num_generations):
    num_cities = len(distance_matrix)
    population = initialize_population(population_size, num_cities)

    for generation in range(num_generations):

        parents = tournament_selection(population, tournament_size, distance_matrix)


        children = []
        for i in range(0, len(parents), 2):
            if random.random() < crossover_prob:
                child1, child2 = partially_mapped_crossover(parents[i], parents[i+1])
                children.extend([child1, child2])
            else:
                children.extend([parents[i], parents[i+1]])


        for i in range(len(children)):
            if random.random() < inversion_mut_prob:
                children[i] = inversion_mutation(children[i])
            if random.random() < swap_mut_prob:
                children[i] = swap_mutation(children[i])


        population = children

    best_solution = min(population, key=lambda ind: calculate_fitness(ind, distance_matrix))
    best_fitness = calculate_fitness(best_solution, distance_matrix)

    return best_solution, best_fitness

if __name__ == "__main__":

    distance_matrix = read_distance_matrix("test.txt")
    population_size = 10
    tournament_size = 5
    crossover_prob = 0.8
    inversion_mut_prob = 0.1
    swap_mut_prob = 0.1
    num_generations = 10

    best_solution, best_fitness = genetic_algorithm(distance_matrix, population_size, tournament_size,
                                                    crossover_prob, inversion_mut_prob, swap_mut_prob, num_generations)

    print("Najlepsze rozwiązanie:", best_solution)
    print("Najlepsza długość trasy:", best_fitness)
