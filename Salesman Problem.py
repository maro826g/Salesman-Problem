import numpy as np

# Define the cities and their coordinates
cities = {
    0: (0, 0),
    1: (1, 2),
    2: (2, 4),
    3: (3, 1),
    4: (5, 3)
}

# constant Parameters the we will use in the program
population_size = 50
max_generations = 1000
mutation_rate = 0.02


# Function to calculate the distance between two cities
# by the rule the gets the distance between any two points on a plane
def distance(city1, city2):
    x1, y1 = cities[city1]
    x2, y2 = cities[city2]
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

# Function to calculate the total distance of a route
def total_distance(route):
    total = 0
    for i in range(len(route) - 1):
        total += distance(route[i], route[i + 1])
    total += distance(route[-1], route[0])
    return total

# Function to generate an initial population
def generate_population(size):
    population = []
    for _ in range(size):
        route = list(cities.keys())
        np.random.shuffle(route)
        population.append(route)
    return population

# Function to perform crossover between two parent routes
def crossover(parent1, parent2,pc=0.7):
    r = np.random.random()
    if r < pc:
        start = np.random.randint(len(parent1))
        end = np.random.randint(start, len(parent1))
        child = [-1] * len(parent1)
        child[start:end] = parent1[start:end]
        remaining_cities = [city for city in parent2 if city not in child]
        index = 0
        for i in range(len(child)):
            if child[i] == -1:
                child[i] = remaining_cities[index]
                index += 1
    else :
        child=parent1.copy()


    return child

# Function to perform mutation
def mutate(route):
    if np.random.rand() < mutation_rate:
        idx1, idx2 = np.random.choice(len(route), 2, replace=False)
        route[idx1], route[idx2] = route[idx2], route[idx1]
    return route

# Function to select parents based on tournament selection
# it selects 3 populations and chooses the least route in distance
def select_parents(population, tournament_size=3):
    tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
    tournament = [population[i] for i in tournament_indices]
    return min(tournament, key=total_distance)

#function to calc fitness value by inverse the total distance
def calculate_fitness(route):

    return 1 / total_distance(route)

# the sales man function where u call all the previous functions
def sales_man():
    population = generate_population(population_size)

    for generation in range(max_generations):
        new_population = []

        for i in range(population_size // 2):
            parent1 = select_parents(population)
            parent2 = select_parents(population)
            child1 = crossover(parent1, parent2)
            child2 = crossover(parent2, parent1)
            child1 = mutate(child1)
            child2 = mutate(child2)
            new_population.extend([child1, child2])

        population = new_population

    best_route = min(population, key=total_distance)
    best_distance = total_distance(best_route)
    best_fitness = calculate_fitness(best_route)
    return best_route, best_distance,best_fitness


best_route, best_distance, best_fitness = sales_man()
print("Best Route:", best_route)
print("Total Distance:", best_distance)
print("Fitness Value:", best_fitness)