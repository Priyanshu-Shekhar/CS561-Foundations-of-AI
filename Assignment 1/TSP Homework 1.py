import random
import math

class City:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def distance(self, other):
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2 + (self.z - other.z)**2)

def read_input(filename):
    with open(filename, 'r') as f:
        n = int(f.readline().strip())
        cities = []
        for _ in range(n):
            x, y, z = map(int, f.readline().strip().split())
            cities.append(City(x, y, z))
    return cities

def create_initial_population(size, cities):
    return [random.sample(cities, len(cities)) for _ in range(size)]

def calculate_fitness(path):
    total_distance = sum(path[i].distance(path[i+1]) for i in range(len(path)-1))
    total_distance += path[-1].distance(path[0])  # Return to start
    return 1 / total_distance

def create_mating_pool(population, rank_list):
    mating_pool = []
    for idx, fitness in rank_list:
        num_copies = max(1, int(fitness * 100))  # Ensure at least one copy
        mating_pool.extend([population[idx]] * num_copies)
    return mating_pool if len(mating_pool) > 1 else population[:2]  # Ensure at least 2 individuals

def crossover(parent1, parent2, start, end):
    child = parent1[start:end+1]
    remaining = [city for city in parent2 if city not in child]
    child = remaining[:start] + child + remaining[start:]
    return child

def mutate(path, mutation_rate):
    if random.random() < mutation_rate:
        i, j = random.sample(range(len(path)), 2)
        path[i], path[j] = path[j], path[i]
    return path

def genetic_algorithm(cities, population_size=100, generations=1000, mutation_rate=0.01):
    population = create_initial_population(population_size, cities)
    best_distance = float('inf')
    best_overall_path = None

    for _ in range(generations):
        fitnesses = [calculate_fitness(path) for path in population]
        rank_list = sorted(enumerate(fitnesses), key=lambda x: x[1], reverse=True)
        
        mating_pool = create_mating_pool(population, rank_list)
        
        new_population = []
        for _ in range(population_size):
            parent1, parent2 = random.sample(mating_pool, 2)
            start, end = sorted(random.sample(range(len(cities)), 2))
            child = crossover(parent1, parent2, start, end)
            child = mutate(child, mutation_rate)
            new_population.append(child)
        
        population = new_population
        
        best_path = max(population, key=calculate_fitness)
        current_distance = 1 / calculate_fitness(best_path)
        if current_distance < best_distance:
            best_distance = current_distance
            best_overall_path = best_path

    return best_overall_path, best_distance

def write_output(filename, path, distance):
    with open(filename, 'w') as f:
        f.write(f"{distance:.3f}\n")
        for city in path:
            f.write(f"{city.x} {city.y} {city.z}\n")
        f.write(f"{path[0].x} {path[0].y} {path[0].z}\n")  # Return to start

def main():
    cities = read_input("input.txt")
    best_path, best_distance = genetic_algorithm(cities)
    write_output("output.txt", best_path, best_distance)

if __name__ == "__main__":
    main()