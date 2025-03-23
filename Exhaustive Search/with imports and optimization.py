from itertools import permutations

adjacency_list = {
    1: {2: 12, 3: 10, 7: 12},
    2: {1: 12, 3: 8, 4: 12},
    3: {1: 10, 2: 8, 4: 11, 5: 3, 7: 9},
    4: {2: 12, 3: 11, 5: 11, 6: 10},
    5: {3: 3, 4: 11, 6: 6, 7: 7},
    6: {4: 10, 5: 6, 7: 9},
    7: {1: 12, 3: 9, 5: 7, 6: 9}
}

def calculate_distance(path, adjacency_matrix):
    total_distance = 0
    for i in range(len(path) - 1):
        total_distance += adjacency_matrix[path[i]][path[i + 1]]
    total_distance += adjacency_matrix[path[-1]][path[0]]  # Return to start
    return total_distance

def tsp_brute_force(adjacency_matrix):
    n = len(adjacency_matrix)
    cities = list(range(n))
    min_distance = float('inf')
    best_route = None

    for perm in permutations(cities[1:]):  # Fix starting point
        path = (0,) + perm
        distance = calculate_distance(path, adjacency_matrix)
        if distance < min_distance:
            min_distance = distance
            best_route = path

    return best_route, min_distance

best_route, min_distance = tsp_brute_force(adjacency_list)
print(f"Best Route: {best_route} with distance {min_distance}")




