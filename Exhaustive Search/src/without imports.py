adjacency_list = {
    1: {2: 12, 3: 10, 7: 12},
    2: {1: 12, 3: 8, 4: 12},
    3: {1: 10, 2: 8, 4: 11, 5: 3, 7: 9},
    4: {2: 12, 3: 11, 5: 11, 6: 10},
    5: {3: 3, 4: 11, 6: 6, 7: 7},
    6: {4: 10, 5: 6, 7: 9},
    7: {1: 12, 3: 9, 5: 7, 6: 9}
}

def generate_permutations(nodes):
    """Generate all permutations of a list of nodes."""
    if len(nodes) == 1:
        return [nodes]
    permutations = []
    for i in range(len(nodes)):
        node = nodes[i]
        remaining_nodes = nodes[:i] + nodes[i+1:]
        for perm in generate_permutations(remaining_nodes):
            permutations.append([node] + perm)
    return permutations

def generate_valid_sequences():
    """
    Generate all valid permutations of nodes 2-7 and prepend + append node 1
    to ensure cycles start and end at 1.
    """
    nodes = ["2", "3", "4", "5", "6", "7"]
    perm_list = ['1' + ''.join(p) + '1' for p in generate_permutations(nodes)]
    return perm_list

sequences = generate_valid_sequences()
results = {}

# Compute distances for each valid route
for sequence in sequences:
    total_distance = 0
    valid = True  # Track validity of the path

    for i in range(len(sequence) - 1):
        current_node = int(sequence[i])
        next_node = int(sequence[i + 1])

        if next_node in adjacency_list[current_node]:
            total_distance += adjacency_list[current_node][next_node]
        else:
            valid = False
            break  # Stop checking if an invalid connection is found

    if valid:
        results[sequence] = total_distance

# Find best and worst routes
best_route = min(results, key=results.get)
worst_route = max(results, key=results.get)

print(f"Best Route: {best_route} with distance {results[best_route]}")
print(f"Worst Route: {worst_route} with distance {results[worst_route]}")
