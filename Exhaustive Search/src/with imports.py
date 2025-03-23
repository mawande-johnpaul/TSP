from itertools import permutations # Library to carry out permutations

# Data structure to represent the points.
adjacency_list = {
    1: {2: 12, 3: 10, 7: 12},
    2: {1: 12, 3: 8, 4: 12},
    3: {1: 10, 2: 8, 4: 11, 5: 3, 7: 9},
    4: {2: 12, 3: 11, 5: 11, 6: 10},
    5: {3: 3, 4: 11, 6: 6, 7: 7},
    6: {4: 10, 5: 6, 7: 9},
    7: {1: 12, 3: 9, 5: 7, 6: 9}
}

# Function to generate permutations.
def generate_valid_sequences():
    """
    Generate all valid permutations of nodes 2-7 and prepend + append node 1
    to ensure cycles start and end at 1. Return a list of all the permutations.
    """
    nodes = "234567"
    perm_list = ['1' + ''.join(p) + '1' for p in permutations(nodes)]
    return perm_list

sequences = generate_valid_sequences() # Assign the permutations list to sequences variable
results = {} # Dictionary to store all the paths and their respective distances.

# Compute distances for each valid route
for sequence in sequences: # Repeat this block for all sequences
    total_distance = 0 # Initialize the distance covered in the path
    valid = True  # Track validity of the path based on adjacency

    for i in range(len(sequence) - 1): # Repeat this block for all digits in a sequence
        current_node = int(sequence[i]) # The point we are currently on
        next_node = int(sequence[i + 1]) # The next to be visited in the sequence

        # Check if the next point is adjacent to the current point.
        if next_node in adjacency_list[current_node]:
            # Update the distance with distance to next point.
            total_distance += adjacency_list[current_node][next_node]
        else: #
            valid = False # If the next point in the sequence is not adjacent to the current point.
            break  # Stop checking if an invalid connection is found

    if valid: # Adds sequence to valid results if it has passed all validity tests
        results[sequence] = total_distance # The value of the sequence key is its distance travelled

# Find best routes
best_route = min(results, key=results.get) # Min function to return key of minimum value in dictionary

# Print the optimal sequence (route) and its distance
print(f"Best Route: {best_route} with distance {results[best_route]}")

