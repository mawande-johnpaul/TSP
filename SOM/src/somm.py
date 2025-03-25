from math import inf, pi, sin, cos, sqrt, exp
import random

# The adjacency list representation of the graph
adjacency_list = {
    1: {2: 12, 3: 10, 7: 12},  # Node 1
    2: {1: 12, 3: 8, 4: 12},   # Node 2
    3: {1: 10, 2: 8, 4: 11, 5: 3, 7: 9},  # Node 3
    4: {2: 12, 3: 11, 5: 11, 6: 10},  # Node 4
    5: {3: 3, 4: 11, 6: 6, 7: 7},  # Node 5
    6: {4: 10, 5: 6, 7: 9},  # Node 6
    7: {1: 12, 3: 9, 5: 7, 6: 9}   # Node 7
}

def convert_to_coordinates(adj_list):
    """
    Convert an adjacency list to 2D coordinates using a simplified approach.
    This creates a valid spatial representation that preserves distances as much as possible.
    """
    n = len(adj_list)
    # Create a simple circular layout
    coords = []
    for i in range(n):
        angle = 2 * pi * i / n
        coords.append([cos(angle), sin(angle)])
    
    # Refine the coordinates using a simple force-directed approach
    for _ in range(100):
        for i in range(n):
            for j in range(i + 1, n):
                # Calculate target distance (normalized)
                if j + 1 not in adj_list[i + 1]:  # Adjust for 1-based index
                    target_dist = 2.0  # Large distance for disconnected cities
                else:
                    target_dist = adj_list[i + 1][j + 1] / 12.0  # Normalize by max distance
                
                # Calculate current distance
                dx = coords[j][0] - coords[i][0]
                dy = coords[j][1] - coords[i][1]
                current_dist = sqrt(dx * dx + dy * dy)
                
                if current_dist > 0:
                    # Calculate force (attraction/repulsion)
                    force = (target_dist - current_dist) / current_dist
                    
                    # Apply force
                    factor = 0.1 * force
                    coords[i][0] -= dx * factor
                    coords[i][1] -= dy * factor
                    coords[j][0] += dx * factor
                    coords[j][1] += dy * factor
    
    return coords

class SOM_TSP:
    def __init__(self, city_coordinates, n_neurons=None, learning_rate=0.5):
        """
        Initialize SOM for TSP
        
        Args:
            city_coordinates: 2D array of city coordinates
            n_neurons: Number of neurons in the ring (default: 2.5 * number of cities)
            learning_rate: Initial learning rate
        """
        self.city_coordinates = city_coordinates
        self.n_cities = len(city_coordinates)
        
        if n_neurons is None:
            self.n_neurons = int(2.5 * self.n_cities)
        else:
            self.n_neurons = n_neurons
        
        # Initialize neurons in a small circle
        self.neuron_coordinates = []
        
        # Calculate center of the cities
        center_x = sum(city[0] for city in city_coordinates) / self.n_cities
        center_y = sum(city[1] for city in city_coordinates) / self.n_cities
        
        # Calculate radius (half the average range)
        max_x = max(city[0] for city in city_coordinates)
        min_x = min(city[0] for city in city_coordinates)
        max_y = max(city[1] for city in city_coordinates)
        min_y = min(city[1] for city in city_coordinates)
        radius = 0.5 * (max(max_x - min_x, max_y - min_y))
        
        # Initialize neurons in a circular layout
        for i in range(self.n_neurons):
            angle = 2 * pi * i / self.n_neurons
            x = center_x + radius * cos(angle)
            y = center_y + radius * sin(angle)
            self.neuron_coordinates.append([x, y])
    
    def train(self, n_iterations=1000000):
        for iteration in range(n_iterations):
            learning_rate = 0.5 * (1 - iteration / n_iterations)
            neighborhood_size = max(1, int(self.n_neurons * (1 - iteration / n_iterations)))
            # ... (rest of training logic)
            
            for city in self.city_coordinates:
                # Find the best matching unit (BMU)
                bmu_index = min(
                    range(self.n_neurons),
                    key=lambda i: sqrt(
                        (self.neuron_coordinates[i][0] - city[0]) ** 2 +
                        (self.neuron_coordinates[i][1] - city[1]) ** 2
                    )
                )
                
                # Update the BMU and its neighbors
                for i in range(self.n_neurons):
                    distance_to_bmu = min(abs(i - bmu_index), self.n_neurons - abs(i - bmu_index))
                    if distance_to_bmu < neighborhood_size:
                        influence = exp(-distance_to_bmu ** 2 / (2 * (neighborhood_size ** 2)))
                        self.neuron_coordinates[i][0] += learning_rate * influence * (city[0] - self.neuron_coordinates[i][0])
                        self.neuron_coordinates[i][1] += learning_rate * influence * (city[1] - self.neuron_coordinates[i][1])
    
    def get_city_sequence(self):
        city_sequence = [1]  # Start with city 1
        visited = {1}  # Track visited cities, starting with city 1
        current_city = 1

        while len(visited) < self.n_cities:
            next_city = None
            min_dist = inf

            # Find the closest unvisited city that is adjacent to the current city
            for neuron in self.neuron_coordinates:
                closest_city = min(
                    range(self.n_cities),
                    key=lambda i: sqrt(
                        (self.city_coordinates[i][0] - neuron[0]) ** 2 +
                        (self.city_coordinates[i][1] - neuron[1]) ** 2
                    )
                )
                candidate = closest_city + 1  # Convert to 1-based index

                # Check if the candidate city is adjacent and unvisited
                if (candidate in adjacency_list[current_city] and
                    candidate not in visited):
                    dist = adjacency_list[current_city][candidate]
                    if dist < min_dist:
                        min_dist = dist
                        next_city = candidate

            if next_city:
                city_sequence.append(next_city)
                visited.add(next_city)
                current_city = next_city
            else:
                break  # No valid path found

        # Ensure the sequence ends at city 1
        if city_sequence[-1] != 1:
            city_sequence.append(1)

        return city_sequence

if __name__ == "__main__":
    # Convert adjacency list to coordinates
    city_coordinates = convert_to_coordinates(adjacency_list)
    
    # Initialize and train SOM
    som = SOM_TSP(city_coordinates)
    som.train(n_iterations=1000)
    
    # Print the resulting sequence of cities
    city_sequence = som.get_city_sequence()
    print("Sequence of cities:", city_sequence)