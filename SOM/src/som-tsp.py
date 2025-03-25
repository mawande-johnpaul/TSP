import numpy as np
import pandas as pd
import networkx as nx
import random
import math

path = []

# open file for reading
with open('SOM/assets/problem-data/cities.tsp', 'r') as file:
    lines = file.readlines()
    i = None
    for line in lines:
        if line.startswith('DIMENSION'):
            dimension = int(line.split()[-1])  # Ensure dimension is an integer
        if line.startswith('NODE_COORD_SECTION'):
            i = lines.index(line)

        if i is not None:
            cities = pd.read_csv(
                file,
                skiprows=i,
                sep=' ',
                names=['city', 'y', 'x'],
                dtype={'city': str, 'x': np.float64, 'y': np.float64},
                header=None,
                nrows=dimension
            )

# Exclude City 1 (start and end point)
iteration = 0
iterations = 1000
cities_to_train = cities.iloc[1:]  # Exclude the first city (index 0)

# Generate weights for the neurons within the bounding box of the cities
weights = dict()

# Get the min and max coordinates of the cities
x_min, x_max = cities_to_train['x'].min(), cities_to_train['x'].max()
y_min, y_max = cities_to_train['y'].min(), cities_to_train['y'].max()

for index, city in cities_to_train.iterrows():
    # Initialize weights within the bounding box of the cities
    weights[int(city['city'])] = [
        random.uniform(x_min, x_max),
        random.uniform(y_min, y_max)
    ]

# Training loop
while iteration < iterations:
    neuron_data = dict()
    for index, city in cities_to_train.iterrows():
        # Calculate Euclidean distance between city and each neuron's weight
        city_coords = [city['x'], city['y']]
        for neuron, weight in weights.items():
            euclidean_distance = math.sqrt((city_coords[0] - weight[0])**2 + (city_coords[1] - weight[1])**2)
            # Add or update neuron data
            if neuron not in neuron_data or euclidean_distance < neuron_data[neuron].get('d', float('inf')):
                neuron_data[neuron] = {'city': int(city['city']), 'd': euclidean_distance}

    # Ensure neuron_data is not empty before finding the winning neuron
    if neuron_data:
        # Find the winning neuron (minimum distance)
        winning_neuron = min(neuron_data, key=lambda n: neuron_data[n]['d'])
        # Append the corresponding city to the path
        path.append(neuron_data[winning_neuron]['city'])
    else:
        print("Warning: neuron_data is empty. Skipping iteration.")
        iteration += 1
        continue

    # Calculate standard deviation (used for neighborhood function)
    standard_deviation = math.exp(-iteration / (iterations / 2))

    # Update weights for the neighborhood of the winning neuron
    for neuron, weight in weights.items():
        # Calculate neighborhood function (Gaussian)
        distance_to_winner = math.sqrt((weights[winning_neuron][0] - weight[0])**2 +
                                       (weights[winning_neuron][1] - weight[1])**2)
        neighborhood = math.exp(-distance_to_winner**2 / (2 * standard_deviation**2))

        # Update weights if the neuron is in the neighborhood
        weight_change = neighborhood * (city_coords[0] - weight[0]), neighborhood * (city_coords[1] - weight[1])
        weights[neuron][0] += weight_change[0]
        weights[neuron][1] += weight_change[1]

    # Increment iteration
    iteration += 1

# Print results
print(path)

# Use networks to plot path using input map and connect them. Put the point name(city number) besides each point.
graph = nx.Graph()
for i in range(len(path) - 1):
    graph.add_edge(path[i], path[i + 1])

positions = {int(city['city']): (city['x'], city['y']) for _, city in cities.iterrows()}
nx.draw(graph, pos=positions, with_labels=True, node_size=500, font_size=10)