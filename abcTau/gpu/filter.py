import numpy as np

def filter_by_epsilon(theta, distances, epsilon):
    mask = distances < epsilon
    return theta[:, mask], distances[mask]
