import numpy as np
from typing import List, Tuple, Set, Optional

def euclidean_distance(p1: np.ndarray, p2: np.ndarray) -> float:
    """Calculate Euclidean distance between two points."""
    return np.linalg.norm(p1 - p2)

def total_path_distance(path: List[int], distance_matrix: np.ndarray) -> float:
    """Calculate total distance of a path using precomputed distances."""
    return sum(distance_matrix[path[i]][path[i + 1]] for i in range(len(path) - 1))

def normalize_path(path: List[int]) -> tuple:
    """Normalize the path by choosing the lexicographically smallest representation."""
    n = len(path)
    cyclic_shifts = [tuple(path[i:] + path[:i]) for i in range(n)]
    reversed_path = list(reversed(path))
    cyclic_shifts += [tuple(reversed_path[i:] + reversed_path[:i]) for i in range(n)]
    return min(cyclic_shifts)

def build_path_incremental(current_path: List[int], remaining_points: Set[int], 
                         distance_matrix: np.ndarray) -> Tuple[List[int], List[List[int]]]:
    """Build path incrementally by inserting points to minimize distance."""
    path = list(current_path)
    steps = [path.copy()]

    while remaining_points:
        best_r = None
        best_insertion_position = None
        best_delta_distance = float('inf')

        for r in remaining_points:
            for i in range(len(path) + 1):
                if i == 0:
                    q = path[0]
                    delta = distance_matrix[r][q]
                elif i == len(path):
                    p = path[-1]
                    delta = distance_matrix[p][r]
                else:
                    p = path[i - 1]
                    q = path[i]
                    delta = distance_matrix[p][r] + distance_matrix[r][q] - distance_matrix[p][q]

                if delta < best_delta_distance:
                    best_delta_distance = delta
                    best_r = r
                    best_insertion_position = i

        if best_r is not None:
            path.insert(best_insertion_position, best_r)
            remaining_points.remove(best_r)
            steps.append(path.copy())
        else:
            break

    return path, steps

def build_path_least_distance_updated(start_edge: Tuple[int, int], 
                                    remaining_points: Set[int], 
                                    distance_matrix: np.ndarray) -> Tuple[List[int], List[List[int]]]:
    """Build a Hamiltonian path with lookahead heuristic."""
    path = list(start_edge)
    steps = [path.copy()]

    while remaining_points:
        best_r = None
        best_insertion_position = None
        best_total_distance = float('inf')

        for r in remaining_points:
            for i in range(len(path) + 1):
                temp_path = path.copy()
                temp_path.insert(i, r)
                temp_remaining = remaining_points.copy()
                temp_remaining.remove(r)

                simulated_path, _ = build_path_incremental(temp_path, temp_remaining.copy(), distance_matrix)
                simulated_distance = total_path_distance(simulated_path, distance_matrix)

                if simulated_distance < best_total_distance:
                    best_total_distance = simulated_distance
                    best_r = r
                    best_insertion_position = i

        if best_r is not None:
            path.insert(best_insertion_position, best_r)
            remaining_points.remove(best_r)
            steps.append(path.copy())
        else:
            break

    return path, steps

def process_edge(args: Tuple[Tuple[int, int], int, np.ndarray]) -> Optional[Tuple[float, List[int], List[List[int]]]]:
    """Process a single edge for parallel execution."""
    edge, n, distance_matrix = args
    remaining = set(range(n)) - set(edge)
    path, steps = build_path_least_distance_updated(edge, remaining.copy(), distance_matrix)

    if len(path) == n:
        distance = total_path_distance(path, distance_matrix)
        return (distance, path, steps)
    return None