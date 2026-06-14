# -*- coding: utf-8 -*-
import math
import requests
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

def compute_euclidean_distance_matrix(locations):
    """Creates callback to return distance between points."""
    distances = {}
    for from_counter, from_node in enumerate(locations):
        distances[from_counter] = {}
        for to_counter, to_node in enumerate(locations):
            if from_counter == to_counter:
                distances[from_counter][to_counter] = 0
            else:
                # Euclidean distance, converting to integer for OR-tools
                # Note: Multiply by 100000 to maintain precision with lat/lon
                dist = math.hypot(from_node[0] - to_node[0], from_node[1] - to_node[1])
                distances[from_counter][to_counter] = int(dist * 100000)
    return distances

def optimize_route(locations):
    """
    Given a list of [lat, lon] locations, returns the optimized order of indices.
    Index 0 is assumed to be the starting point (depot).
    """
    if len(locations) <= 2:
        return list(range(len(locations)))

    # Instantiate the data problem.
    data = {}
    data['distance_matrix'] = compute_euclidean_distance_matrix(locations)
    data['num_vehicles'] = 1
    data['depot'] = 0

    # Create the routing index manager.
    manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']),
                                           data['num_vehicles'], data['depot'])

    # Create Routing Model.
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['distance_matrix'][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    # Define cost of each arc.
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

    # Solve the problem.
    solution = routing.SolveWithParameters(search_parameters)

    # Return optimized sequence
    if solution:
        index = routing.Start(0)
        route_sequence = []
        while not routing.IsEnd(index):
            route_sequence.append(manager.IndexToNode(index))
            index = solution.Value(routing.NextVar(index))
        return route_sequence
    else:
        # Fallback to original if solving fails
        return list(range(len(locations)))
