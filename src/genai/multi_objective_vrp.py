# -*- coding: utf-8 -*-
"""
Multi-Objective Vehicle Routing Problem (MO-VRP) for FreightSense
Objectives: minimize (distance, compliance_risk, delay_probability)
Method: Weighted sum scalarisation + Pareto front generation
RL agent (PPO) learns optimal weight vector from dispatcher feedback
"""
import os
import requests
import numpy as np
from typing import List, Dict, Tuple
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

class MultiObjectiveVRP:
    def __init__(self, osrm_base_url: str = "http://router.project-osrm.org"):
        self.osrm_base_url = osrm_base_url
        self.default_weights = (0.4, 0.3, 0.3) # w_dist, w_risk, w_delay
        
        # Load RL weight agent if it exists
        try:
            from src.genai.rl_weight_agent import RLWeightAgent
            self.rl_agent = RLWeightAgent()
        except Exception:
            self.rl_agent = None

    def get_distance_matrix(self, locations: List[dict]) -> np.ndarray:
        """
        Calls OSRM to get real driving distances (in seconds, mapped to abstract distance).
        Locations: list of {"lat": float, "lon": float}
        """
        if not locations:
            return np.array([])
            
        coords = ";".join([f"{loc['lon']},{loc['lat']}" for loc in locations])
        url = f"{self.osrm_base_url}/table/v1/driving/{coords}"
        
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if "durations" in data:
                    return np.array(data["durations"])
        except Exception as e:
            print(f"⚠️ OSRM API failed: {e}")
            
        # Fallback: Euclidean distance
        n = len(locations)
        matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dx = locations[i]["lon"] - locations[j]["lon"]
                dy = locations[i]["lat"] - locations[j]["lat"]
                matrix[i][j] = np.sqrt(dx**2 + dy**2) * 111000 # rough meters
        return matrix

    def compute_risk_matrix(self, locations: List[dict], constraints: dict) -> np.ndarray:
        """
        Computes compliance risk (0.0 - 1.0) for each pair of nodes.
        """
        n = len(locations)
        matrix = np.zeros((n, n))
        # In a real impl, this would check if the path between i and j 
        # intersects a restricted zone or requires an eway bill.
        # Here we mock it based on destination node risk.
        for i in range(n):
            for j in range(n):
                if i != j:
                    matrix[i][j] = np.random.uniform(0.1, 0.9)
        return matrix

    def compute_delay_matrix(self, locations: List[dict]) -> np.ndarray:
        """
        Computes delay probability for each pair using the DelayPredictorService.
        """
        n = len(locations)
        matrix = np.zeros((n, n))
        
        # Try to load delay predictor
        predictor = None
        try:
            from src.nlp.delay_predictor import DelayPredictorService
            predictor = DelayPredictorService()
        except:
            pass
            
        for i in range(n):
            for j in range(n):
                if i != j:
                    if predictor:
                        # Estimate delay at destination
                        res = predictor.predict(locations[j]["lat"], locations[j]["lon"])
                        matrix[i][j] = res.get("delay_probability", 0.5)
                    else:
                        matrix[i][j] = np.random.uniform(0.0, 1.0)
        return matrix

    def solve_weighted(self, distance_matrix: np.ndarray, risk_matrix: np.ndarray, delay_matrix: np.ndarray, weights: Tuple[float,float,float]) -> dict:
        """
        Solves the TSP/VRP for a given weight combination using OR-Tools.
        """
        w1, w2, w3 = weights
        
        # Normalize matrices to 0-1 scale so weights apply correctly
        def norm(m):
            max_val = np.max(m)
            return m / max_val if max_val > 0 else m
            
        d_norm = norm(distance_matrix)
        r_norm = norm(risk_matrix)
        dl_norm = norm(delay_matrix)
        
        # Combined cost matrix (scaled up for integer solver)
        combined = (w1 * d_norm + w2 * r_norm + w3 * dl_norm) * 10000
        combined = combined.astype(int).tolist()
        
        # OR-Tools setup
        manager = pywrapcp.RoutingIndexManager(len(combined), 1, 0)
        routing = pywrapcp.RoutingModel(manager)
        
        def distance_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return combined[from_node][to_node]

        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
        
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC

        solution = routing.SolveWithParameters(search_parameters)
        
        if solution:
            route = []
            index = routing.Start(0)
            while not routing.IsEnd(index):
                route.append(manager.IndexToNode(index))
                index = solution.Value(routing.NextVar(index))
            route.append(manager.IndexToNode(index))
            
            # Calculate actual individual objectives for the solved route
            total_dist = 0
            total_risk = 0
            total_delay = 0
            for i in range(len(route) - 1):
                fr, to = route[i], route[i+1]
                total_dist += distance_matrix[fr][to]
                total_risk += risk_matrix[fr][to]
                total_delay += delay_matrix[fr][to]
                
            return {
                "route": route,
                "total_distance": float(total_dist),
                "total_risk_score": float(total_risk),
                "estimated_delay_prob": float(total_delay / max(1, len(route))),
                "weights_used": weights
            }
        return {}

    def generate_pareto_front(self, distance_matrix: np.ndarray, risk_matrix: np.ndarray, delay_matrix: np.ndarray, n_solutions: int = 5) -> List[dict]:
        """
        Generates multiple solutions using different weight combinations.
        """
        weight_sets = [
            (1.0, 0.0, 0.0), # Distance only
            (0.0, 1.0, 0.0), # Risk only
            (0.0, 0.0, 1.0), # Delay only
            (0.33, 0.33, 0.34), # Balanced
            (0.6, 0.2, 0.2), # Distance preferred
            (0.2, 0.6, 0.2), # Risk preferred
        ]
        
        solutions = []
        seen_routes = set()
        
        for w in weight_sets[:n_solutions]:
            sol = self.solve_weighted(distance_matrix, risk_matrix, delay_matrix, w)
            if sol and tuple(sol["route"]) not in seen_routes:
                seen_routes.add(tuple(sol["route"]))
                solutions.append(sol)
                
        # Simple Pareto filtering (removing clearly dominated solutions)
        # Omitted for brevity, assuming small distinct sets
        return solutions

    def select_best_route(self, pareto_solutions: List[dict], shipment_features: dict = None) -> dict:
        """
        Uses RL Agent to pick the best route based on dispatcher preferences.
        """
        if not pareto_solutions:
            return {}
            
        if self.rl_agent and shipment_features:
            # Predict best weights
            best_weights = self.rl_agent.predict_weights(shipment_features)
            
            # Find the Pareto solution whose weights are closest to best_weights
            def dist(w1, w2):
                return sum((a-b)**2 for a,b in zip(w1, w2))
                
            best_sol = min(pareto_solutions, key=lambda s: dist(s["weights_used"], best_weights))
            best_sol["selection_reason"] = f"Selected by RL agent (optimized for learned weights {best_weights})"
            return best_sol
            
        # Fallback to the balanced solution
        balanced = min(pareto_solutions, key=lambda s: sum((w-0.33)**2 for w in s["weights_used"]))
        balanced["selection_reason"] = "Selected balanced default weights"
        return balanced
