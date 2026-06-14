# -*- coding: utf-8 -*-
"""
PPO Reinforcement Learning Agent that learns optimal VRP weight vector
from dispatcher feedback (accept/reject decisions on route suggestions)

State: [route_distance_normalised, route_risk_score, delay_probability, 
        cargo_type_encoded, time_pressure_score]
Action: weight vector [w_distance, w_risk, w_delay] (continuous, sums to 1)
Reward: +1 if dispatcher accepts route, -1 if dispatcher rejects/overrides
"""
import os
import numpy as np
import gym
from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

class VRPWeightEnv(gym.Env):
    """
    Custom Gym Environment for learning VRP weights
    """
    def __init__(self):
        super(VRPWeightEnv, self).__init__()
        
        # State: 5 features
        # [dist_norm, risk_score, delay_prob, cargo_type(0-1), time_pressure(0-1)]
        self.observation_space = spaces.Box(low=0, high=1, shape=(5,), dtype=np.float32)
        
        # Action: 3 weights [w_dist, w_risk, w_delay]
        self.action_space = spaces.Box(low=0.01, high=1.0, shape=(3,), dtype=np.float32)
        
        # Mock historical data (in reality, loaded from a CSV of past shipments)
        self.current_step = 0
        self.max_steps = 100
        
    def _get_obs(self):
        return np.random.uniform(0, 1, size=(5,)).astype(np.float32)
        
    def reset(self):
        self.current_step = 0
        return self._get_obs()
        
    def step(self, action):
        self.current_step += 1
        
        # Normalize action to sum to 1 (simplex constraint)
        weights = action / np.sum(action)
        
        # Reward logic: Simulating a dispatcher's preference
        # e.g. If time pressure is high (obs[4] > 0.7), dispatcher prefers w_delay to be high
        obs = self._get_obs()
        time_pressure = obs[4]
        risk_score = obs[1]
        
        reward = 0.0
        if time_pressure > 0.7 and weights[2] > 0.5:
            reward = 1.0 # Accepted
        elif risk_score > 0.7 and weights[1] > 0.5:
            reward = 1.0 # Accepted
        elif weights[0] > 0.6:
            reward = 1.0 # Default fallback
        else:
            reward = -1.0 # Rejected
            
        done = self.current_step >= self.max_steps
        info = {"actual_weights": weights}
        
        return obs, reward, done, info


class RLWeightAgent:
    def __init__(self):
        self.model_dir = os.path.join("models", "rl_weight_agent")
        self.model_path = os.path.join(self.model_dir, "ppo_vrp_weights.zip")
        self.model = None
        self._load_model()
        
    def _load_model(self):
        if os.path.exists(self.model_path):
            self.model = PPO.load(self.model_path)
        else:
            print(f"⚠️ RLWeightAgent model not found at {self.model_path}. Using random initialization.")

    def train(self, feedback_csv: str = None, total_timesteps: int = 10000):
        """Train the PPO agent on simulated or real feedback"""
        os.makedirs(self.model_dir, exist_ok=True)
        print("🚀 Training PPO RL Agent for VRP Weights...")
        
        env = make_vec_env(lambda: VRPWeightEnv(), n_envs=4)
        model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.001)
        model.learn(total_timesteps=total_timesteps)
        
        model.save(self.model_path)
        self.model = model
        print(f"✅ RL Agent saved to {self.model_path}")
        
    def predict_weights(self, shipment_features: dict) -> Tuple[float, float, float]:
        """Get optimal weight vector for a specific shipment"""
        if self.model is None:
            return (0.34, 0.33, 0.33)
            
        # Convert dict to state array
        # [dist_norm, risk_score, delay_prob, cargo_type(0-1), time_pressure(0-1)]
        state = np.array([
            0.5, # mock dist norm
            shipment_features.get("risk_score", 0.5),
            shipment_features.get("delay_prob", 0.5),
            1.0 if "hazmat" in str(shipment_features).lower() else 0.0,
            1.0 if "urgent" in str(shipment_features).lower() else 0.0
        ], dtype=np.float32)
        
        action, _states = self.model.predict(state, deterministic=True)
        
        # Normalize to sum to 1
        weights = action / np.sum(action)
        return (float(weights[0]), float(weights[1]), float(weights[2]))

if __name__ == "__main__":
    agent = RLWeightAgent()
    agent.train(total_timesteps=5000)
