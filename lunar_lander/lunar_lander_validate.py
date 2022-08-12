from statistics import mode
import gym
from stable_baselines3 import PPO
import os

# Environment Setup
env = gym.make("LunarLander-v2")
env.reset()

# Load the model
models_dir = "models/PPO_2022-08-03_18-55-27"
model_number = input("Enter Model Number: ")
model_path = f"{models_dir}/{model_number}"

model = PPO.load(model_path, env=env)

episodes = 10

for ep in range(episodes):
    obs = env.reset()
    done = False

    while not done:
        env.render()
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)

env.close()
