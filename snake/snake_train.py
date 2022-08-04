import gym
from stable_baselines3 import PPO
import os
import datetime
from snake_env import SnakeEnv

timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
model_name = "PPO_" + timestamp

# Directory Setup
models_dir = "models/" + model_name
logdir = "logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)
if not os.path.exists(logdir):
    os.makedirs(logdir)

# Environment Setup
env = SnakeEnv()
env.reset()

# Model Setup
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=logdir)

TIMESTEPS = 10000
i = 0
while True:
    i += 1
    model.learn(total_timesteps=10000,
                reset_num_timesteps=False, tb_log_name=model_name)
    model.save(f"{models_dir}/{TIMESTEPS*i}")

env.close()
