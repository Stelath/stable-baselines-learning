from pettingzoo.atari import pong_v3
import supersuit as ss

from stable_baselines3 import PPO
from stable_baselines3.ppo import CnnPolicy
from stable_baselines3.common.callbacks import EvalCallback

import os
import datetime

timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
model_name = "PPO_" + timestamp

# Directory Setup
model_dir = "models/" + model_name
logdir = "logs"

if not os.path.exists(model_dir):
    os.makedirs(model_dir)
if not os.path.exists(logdir):
    os.makedirs(logdir)

# Environment Setup
env = pong_v3.parallel_env(num_players=2)
env = ss.max_observation_v0(env, 2)
env = ss.sticky_actions_v0(env, repeat_action_probability=0.25)
env = ss.frame_skip_v0(env, 4)
env = ss.color_reduction_v0(env, mode='B')      # Use Green Channel in Image for faster training time
env = ss.resize_v1(env, x_size=84, y_size=84)   # Reduce image size to 84 x 84
env = ss.frame_stack_v1(env, 4)                 # Frame stack to give the ai a better idea of movement speed
env = ss.pettingzoo_env_to_vec_env_v1(env)      # Convert to Multi Agent Environment
env = ss.concat_vec_envs_v1(env, 2, base_class='stable_baselines3')
env.reset()

# eval_env = deepcopy(env)
# eval_env.reset()

# eval_callback = EvalCallback(eval_env, best_model_save_path=model_dir,
#                              log_path=logdir, eval_freq=10000,
#                              deterministic=True, render=False)

model = PPO(CnnPolicy, env, verbose=3)

i = 1
while True:
    model.learn(total_timesteps=100000, reset_num_timesteps=False)
    model.save(f"{model_dir}/{i*100000}")
    i += 1

env.close()
