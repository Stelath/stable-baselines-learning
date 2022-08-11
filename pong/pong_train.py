from pettingzoo.atari import pong_v3
import supersuit as ss

from stable_baselines3 import PPO
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
def create_env():
    env = pong_v3.parallel_env(num_players=2)
    env = ss.max_observation_v0(env, 2)
    env = ss.sticky_actions_v0(env, repeat_action_probability=0.25)
    env = ss.frame_skip_v0(env, 4)
    env = ss.color_reduction_v0(env, mode='G')      # Use Green Channel in Image for faster training time
    env = ss.resize_v1(env, x_size=84, y_size=84)   # Reduce image size to 84 x 84
    env = ss.frame_stack_v1(env, 4)                 # Frame stack to give the ai a better idea of movement speed
    env = ss.pettingzoo_env_to_vec_env_v1(env)      # Convert to Multi Agent Environment
    env = ss.concat_vec_envs_v1(env, 1, num_cpus=1, base_class='stable_baselines3')
    return env

env = create_env()
env.reset()

eval_env = create_env()
eval_env.reset()

eval_callback = EvalCallback(eval_env, best_model_save_path=model_dir,
                             log_path=logdir, eval_freq=10000,
                             deterministic=True, render=False)

model = PPO('CnnPolicy', env, n_steps=2048, batch_size=64, n_epochs=5, verbose=1, tensorboard_log=logdir)
model.learn(total_timesteps=20000000, tb_log_name=model_name)

env.close()
