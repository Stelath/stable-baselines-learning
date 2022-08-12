from pettingzoo.atari import pong_v3
import supersuit as ss

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback

env = pong_v3.env(num_players=2)
env = ss.max_observation_v0(env, 2)
env = ss.sticky_actions_v0(env, repeat_action_probability=0.25)
env = ss.frame_skip_v0(env, 4)
env = ss.color_reduction_v0(env, mode='G')      # Use Green Channel in Image for faster training time
env = ss.resize_v1(env, x_size=84, y_size=84)   # Reduce image size to 84 x 84
env = ss.frame_stack_v1(env, 4)                 # Frame stack to give the ai a better idea of movement speed

model = PPO.load("policy")

env.reset()
for agent in env.agent_iter():
    obs, reward, done, info = env.last()
    act, _ = model.predict(obs, deterministic=True)
    print(act)
    env.step(act)
    env.render()