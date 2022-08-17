from stable_baselines3 import PPO
from pettingzoo.butterfly import pistonball_v6
import supersuit as ss

env = pistonball_v6.env()
env = ss.color_reduction_v0(env, mode="B")
env = ss.resize_v1(env, x_size=84, y_size=84)
env = ss.frame_stack_v1(env, 3)

model_num = input("Enter Model Num: ")
model = PPO.load(f"policy_{model_num}")

env.reset()
for agent in env.agent_iter():
    obs, reward, done, info = env.last()
    act = model.predict(obs, deterministic=True)[0] if not done else None
    env.step(act)
    env.render()