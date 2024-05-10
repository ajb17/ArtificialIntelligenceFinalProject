"Use included venv (not venv_GA to execute this file)"
import gym

# import stable_baselines
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
import tensorflow as tf


from stable_baselines.common.schedules import LinearSchedule

# import gym_snake
# env = gym.make('snake-v0')
# env.grid_size = [6,6]
# env.unit_gap = 0
# env.unit_size = 1
# env.snake_size = 2
# import matplotlib
# matplotlib.use("TkAgg")

import gym_snake
snake = gym.make('Snake-8x8-v0')
env = DummyVecEnv([lambda: snake])

policy_kwargs = dict(act_fun=tf.nn.relu, net_arch=[128, 128])
model = PPO2('MlpPolicy',  env, verbose=0, policy_kwargs=policy_kwargs, learning_rate=LinearSchedule( 40000*5, initial_p=0.0005, final_p=0.00005).value)#, tensorboard_log="./mlp")

import time
start = time.time()
model.learn(2*4000*5*100) # = 4000000 = 4E6
end = time.time()
print('training: ', end-start)

obs = env.reset()
print("observation space", env.observation_space)
print("action space", env.action_space)

for i_episode in range(5):
	observation = env.reset()
	utility = 0
	for i in range(1000):
		env.render()
		for i in range(3000000):
			pass
		action, _states = model.predict(obs)
		obs, rewards, dones, info = env.step(action)
		if dones:
			break
			env.reset()

def run(use_model=True, episodes=5, delay=3000000):
	obs = env.reset()
	for i_episode in range(episodes):
		observation = env.reset()
		utility = 0
		for i in range(1000):
			env.render()
			for i in range(delay):
				pass
			if use_model:
				action, _states = model.predict(obs)
			else:
				action = [env.action_space.sample()]
			# action = env.action_space.sample()
			obs, rewards, dones, info = env.step(action)
			# print(observation)
			if dones:
				break
				env.reset()

# env.close()
