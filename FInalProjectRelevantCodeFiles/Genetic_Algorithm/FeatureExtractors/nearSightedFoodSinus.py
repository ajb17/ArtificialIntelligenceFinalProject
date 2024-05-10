import numpy as np
from FeatureExtractors.nearSighted import nearSighted
import math
class nearSightedFoodSinus(nearSighted):
	# def __init__(self):
	# 	self.actions = [Action4.left, Action4.forward, Action4.right]

	def get_apple(self, env):
		#TODO only  works for single apple enviornment
		apples = env.grid.apples
		# apple = None
		for app in apples:
			apple = app
		return apple

	def get_head_apple_sin(self, env):
		"""
		_________________
		|          F    |
		|         /     |
		|     h / |o    |
		|     /   |     |
		|sssH------>    |
		|               |
		-----------------
		:param observation:
		:param env:
		:return:
		"""
		head = self.get_head(env)
		apple = self.get_apple(env)
		o = head[1] - apple[1]
		a = head[0] = apple[0]
		h = math.sqrt(pow(o, 2) + pow(a,2))
		#TODO: is this right?
		sin_value =  math.sin(o/h)
		return sin_value




	def extract(self, observation, env):
		next_positions = self.get_next_positions(env)
		features = [self.next_position_saftey_rating(observation, pos) for pos in next_positions]

		apple_sin = self.get_head_apple_sin(env)
		features.append(apple_sin)
		# print("features", features)
		return np.array(features).reshape(1,-1).astype("float32")
