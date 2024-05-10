import numpy as np
from gym_snake.envs.constants import Direction4
from FeatureExtractors.nearSighted import nearSighted

class nearSightedHotCodedDirectionSense(nearSighted):
	# def __init__(self):
	# 	self.actions = [Action4.left, Action4.forward, Action4.right]
	def get_direction_info(self, env):
		direction = env.grid.snakes[0]._direction
		if direction == Direction4.south:
			return 1, 0, 0, 0
		elif direction == Direction4.north:
			return 0, 1, 0, 0
		elif direction == Direction4.east:
			return 0, 0, 1, 0
		elif direction == Direction4.west:
			return 0, 0, 0, 1
		print(direction)
		raise Exception("bad direction")


	def extract(self, observation, env):
		next_positions = self.get_next_positions(env)
		features = [self.next_position_saftey_rating(observation, pos) for pos in next_positions]
		apple_dists = self.get_apple_dists(env)
		features.extend(apple_dists)

		features.extend(self.get_direction_info(env))
		# print("features", features)
		return np.array(features).reshape(1,-1).astype("int32")
