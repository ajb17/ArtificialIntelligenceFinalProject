import numpy as np
from gym_snake.envs.constants import Action4, Direction4
from math import atan2

class DangerSenseFoodAngle(object):
	def __init__(self):
		self.actions = [Action4.left, Action4.forward, Action4.right]
		# self.directions = [Di]
		# pass

	def get_head(self, env):
		return env.grid.snakes[0]._deque[-1]

	def get_next_positions(self, env):
		"left, forward, right"
		snake = env.grid.snakes[0]
		left = snake.next_head(self.actions[0])
		forward = snake.next_head(self.actions[1])
		right = snake.next_head(self.actions[2])
		return left, forward, right
	# def condenseObsPixelGrid(self, arr: np.ndarray):
	# 	return np.apply_along_axis(self.condensePixel, -1, arr.reshape(1, -1, 3)).astype("uint8")
	def next_position_saftey_rating(self, obs, pos):
		grid = obs#[0]
		#TODO: check that these ranges work

		# print("succ", "pos", pos, "obs", obs.shape, "grid[pos[0]][pos[1]]", grid[pos[0]][pos[1]])
		if not (0 <= pos[0] < len(grid)):
			return 1
		if not (0 <= pos[1] < len(grid[0])):
			return 1

		#TODO am i indexing right? or is it the other way ?
		if (grid[pos[0]][pos[1]] == [0,0,0]).all():
			return 0
		if (grid[pos[0]][pos[1]] == [0,255,0]).all():
			return 1
		if (grid[pos[0]][pos[1]] == [255,0,0]).all():
			# print("red")
			return 0
		print("failed", "pos", pos, "obs", obs, "grid[pos[0]][pos[1]]", grid[pos[0]][pos[1]])
		# return pos, obs, grid, grid[pos[0]][pos[1]]
		raise Exception("No position safetery rating")

	def get_apple(self, env):
		#TODO only  works for single apple enviornment
		apples = env.grid.apples
		apple = None
		# head = self.get_head(env)
		for app in apples:
			apple = app
		return apple#(head[0] - apple[0])/env.grid.width, (head[1] - apple[1])/env.grid.height
	# def vectorize_direction(self, env):
	# 	snake = env.grid.snakes[0]

	def get_apple_angle(self, env):
		direction = env.grid.snakes[0]._direction
		head = np.array(self.get_head(env))
		apple = self.get_apple(env)
		literal_dists = head - apple

		if direction == Direction4.north:
			relX = literal_dists[0]
			relY = literal_dists[1]
		elif direction == Direction4.east:
			relX = literal_dists[1]
			relY = -literal_dists[0]
		elif direction == Direction4.south:
			relX = -literal_dists[0]
			relY = -literal_dists[1]
		elif direction == Direction4.west:
			relX = -literal_dists[1]
			relY = literal_dists[0]
		return atan2(relX, relY)


	def extract(self, observation, env):
		next_positions = self.get_next_positions(env)
		features = [self.next_position_saftey_rating(observation, pos) for pos in next_positions]
		apple_angle = self.get_apple_angle(env)
		features.append(apple_angle)
		# print("features", features)
		return np.array(features).reshape(1,-1).astype("float32")
