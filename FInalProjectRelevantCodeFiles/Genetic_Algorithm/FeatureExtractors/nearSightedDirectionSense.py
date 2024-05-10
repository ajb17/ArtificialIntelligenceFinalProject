import numpy as np
from gym_snake.envs.constants import Action4, Direction4
from FeatureExtractors.nearSighted import nearSighted

class nearSightedDirectionSense(nearSighted):
	def __init__(self):
		self.actions = [Action4.left, Action4.forward, Action4.right]

	# def get_head(self, env):
	# 	return env.grid.snakes[0]._deque[0]
	#
	# def get_next_positions(self, env):
	# 	"left, forward, right"
	# 	snake = env.grid.snakes[0]
	# 	left = snake.next_head(self.actions[0])
	# 	forward = snake.next_head(self.actions[1])
	# 	right = snake.next_head(self.actions[2])
	# 	return left, forward, right
	# def next_position_saftey_rating(self, obs, pos):
	# 	grid = obs#[0]
	# 	#TODO: check that these ranges work
	#
	# 	if not (0 <= pos[0] < len(grid)):
	# 		return -1
	# 	if not (0 <= pos[1] < len(grid[0])):
	# 		return -1
	#
	# 	#TODO am i indexing right? or is it the other way ?
	# 	if (grid[pos[0]][pos[1]] == [0,0,0]).all():
	# 		return 0
	# 	if (grid[pos[0]][pos[1]] == [0,255,0]).all():
	# 		return -1
	# 	if (grid[pos[0]][pos[1]] == [255,0,0]).all():
	# 		return 1
	# 	print("failed", "pos", pos, "obs", obs, "grid[pos[0]][pos[1]]", grid[pos[0]][pos[1]])
	# 	# return pos, obs, grid, grid[pos[0]][pos[1]]
	# 	raise Exception("No position safetery rating")
	# def get_apple_dists(self, env):
	# 	#TODO only  works for single apple enviornment
	# 	apples = env.grid.apples
	# 	apple = None
	# 	head = self.get_head(env)
	# 	for app in apples:
	# 		apple = app
	# 	return head[0] - apple[0], head[1] - apple[1]

	# def vectorize_direction(self, env):
	# 	snake = env.grid.snakes[0]
	def get_direction_info(self, env):
		direction = env.grid.snakes[0]._direction
		if direction == Direction4.south:
			return -1, 0
		elif direction == Direction4.north:
			return 1, 0
		elif direction == Direction4.east:
			return 0, 1
		elif direction == Direction4.west:
			return 0, -1
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
