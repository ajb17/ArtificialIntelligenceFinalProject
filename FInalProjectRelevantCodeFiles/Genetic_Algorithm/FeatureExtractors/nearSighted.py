import numpy as np
from gym_snake.envs.constants import Action4

class nearSighted(object):
	def __init__(self):
		self.actions = [Action4.left, Action4.forward, Action4.right]
		# self.directions = [Di]
		# pass

	# def condensePixel(self, arr: np.ndarray):
	# 	return arr[0] % 252 + arr[1] % 253 + arr[2] % 254
	#
	# def condesePixelGrids(self, arr: np.ndarray):
	# 	# return np.apply_along_axis(condensePixel, -1, arr.reshape(-1,3)).astype("uint8")
	# 	return np.apply_along_axis(func1d=self.condensePixel, axis=-1, arr=arr).reshape(8, 8)

	# def findHead(condensedArr):
	# 	rowInd = 0
	# 	for i in range(0, len(condensedArr)):
	# 		colInd = 0
	# 		for j in range(0, len(condensedArr[0])):
	# 			if condensedArr[i][j] == 1:
	# 				return (i, j)

	# def findHeadFromObs(self, condensedArr):
	# 	condensedArr = condensedArr[0]
	# 	rowInd = 0
	# 	for i in range(0, len(condensedArr)):
	# 		colInd = 0
	# 		for j in range(0, len(condensedArr[0])):
	# 			if (condensedArr[i][j] == [0, 0, 255]).all():
	# 				return i, j

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
		# condensedArr = condensedArr[0]
		# print("obs type", type(obs), end = " ")
		# print(" obs shape", obs.shape)
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

	def get_apple_dists(self, env):
		#TODO only  works for single apple enviornment
		apples = env.grid.apples
		apple = None
		head = self.get_head(env)
		for app in apples:
			apple = app
		return (head[0] - apple[0])/env.grid.width, (head[1] - apple[1])/env.grid.height

	# def vectorize_direction(self, env):
	# 	snake = env.grid.snakes[0]


	def extract(self, observation, env):
		next_positions = self.get_next_positions(env)
		features = [self.next_position_saftey_rating(observation, pos) for pos in next_positions]
		apple_dists = self.get_apple_dists(env)
		features.extend(apple_dists)
		# print("features", features)
		return np.array(features).reshape(1,-1).astype("float32")
