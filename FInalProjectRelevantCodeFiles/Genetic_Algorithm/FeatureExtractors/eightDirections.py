import numpy as np
from gym_snake.envs.constants import Action4, Direction4


class eightDirections(object):
	def __init__(self):
		self.actions = [Action4.left, Action4.forward, Action4.right]
		self.direction_vectors = []
		# self.directions = [Di]
		# pass

	# def get_direction_info(self, env):
	# 	direction = env.grid.snakes[0]._direction
	# 	if direction == Direction4.south:
	# 		return 1, 0, 0, 0
	# 	elif direction == Direction4.north:
	# 		return 0, 1, 0, 0
	# 	elif direction == Direction4.east:
	# 		return 0, 0, 1, 0
	# 	elif direction == Direction4.west:
	# 		return 0, 0, 0, 1
	# 	print(direction)
	# 	raise Exception("bad direction")

	def get_head(self, env):
		return env.grid.snakes[0]._deque[-1]

	# def get_lines_of_sight(self, observation, env):
	# 	head = self.get_head(env)
	# 	self.get_direction_info()

	# def combine_vecs(self, posA, posB):
	# 	return (posA[0] - posB[0], posA[1] - posB[1])

	def get_apple(self, env):
		#TODO only  works for single apple enviornment
		apples = env.grid.apples
		apple = None
		# head = self.get_head(env)
		for app in apples:
			apple = app
		return apple#(head[0] - apple[0])/env.grid.width, (head[1] - apple[1])/env.grid.height

	def scan_directions(self, observation, env, amtDirs=8):
		head = np.array(self.get_head(env))
		direction = env.grid.snakes[0]._direction
		dirVectors = []
		# for action in self.actions:
		# 	next_head = env.grid.snakes[0].next_head(action)
		# 	dirVec = self.combine_vecs(head, next_head)
		# 	# dirVec = (self.comb (head[0]-next_head[0], head[1]-next_head[1])
		# 	dirVectors.append(dirVec)
		# # if 4 or more dirs, add the behind direction
		# if amtDirs > 3:
		# 	# dirVec = (-1*(head[0]-dirVectors[1]), -1*(head[1]-dirVectors[1]))
		# 	dirVec = self.combine_vecs(head, (-1*dirVectors[1][0], -1*dirVectors[1][1]))
		# 	dirVectors.append(dirVec)

		# if 8 directions
		if amtDirs == 8:
			# dirVec1 = self.combine_vecs(dirVectors[0], dirVectors[1])
			# dirVec2 = self.combine_vecs(dirVectors[0], dirVectors[3])
			# dirVec3 = self.combine_vecs(dirVectors[2], dirVectors[1])
			# dirVec4 = self.combine_vecs(dirVectors[2], dirVectors[3])
			# dirVectors.extend([dirVec1,dirVec2,dirVec3,dirVec4])
			if direction == Direction4.north:
				dirVectors = [(-1,0), (0,-1), (1,0), (0,1), (-1, -1), (1,-1), (-1,1), (1, 1)]# north
			elif direction == Direction4.west:
				dirVectors = [(0,1), (-1,0), (0,-1), (1,0), (-1, 1), (-1,-1), (1,1), (1, -1)]# west
			elif direction == Direction4.south:
				dirVectors = [(1,0), (0,1), (-1,0), (0,-1), (1, 1), (-1,1), (1,-1), (-1, -1)]# south
			elif direction == Direction4.east:
				dirVectors = [(0,-1), (1,0), (0,1), (-1,0), (1, -1), (1,1), (-1,-1), (-1, 1)]# east

		feats = []
		apple = self.get_apple(env)
		for vec in dirVectors:
			seeApple = 0
			body = 0
			# wall = 0
			curr :np.ndarray = head + vec
			count = 1

			while (-1 < curr[0] < len(observation[0])) and (-1 < curr[1] < len(observation[0])):
				# print(curr)
				if (apple == curr).all():
					seeApple = 1
				if (observation[curr[0]][curr[1]] == [0, 255, 0]).all():
					body = 1
				curr = curr + vec
				count += 1
			feats.extend([count, seeApple, body])

		return feats#dirVectors
		# west   north  east    south    NW     NE     SW     SE
		# [(-1,0), (0,-1), (1,0), (0,1), (-1, -1), (1,-1), (-1,1), (1, 1)]
		# left  forward right  back     FL      FR          BL      BR

		# [(-1,0), (0,1), (1,0), (0,-1), (-1, 1), (1,1), (-1,-1), (1, -1)]
		# for vec in dirVec:


	# def get_next_positions(self, env):
	# 	"left, forward, right"
	# 	snake = env.grid.snakes[0]
	# 	left = snake.next_head(self.actions[0])
	# 	forward = snake.next_head(self.actions[1])
	# 	right = snake.next_head(self.actions[2])
	# 	return left, forward, right

	# def condenseObsPixelGrid(self, arr: np.ndarray):
	# 	return np.apply_along_axis(self.condensePixel, -1, arr.reshape(1, -1, 3)).astype("uint8")
	# def next_position_saftey_rating(self, obs, pos):
	# 	# condensedArr = condensedArr[0]
	# 	# print("obs type", type(obs), end = " ")
	# 	# print(" obs shape", obs.shape)
	# 	grid = obs#[0]
	# 	#TODO: check that these ranges work
	#
	# 	# print("succ", "pos", pos, "obs", obs.shape, "grid[pos[0]][pos[1]]", grid[pos[0]][pos[1]])
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
	# 		# print("red")
	# 		return 0
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
	# 	return (head[0] - apple[0])/10, (head[1] - apple[1])/10

	# def vectorize_direction(self, env):
	# 	snake = env.grid.snakes[0]


	def extract(self, observation, env):
		# next_positions = self.get_next_positions(env)
		# features = [self.next_position_saftey_rating(observation, pos) for pos in next_positions]
		# apple_dists = self.get_apple_dists(env)
		# features.extend(apple_dists)
		# print("features", features)
		features = self.scan_directions(observation, env)
		return np.array(features).reshape(1,-1).astype("uint8")#"float32")
