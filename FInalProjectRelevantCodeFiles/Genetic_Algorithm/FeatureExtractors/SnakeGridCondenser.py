import numpy as np


class condenseGridData(object):
	def __init__(self):
		pass

	def condensePixel(self, arr: np.ndarray):
		return arr[0] % 252 + arr[1] % 253 + arr[2] % 254

	def condenseObsPixelGrid(self, arr: np.ndarray):
		return np.apply_along_axis(self.condensePixel, -1, arr.reshape(1, -1, 3)).astype("uint8")

	def extract(self, observation, env=None):
		return self.condenseObsPixelGrid(observation)

# def condesePixelGrids(self, arr: np.ndarray):
# 	# return np.apply_along_axis(condensePixel, -1, arr.reshape(-1,3)).astype("uint8")
# 	return np.apply_along_axis(self, func1d=self.condensePixel, axis=-1, arr=arr)
