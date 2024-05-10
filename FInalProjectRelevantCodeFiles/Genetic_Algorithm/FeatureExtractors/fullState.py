import numpy as np


class fullState(object):
	def __init__(self):
		pass

	def extract(self, observation, env=None):
		features = observation.reshape(1, -1)
		# print(features)
		return features # soelf.condenseObsPixelGrid(observation)

# def condesePixelGrids(self, arr: np.ndarray):
# 	# return np.apply_along_axis(condensePixel, -1, arr.reshape(-1,3)).astype("uint8")
# 	return np.apply_along_axis(self, func1d=self.condensePixel, axis=-1, arr=arr)
