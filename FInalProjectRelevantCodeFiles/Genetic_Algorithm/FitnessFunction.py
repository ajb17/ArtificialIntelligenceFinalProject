import gym
from keras.models import Sequential
"""Interface for defining fitness functions"""
class FitnessFunction:
	def __init__(self, featureExtractor, seed=False):
		self.featureExtractor = featureExtractor
		self.env = gym.make('Snake-8x8-v0')
	# def __init__(self):
	# 	pass
	def extractFeatures(self, observation) -> list:
		return self.featureExtractor.extract(observation)

	def evaluate(self, model: Sequential, render=False) -> int:
		pass
