from Genetic_Algorithm.FitnessFunction import FitnessFunction
import gym
import gym_snake
from keras.models import Sequential
import numpy as np
import math


class snakeGymEval(FitnessFunction):

	def __init__(self, featureExtractor, seed=False):
		self.featureExtractor = featureExtractor
		self.env = gym.make('Snake-8x8-v0')

	def extractFeatures(self, observation, env=None) -> list:
		return self.featureExtractor.extract(observation, env)

	def evaluate(self, model: Sequential, render=False) -> int:
		# self.env.seed(6)
		obs = self.env.reset()

		score = 0
		steps = 0
		apples = 0
		# print("evaluating")
		steps_till_starvation = 100
		while steps_till_starvation > 0:
		# for i in range(100):
			if render:
				self.env.render()
			# print(i,type(obs), type(obs[0]), type(obs[0][0]),type(obs[0][0][0]))
			features = self.extractFeatures(obs, self.env)
			# self.fail()
			action = np.argmax(model.predict(features))
			obs, reward, done, info = self.env.step(action)
			# score += max(0, reward*20)
			steps+=1
			steps_till_starvation -= 1
			if reward == 1:
				steps_till_starvation = 100
				apples+=1

			if done:
				# steps = i
				if render:
					self.env.render()
					# print(apples)
				break
		# ensures no scores are negative or 0

		# score+=steps/4
		# score += 1.1
		score = steps + pow(2, apples)+500*pow(apples, 2.1) -0.25*pow(steps, 1.3)*pow(apples, 1.2)
		# print("score", score)
		return score


