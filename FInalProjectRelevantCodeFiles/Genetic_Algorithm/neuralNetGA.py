from GeneticAlgorithm import GeneticAlgorithm
import random
from keras.models import Sequential, model_from_json
import os
import tensorflow as tf
# from FitnessFunction import FitnessFunction
# from FitnessFunctions.snakeGymEval import snakeGymEval
import numpy as np
from math import inf
import csv
from Genetic_Algorithm.snake8x8Simulator import snake8x8Simulator
# from time import time
# import time

"""
Genetic Algorithm implementation for Keras neural networks and the Snake game
"""

one_child = 1
two_children = 2
run_none = "show_none"
run_best = "show_best"
run_all = "show_all"


# network_scoring_fn,
class NeuralNetGA:

	def __init__(self, NeuralNetArchitecture: str, feature_extractor, evaluation_function, session_name: str, population_size=100, mutation_rate=0.05, elitism=2, offspring=two_children, overwrite_dir=False, saves=5, show_all=False, show_best=False, track_data=False):
		self.amt_saves = saves
		self.elitism = elitism
		self.generation_number = -1
		self.mutationRate = mutation_rate
		self.saveInd = 0
		self.show_all = show_all
		self.show_best = show_best
		self.amount_children = offspring
		# initialize all weights for initial population, create save files for each
		self.population_size = population_size
		self.simulator = snake8x8Simulator(feature_extractor)
		# Create initial model from NNA
		self.neuralNetwork : Sequential = model_from_json(NeuralNetArchitecture)

		# self.sessionsDir = "Sessions/" + session_name + "/"
		# self.sessionsDir = "ReportData/comparative_architecture/" + session_name + "/"
		# self.sessionsDir = "ReportData/mutation_rates/" + session_name + "/"
		self.sessionsDir = "ReportData/DFSA/" + session_name + "/"
		if not overwrite_dir:
			os.mkdir(self.sessionsDir)

		if track_data:
			self.run_results = {"apples":[], "steps":[], "utils":[]}
			self.track_data= True
			with open(self.sessionsDir+"steps.csv", "w") as csvfile:
				csvwriter = csv.writer(csvfile)
				csvwriter.writerow(["Generation"] + ["steps " + str(i) for i in range(self.population_size)])
			with open(self.sessionsDir+"apples.csv", "w") as csvfile:
				csvwriter = csv.writer(csvfile)
				csvwriter.writerow(["Generation"] + ["apples " + str(i) for i in range(self.population_size)])
			with open(self.sessionsDir+"utility.csv", "w") as csvfile:
				csvwriter = csv.writer(csvfile)
				csvwriter.writerow(["Generation"] + ["utility " + str(i) for i in range(self.population_size)])
			# csvfile.close()
		else:
			self.track_data = False
		# Create data store for NN evaluations
		self.neuralNets = []#{}#[]

		#
		# self.evalFunc : FitnessFunction = evaluation_function
		self.evalFunc = evaluation_function


	def generateRandomPopulation(self):
		self.neuralNets = []#{}#[]
		self.generation_number = 0
		for i in range(self.population_size):
			for layer in self.neuralNetwork.weights:
				layer.assign(tf.random.uniform(layer.shape), -1, 1)
			self.neuralNets.append([self.neuralNetwork.get_weights(), None])

	def select(self):
		# start = time.time()
		scores = np.array([i[1] for i in self.neuralNets])
		normalize_val = scores/np.sum(scores)
		# a, b = np.random.choice([i[0] for i in self.neuralNets], size=2, p=normalize_val, replace=False)
		a, b = np.random.choice(range(len(self.neuralNets)), size=2, p=normalize_val, replace=False)
		# end = time.time()
		# print("select", end-start)
		return self.neuralNets[a][0], self.neuralNets[b][0]
		# return random.choices([i[0] for i in self.neuralNets], weights=[i[1] for i in self.neuralNets], k=2)
		# return random.choices(list(self.neuralNets.keys()),list(self.neuralNets.values()), k=2)#[0]
		# super().select()

	def nextGeneration(self):
		# start = time.time()
		# This selects all parent based on proabalistic randomness. So best cantidate could
		# be eliminated by chance without reproducing. All individuals have random chance of
		# reproducing
		self.generation_number += 1
		nextGen = []
		# numChildren = (self.population_size - self.elitism)
		# extra = self.population_size - (numChildren//2)*2 - self.elitism
		if self.elitism:
			self.neuralNets.sort(key=lambda x: x[1], reverse=True)
			nextGen = [self.neuralNets[i] for i in range(self.elitism)] #TODO should we bother changing score to none again?
			# best_net = max(self.neuralNets, key=lambda x: x[1])[0]

		while len(nextGen) < self.population_size:
		# for i in range(numChildren//2):
			parentX, parentY = self.select()
			# parentY = self.select()

			# self.neuralNetwork.set_weights(self.reproduce(parentX, parentY))
			newWeights1, newWeights2 = self.reproduce(parentX, parentY)
			if self.amount_children == one_child:
				nextGen.append([random.choice([newWeights2, newWeights1]), None])
			elif self.amount_children == two_children:
				nextGen.append([newWeights1, None])
				nextGen.append([newWeights2, None])

		self.neuralNets = nextGen[:self.population_size]
		# end = time.time()
		# print("nextGeneration", end-start)

	def reproduce(self, parentX, parentY):
		# start = time.time()
		weightsX = [i.copy() for i in parentX] #self.neuralNetwork.get_weights()
		weightsY = [i.copy() for i in parentY] # self.neuralNetwork.get_weights()
		#
		iterWX = iter(weightsX)
		iterWY = iter(weightsY)
		for layerX, layerY in zip(iterWX, iterWY):
			biasX = next(iterWX)
			biasY = next(iterWY)

			bcp1 = random.randint(0,len(biasX)-1)
			bcp2 = random.randint(0,len(biasX)-1)
			# print(biasX, biasY)
			biasX[bcp1:bcp2], biasY[bcp1:bcp2] = biasY[bcp1:bcp2], biasX[bcp1:bcp2]
			# print(biasX, biasY)
			biasY = self.mutate_chromosome(biasY)
			biasX = self.mutate_chromosome(biasX)

			for chromeInd in range(0, len(layerX)):
				# ccp1 = random.randint(0, len(chrome)-1)
				# ccp2 = random.randint(0, len(chrome)-1)
				ccp1 = random.randint(0, len(layerX[chromeInd])-1)
				ccp2 = random.randint(0, len(layerX[chromeInd])-1)
				layerX[chromeInd][ccp1:ccp2], layerY[chromeInd][ccp1:ccp2] = layerY[chromeInd][ccp1:ccp2], layerX[chromeInd][ccp1:ccp2]
				layerX[chromeInd] = self.mutate_chromosome(layerX[chromeInd])
				layerY[chromeInd] = self.mutate_chromosome(layerY[chromeInd])
		# end = time.time()
		# print("reproduce", end-start)
		return weightsX, weightsY

	def mutate_chromosome(self, chrome):
		# start = time.time()
		for i in range(len(chrome)):
			chrome[i] = random.choices([chrome[i], tf.random.uniform([1], -1, 1).numpy()[0]], [1-self.mutationRate, self.mutationRate])[0]
		# end = time.time()
		# print("mutate_chromosome", end-start, end=" ")
		return chrome

	# def crossover(self, chromeA: Chromosome, chromeB: Chromosome):
	# 	# start = time.time()
	# 	chromeC = Chromosome([0]*chromeA.getLength())
	# 	for i in range(chromeA.getLength()):
	# 		chromeC.setGene(i, random.choice([chromeA.getGene(i), chromeB.getGene(i)]))
	# 	# end = time.time()
	# 	# print("crossover", end-start)
	# 	return chromeC

	def evaluateNet(self, neuralNetwork):
		# start = time.time()
		results = self.simulator.run_model(neuralNetwork, render=self.show_all)
		util = self.evalFunc(*results)
		self.run_results["apples"].append(results[0])
		self.run_results["steps"].append(results[1])
		self.run_results["utils"].append(util)
		# if self.track_data:
		# 	self.run_results.append(list(results).append(util))
		# end = time.time()
		# print("evaluateNet", end-start, end=" ")
		return util# self.evalFunc(*results)#.evaluate(neuralNetwork, render=self.show_all)

	def runGeneration(self):
		# start = time.time()
		best_score = -inf
		for i in range(self.population_size):
			self.neuralNetwork.set_weights(self.neuralNets[i][0])
			# results = self.evaluateNet(self.neuralNetwork)
			utility = self.evaluateNet(self.neuralNetwork)#results[0]
			self.neuralNets[i][1] = utility
			best_score = max(best_score, utility)
		# end = time.time()
		# print("\nrunGeneration", end-start)
		return best_score

	def get_net_scores(self) -> list:
		return [t[1] for t in self.neuralNets]

	def save_best(self):
		# start = time.time()
		self.neuralNetwork.save_weights(self.sessionsDir + str(self.saveInd))
		self.saveInd = int((self.saveInd + 1) % self.amt_saves)
		# end = time.time()
		# print("save_best", end-start)

	def add_scores_to_csv(self):
		# start = time.time()
		with open(self.sessionsDir + "steps.csv", "a") as csvfile:
			csvwriter = csv.writer(csvfile)
			csvwriter.writerow([self.generation_number] + self.run_results["steps"])  # self.get_net_scores())
		with open(self.sessionsDir + "apples.csv", "a") as csvfile:
			csvwriter = csv.writer(csvfile)
			csvwriter.writerow([self.generation_number] + self.run_results["apples"])  # self.get_net_scores())
		with open(self.sessionsDir + "utility.csv", "a") as csvfile:
			csvwriter = csv.writer(csvfile)
			csvwriter.writerow([self.generation_number] + self.run_results["utils"])  # self.get_net_scores())
		# end = time.time()
		# print("add_scores_to_csv", end-start)

	def run_best(self):
		best_net = max(self.neuralNets, key=lambda x: x[1])[0]
		self.neuralNetwork.set_weights(best_net)
		results = self.simulator.run_model(self.neuralNetwork, render=True)
		util = self.evalFunc(*results)
		print("restuls", results, util)


	def evolve(self, amount_generations: int):


		for i in range(amount_generations):
			# start = time.time()
			self.nextGeneration() if len(self.neuralNets) > 0 else self.generateRandomPopulation()
			print("Running generation", self.generation_number, end=" ")
			best_score = self.runGeneration()
			print(best_score)
			self.save_best()
			if self.track_data:
				self.add_scores_to_csv()
				self.run_results = {"apples": [], "steps": [], "utils": []}
			# end = time.time()
			# print("evolve iteration time",end-start)
			if self.show_best:
				self.run_best()










