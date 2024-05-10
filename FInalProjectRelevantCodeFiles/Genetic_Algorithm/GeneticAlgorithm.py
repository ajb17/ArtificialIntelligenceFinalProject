"""
Interface for Genetic Algorithm implementations
"""
class GeneticAlgorithm:
	def __init__(self, population, fitnessFunction, elitism=0, mutationRate=0):
		self.population = population
		self.finessFunction = fitnessFunction
		self.mutationRate = mutationRate

	def generateRandomPopulation(self):
		pass

	def select(self):
		pass

	def mutate(self, organism):
		pass

	def reproduce(self, parentA: Organism,  parentB: Organism):
		pass

	def crossover(self, chromeA: Chromosome, chromeB: Chromosome):

		pass



