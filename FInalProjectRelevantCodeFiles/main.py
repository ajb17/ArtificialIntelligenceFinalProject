"""
Use this file to setup the simulator, model, and parameters to run the Genetic Algorithm
"""
from Genetic_Algorithm.neuralNetGA import *
from FeatureExtractors.DangerSenseFoodAngle import DangerSenseFoodAngle
from keras.models import Sequential
from keras.layers import Dense
from math import inf


m = Sequential()


# Full state 8x8
# m.add(Dense(8, input_dim=64, activation='relu'))
# m.add(Dense(3, activation='softmax'))
# featExtract = condenseGridData()

# left, forward, right clear, foodX dist, foodY dist
# m.add(Dense(10, input_dim=5, activation='relu'))
# m.add(Dense(3, activation='softmax'))
# featExtract = nearSighted()

# left, forward, right clear, foodX dist, foodY dist, vert direction, horozontal direction
# m.add(Dense(14, input_dim=7, activation='relu'))
# m.add(Dense(3, activation='softmax'))
# featExtract = nearSightedDirectionSense()

# left, forward, right clear, foodX dist, foodY dist, south, north, east, west
# m.add(Dense(6, input_dim=9, activation='relu'))
# m.add(Dense(3, activation='softmax'))
# featExtract = nearSightedHotCodedDirectionSense()

# full state
# m.add(Dense(64, input_dim=192, activation='relu'))
# m.add(Dense(3, activation='softmax'))
# featExtract = fullState()

# # eight directions
# m.add(Dense(12, input_dim=24, activation='relu'))
# # m.add(Dense(6, activation='relu'))
# m.add(Dense(3, activation='softmax'))
# featExtract = eightDirections()

# DangerSenseFoodAngle
m.add(Dense(8, input_dim=4, activation='relu'))
m.add(Dense(3, activation='softmax'))
featExtract = DangerSenseFoodAngle()

# evalFn = snakeGymEval(featExtract)

def snake8x8Eval(apples, steps):
	score = steps + pow(2, apples)+500*pow(apples, 2.1) -0.25*pow(steps, 1.3)*pow(apples, 1.2)
	return score
mArch = m.to_json()

# ga = NeuralNetGA(mArch, evalFn, "NearSightDirSenseHotCodedFixed", 100, elitism=10, mutation_rate=.1, overwrite_dir=True)
# ga = NeuralNetGA(mArch, evalFn, "fullStateTest", 100, elitism=10, mutation_rate=.1, overwrite_dir=True)

# filename1 = "eightDirectionsShallowNN_popComp_pop-100_eli-10_mut-1p"
# ga = NeuralNetGA(mArch, evalFn, filename1, 100, elitism=10, mutation_rate=.1, saves=inf, overwrite_dir=True)
# filename2 = "eightDirectionsShallowNN_popComp_pop-100_eli-10_mut-25p"
# ga = NeuralNetGA(mArch, evalFn, filename2, 100, elitism=10, mutation_rate=.25, saves=inf, overwrite_dir=True)
# filename3 = "eightDirectionsShallowNN_popComp_pop-100_eli-10_mut-5p"
# ga = NeuralNetGA(mArch, evalFn, filename3, 100, elitism=10, mutation_rate=.5, saves=inf, overwrite_dir=True)

# ga = NeuralNetGA(mArch, evalFn, "eightDirectionsShallowNN_pop-100_eli-10_mut-1p", 100, elitism=10, mutation_rate=.1, saves=inf, overwrite_dir=True)

# filename = "8D_NN24-12-3_test4"
# ga = NeuralNetGA(mArch, featExtract,snake8x8Eval, filename, population_size=30, elitism=10, offspring=one_child, mutation_rate=.1, saves=inf, overwrite_dir=False, track_data=True, show_all=True, show_best=True)



# filename  = "8D_NN24-12-3_pop-100_eli-10_mut-p01_two-child_std-eval"
# mute_rate = 0.01
# filename  = "8D_NN24-12-3_pop-100_eli-10_mut-p10_two-child_std-eval"
# mute_rate = 0.1
# filename  = "8D_NN24-12-3_pop-100_eli-10_mut-p25_two-child_std-eval"
# mute_rate = 0.25
# filename  = "8D_NN24-12-3_pop-100_eli-10_mut-p50_two-child_std-eval"
# mute_rate = 0.5
# ga = NeuralNetGA(mArch, featExtract,snake8x8Eval, filename, population_size=100, elitism=10, offspring=two_children, mutation_rate=mute_rate, saves=inf, overwrite_dir=False, track_data=True, show_all=False, show_best=False)


# filename = "8D_NN24-12-3_pop-60_eli-6_mut-p05_two-child_std-eval"
# pop_size = 60
# elitism = 6
# filename = "8D_NN24-12-3_pop-90_eli-9_mut-p05_two-child_std-eval"
# pop_size = 90
# elitism = 9


# filename = "DS_NN9-6-3_pop-100_eli-10_mut-p05_two-child_std-eval"
# pop_size = 100
# elitism = 10
# ga = NeuralNetGA(mArch, featExtract,snake8x8Eval, filename, population_size=pop_size, elitism=elitism, offspring=two_children, mutation_rate=.05, saves=inf, overwrite_dir=False, track_data=True, show_all=False, show_best=False)


# filename = "8D_NN_24-12-6-3_pop-100_eli-10_mut-p05_two-child_std-eval"
# ga = NeuralNetGA(mArch, featExtract,snake8x8Eval, filename, population_size=100, elitism=10, offspring=two_children, mutation_rate=.05, saves=inf, overwrite_dir=False, track_data=True, show_all=False, show_best=False)


# filename = "8D_NN_24-12-3_pop-100_eli-0_mut-p05_two-child_std-eval"
# eli = 0
# ga = NeuralNetGA(mArch, featExtract,snake8x8Eval, filename, population_size=100, elitism=eli, offspring=two_children, mutation_rate=.05, saves=inf, overwrite_dir=False, track_data=True, show_all=False, show_best=False)
# filename = "8D_NN_24-12-3_pop-100_eli-1_mut-p05_two-child_std-eval"
# eli = 1
# ga = NeuralNetGA(mArch, featExtract,snake8x8Eval, filename, population_size=100, elitism=eli, offspring=two_children, mutation_rate=.05, saves=inf, overwrite_dir=False, track_data=True, show_all=False, show_best=False)
# filename = "8D_NN_24-12-3_pop-100_eli-10_mut-p05_two-child_std-eval"
# eli = 10
# ga = NeuralNetGA(mArch, featExtract,snake8x8Eval, filename, population_size=100, elitism=eli, offspring=two_children, mutation_rate=.05, saves=inf, overwrite_dir=False, track_data=True, show_all=False, show_best=False)



# filename = "8D_NN24-12-3_pop-100_eli-10_mut-p0_two-child_std-eval"
# ga = NeuralNetGA(mArch, featExtract,snake8x8Eval, filename, population_size=100, elitism=10, offspring=two_children, mutation_rate=0, saves=inf, overwrite_dir=False, track_data=True, show_all=False, show_best=False)

# filename = "timing_diagnostics"
# ga = NeuralNetGA(mArch, featExtract,snake8x8Eval, filename, population_size=100, elitism=10, offspring=two_children, mutation_rate=0, saves=inf, overwrite_dir=True, track_data=True, show_all=False, show_best=False)


# filename = "DFSA_NN4-8-3_pop-100_eli-10_mut-p05_two-child_std-eval"
# ga = NeuralNetGA(mArch, featExtract,snake8x8Eval, filename, population_size=100, elitism=10, offspring=two_children, mutation_rate=.05, saves=inf, overwrite_dir=False, track_data=True, show_all=False, show_best=False)
# TODO
filename = "DFSA_NN4-8-3_pop-30_eli-3_mut-p05_two-child_std-eval"
ga = NeuralNetGA(mArch, featExtract,snake8x8Eval, filename, population_size=30, elitism=3, offspring=two_children, mutation_rate=.05, saves=inf, overwrite_dir=False, track_data=True, show_all=False, show_best=False)

# filename = "8D_NN24-12-3_pop-100_eli-9_mut-p05_two-child_std-eval"
# pop_size = 90
# ga = NeuralNetGA(mArch, featExtract,snake8x8Eval, filename, population_size=100, elitism=10, offspring=two_children, mutation_rate=.05, saves=inf, overwrite_dir=False, track_data=True, show_all=False, show_best=False)

ga.evolve(100)




