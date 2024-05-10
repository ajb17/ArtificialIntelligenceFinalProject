from keras.models import Sequential
from FeatureExtractors.eightDirections import eightDirections
import gym
import numpy as np
from keras import models


m = Sequential()
# m.add(Dense(10, input_dim=5, activation='relu'))
# m.add(Dense(3, activation='softmax'))
# featExtract = nearSighted()

# m.add(Dense(14, input_dim=7, activation='relu'))
# m.add(Dense(3, activation='softmax'))
# featExtract = nearSightedDirectionSense()


# evalFn = snakeGymEval(featExtract)
# mArch = m.to_json()
# env = evalFn.env


m = models.model_from_json('{"class_name": "Sequential", "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": [null, 24], "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_input"}}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "batch_input_shape": [null, 24], "dtype": "float32", "units": 12, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 3, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "keras_version": "2.4.0", "backend": "tensorflow"}')
featExtract = eightDirections()

env = gym.make('Snake-8x8-v0')
# dir = "NearSightDirSenseWeightRangeFixed/"#"testNewArch/"
# trials = 3
# max_steps = 100
# amtFiles= 5


def run_model(env, model, featExtract, max_steps=100):
	obs = env.reset()
	steps_remaining = max_steps
	score = 1
	while steps_remaining > 0:
		env.render()
		steps_remaining -= 1
		features = featExtract.extract(obs, env)
		# print(features)
		predictions = model.predict(features)
		action = np.argmax(predictions)
		obs, rewards, dones, info = env.step(action)
		if rewards > 0:
			steps_remaining = max_steps
		score += rewards
		if dones:

			# print("     score", score)
			env.render()
			print( score,end="   ")
			break
	return score



def watch_evolution(env, model, featExtract, dir, trials, max_steps, minFile, maxFile):
	while True:
		for i in range(minFile, maxFile):
			file = "Sessions/" +dir + str(i)
			m.load_weights(file)
			print("\n"+file)
			print("     score:  ", end=" ")
			for j in range(trials):
				run_model(env, model, featExtract, max_steps)


# dir1='eightDirectionsShallowNN_pop-25_eli-3_mut-1p/'
# dir2='eightDirectionsShallowNN_pop-50_eli-5_mut-1p/'
dir3='eightDirectionsShallowNN_pop-100_eli-10_mut-1p_comp/'
# watch_evolution(env, m, featExtract, dir3, trials=1, max_steps=100, minFile=90, maxFile=100)



# while True:
# 	for i in range(amtFiles):
# 		file = "Sessions/" +dir + str(i)
# 		m.load_weights(file)
# 		print(file)
# 		for j in range(trials):
# 			obs = env.reset()
# 			steps_remaining = max_steps
# 			while steps_remaining > 0:
# 				env.render()
# 				features = featExtract.extract(obs, env)
# 				action = np.argmax(m.predict(features))
# 				obs, rewards, dones, info = env.step(action)
# 				if rewards > 0:
# 					steps_remaining = max_steps
# 				if dones:
# 					break
# 					env.reset()


