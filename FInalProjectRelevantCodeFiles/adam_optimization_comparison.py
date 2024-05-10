import gym
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

from FeatureExtractors.DangerSenseFoodAngle import DangerSenseFoodAngle

env = gym.make('Snake-8x8-v0')
import csv
import time
import os

env.reset()
# goal_steps= 500
# goal_steps= 100
# goal_steps= 50
# score_requirement = 60
# initial_games  = 10000
# initial_games  = 100000
# initial_games  = 1000



def evalFnApplesSteps(apples, steps):
	score = steps + pow(2, apples)+500*pow(apples, 2.1) -0.25*pow(steps, 1.3)*pow(apples, 1.2)
	return score
def evalFnApplesOnly(apples, steps):
	return apples




def model_data_preparation_min_score(feature_extractor, eval_func, goal_steps, initial_games, score_requirement):
	training_data = []
	accepted_scores = []
	for game_index in range(initial_games):
		score = 0
		game_memory = []
		previous_observation = []
		steps_taken = 0
		apples = 0
		steps_remaining = goal_steps
		observation = env.reset()
		while steps_remaining > 0:
		# for step_index in range(goal_steps):
			steps_remaining-=1
			# previous_observation = feature_extractor.extract(observation, env)
			# env.render()
			action = env.action_space.sample()# random.randrange(0,2)
			observation, reward, done, info = env.step(action)
			if len(previous_observation) > 0:
				game_memory.append([previous_observation, action])
			# previous_observation = observation
			if done:
				break

			previous_observation = feature_extractor.extract(observation, env)
			# score += reward#+step_index
			score += reward
			if reward == 1:
				steps_remaining=goal_steps
				apples += 1
			steps_taken += 1
			# if action == 3:
			# 	score += 500
			# print(score)
			# if done:
			# 	break

		# print("score", score)
		# score = apples
		score = eval_func(apples, steps_taken)
		if score >= score_requirement:
			accepted_scores.append(score)
			for data in game_memory:
				if data[1] == 0:
					output = [1, 0, 0]
				elif data[1] == 1:
					output = [0, 1, 0]
				elif  data[1] == 2:
					output = [0, 0, 1]
				else:
					print("data", data)
					raise Exception("No matching data")
				training_data.append([data[0], output])
		env.reset()

	print("accepted scores", accepted_scores)
	return training_data


def NN_EightDirections_24_12_3():
	model = Sequential()
	model.add(Dense(12, input_dim=24, activation='relu'))
	model.add(Dense(3, activation='softmax'))

	model.compile(loss='mse', optimizer=Adam())
	return model

def NN_NearSighted_DirectionSense_9_6_3():
	model = Sequential()
	model.add(Dense(6, input_dim=9, activation='relu'))
	model.add(Dense(3, activation='softmax'))
	model.compile(loss='mse', optimizer=Adam())
	return model

def NN_DangerSense_FoodAngle_4_8_3():
	model = Sequential()
	model.add(Dense(8, input_dim=4, activation='relu'))
	model.add(Dense(3, activation='softmax'))
	model.compile(loss='mse', optimizer=Adam())
	return model




def train_model(training_data, model):
	num_inputs = model.input_shape[1]
	X = np.array([i[0] for i in training_data]).reshape(-1, num_inputs)
	y = np.array([i[1] for i in training_data]).reshape(-1, 3)
	if len(X) > 0:
		model.fit(X, y, epochs=10, verbose=0)
	return model


def _old_run_model(model, delay=0, steps_to_starve=100, amt_games=1):
	scores = []
	choices = []
	for each_game in range(amt_games):#amt_games
		score = 0
		prev_obs = []
		obs = env.reset()
		steps_remaining = steps_to_starve
		apples = 0
		steps_taken = 0
		while steps_remaining > 0:
		# for step_index in range(100):#goal_steps
			steps_remaining-=1
			env.render()
			time.sleep(delay)
			# if len(prev_obs) == 0:
			# 	action = random.randrange(0, 3)
			# else:
				# action =
			features = featExtract.extract(obs, env)
			action = np.argmax(model.predict(features))

			choices.append(action)
			new_observation, reward, done, info = env.step(action)
			obs = new_observation
			if reward == 1:
				apples += 1
				steps_remaining = steps_to_starve
			score += reward
			if done:
				env.render()
				break
			steps_taken+=1
		env.reset()
		scores.append(score)

	print(scores)
	print('Average Score:', sum(scores)/len(scores))
	print('choice 1: {}   choice 0:{}'.format(choices.count(1)/len(choices), choices.count(0)/len(choices)))
	return apples, steps_taken

def run_model(model, delay=0, steps_to_starve=100):
	score = 0
	obs = env.reset()
	steps_remaining = steps_to_starve
	apples = 0
	steps_taken = 0
	while steps_remaining > 0:
		# for step_index in range(100):#goal_steps
		steps_remaining-=1
		env.render()
		time.sleep(delay)
		# if len(prev_obs) == 0:
		# 	action = random.randrange(0, 3)
		# else:
		# action =
		features = featExtract.extract(obs, env)
		action = np.argmax(model.predict(features))

		new_observation, reward, done, info = env.step(action)
		obs = new_observation
		if reward == 1:
			apples += 1
			steps_remaining = steps_to_starve
		score += reward
		if done:
			env.render()
			break
		steps_taken+=1
	# env.reset()
	return apples, steps_taken


def track_model(model, steps_till_starve, amt_games, generation_number, session_dir):
	apples = []
	steps_taken = []
	# utilities = []
	for i in range(amt_games):
		apps, steps = run_model(model, 0, steps_till_starve)
		apples.append(apps)
		steps_taken.append(steps)
	with open(session_dir + "steps.csv", "a") as csvfile:
		csvwriter = csv.writer(csvfile)
		csvwriter.writerow([generation_number] + steps_taken)  # self.get_net_scores())
	with open(session_dir + "apples.csv", "a") as csvfile:
		csvwriter = csv.writer(csvfile)
		csvwriter.writerow([generation_number] + apples)  # self.get_net_scores())

# results_dir = "Sessions/adam_runs/"
results_dir = "ReportData/Adam_Optimizer/"

def train_and_track(model, feature_extractor, eval_function, session_name, amt_generations, goal_steps, game_per_generation, min_prep_score=1, amt_tracking_games=5, overwrite=False):
	session_dir = results_dir + session_name + "/"
	if not overwrite:
		os.mkdir(session_dir)
	with open(session_dir + "steps.csv", "w") as csvfile:
		csvwriter = csv.writer(csvfile)
		csvwriter.writerow( ["Generation"] + ["steps " + str(i) for i in range(amt_tracking_games)])  # self.get_net_scores())
	with open(session_dir + "apples.csv", "w") as csvfile:
		csvwriter = csv.writer(csvfile)
		csvwriter.writerow(["Generation"] + ["apples " + str(i) for i in range(amt_tracking_games)])
	for i in range(amt_generations):
		training_data = model_data_preparation_min_score(feature_extractor, eval_function, goal_steps=goal_steps, initial_games=game_per_generation, score_requirement=min_prep_score)

		# training_data = model_data_preparation_min_score(featExtract, evalFnApplesOnly, goal_steps=100, initial_games=100, score_requirement=1)
		# training_data = model_data_preparation_min_score(feature_extractor, eval_function, goal_steps=goal_steps, initial_games=game_per_generation, score_requirement=1)

		model = train_model(training_data, model)
		track_model(model, steps_till_starve=goal_steps,amt_games=amt_tracking_games, generation_number=i, session_dir=session_dir)
		print("Running generation", i, end=" ")
		apples, steps = run_model(model)
		print(apples)
		model.save_weights(session_dir + str(i))
	return model


# m = NN_NearSighted_DirectionSense_9_6_3()
# featExtract = nearSightedHotCodedDirectionSense()
# population = 100
# session = "NearSighted-DirSenseHC_NN9-6-3_pop-100_gens-100_minscore-1"


m = NN_DangerSense_FoodAngle_4_8_3()
featExtract = DangerSenseFoodAngle()
session = "DFSA_NN4-8-3_pop-100_gens-100_minscore-2"
population = 100
min_score = 2


# featExtract = eightDirections()
# m = NN_EightDirections_24_12_3()
# population = 100
# session = "8D_NN24-12-3_pop-100_gens-100_minscore-1"

m = train_and_track(m, featExtract, evalFnApplesOnly, session_name=session, amt_generations=100, goal_steps=100, game_per_generation=population, min_prep_score=min_score, amt_tracking_games=5, overwrite=False)
# m = train_and_track(m, featExtract, evalFnApplesOnly, session_name="DFSA_NN-4-8-3-test/", amt_generations=100, goal_steps=100, game_per_generation=100, min_prep_score=1, amt_tracking_games=5, overwrite=False)
# train_and_track(m, featExtract, evalFnApplesOnly, session_name="DFSA_NN-4-8-3-test/", amt_generations=100, goal_steps=100, game_per_generation=5, min_prep_score=1, amt_tracking_games=5)

