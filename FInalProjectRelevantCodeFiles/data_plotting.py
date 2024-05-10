import matplotlib as mpl
import matplotlib.pyplot as plt
# from matplotlib.pyplot import Fun
from matplotlib.animation import FuncAnimation
import numpy as np
from pylab import cm
from scipy import optimize
from scipy.interpolate import UnivariateSpline, CubicSpline

# filename = "ReportData/mutation_rates/8D_NN24-12-3_pop-100_eli-10_mut-p10_two-child_std-eval/apples.csv"
# filename = "Sessions/adam_runs/DFSA_NN-4-8-3-test/apples.csv"
# filename = "ReportData/DFSA/DFSA_NN4-8-3_pop-100_eli-10_mut-p05_two-child_std-eval/apples.csv"

# filename = "ReportData/DFSA/DFSA_NN4-8-3_pop-30_eli-3_mut-p05_two-child_std-eval/apples.csv"
# gens, *vals = np.loadtxt(filename, unpack=True, delimiter=",",skiprows=1)

# gens, *v = np.loadtxt(filename, unpack=True, delimiter=",",skiprows=1)
# vals = np.loadtxt(filename, unpack=False, delimiter=",",skiprows=1, usecols=range(1, len(v)+1))
# gens, *vals = np.loadtxt(filename, unpack=True, delimiter=",",skiprows=1)


# for i in range(len(vals)):
# 	print(i)
# 	plt.scatter(gens, vals[i])

def plot_points(session, score_type, plot_type, spline=False):
	if score_type == "apples" or score_type == "a":
		filename = session + "/" + "apples.csv"
		plt.ylabel("Apples")
	elif score_type == "utilities" or score_type == "u":
		filename = session + "/" + "utility.csv"
		plt.ylabel("Fitness Score")
	elif score_type == "steps" or score_type == "s":
		filename = session + "/" + "steps.csv"
		plt.ylabel("Steps")
	# filename = "ReportData/DFSA/DFSA_NN4-8-3_pop-30_eli-3_mut-p05_two-child_std-eval/apples.csv"
	if plot_type == "all":
		gens, *vals = np.loadtxt(filename, unpack=True, delimiter=",", skiprows=1)
		for i in range(len(vals)):
			# print(i)
			plt.scatter(gens, vals[i])
		if spline:
			print("no spline available for all points ption")
		# 	s = UnivariateSpline(gens, vals, s=10000)
		# 	xs = gens
		# 	ys = s(xs)
		# 	plt.plot(xs, ys)
	else:
		gens, *v = np.loadtxt(filename, unpack=True, delimiter=",",skiprows=1)
		if plot_type == "mean":
			all_vals = np.loadtxt(filename, unpack=False, delimiter=",",skiprows=1, usecols=range(1, len(v)+1))
			vals = [np.average(v) for v in all_vals]
		elif plot_type == "top":
			all_vals = np.loadtxt(filename, unpack=False, delimiter=",",skiprows=1, usecols=range(1, len(v)+1))
			vals = [max(v) for v in all_vals]
			# plt.title(session)
		plt.xticks(gens[::10])
		plt.scatter(gens, vals)
		# plt.plot(gens, vals)
		# ax.scatter(gens, vals)
		if spline:
			s = UnivariateSpline(gens, vals, s=10000)
			xs = gens
			ys = s(xs)
			plt.plot(xs, ys, color="black")
	return gens, vals





def update_graph(i):
	gens, *vals = np.loadtxt("Sessions/8D_NN24-12-3_test4/utility.csv", unpack=True, delimiter=",", skiprows=1)
	plt.scatter(gens, vals[-1])
	print(len(gens))
# update = update_graph()
# ani = FuncAnimation(plt.gcf(), update_graph, interval=1000)




mpl.rcParams['font.family'] = 'Avenir'
plt.rcParams['font.size'] = 18
plt.rcParams['axes.linewidth'] = 2
fig = plt.figure(figsize=(6, 6))


# session = "ReportData/constant_params/8D_NN24-12-3_pop-100_eli-10_mut-p05_two-child_std-eval_trial1"
# session = "ReportData/constant_params/8D_NN24-12-3_pop-100_eli-10_mut-p05_two-child_std-eval_trial2"
# session = "ReportData/constant_params/8D_NN24-12-3_pop-100_eli-10_mut-p05_two-child_std-eval_trial-3"

# session = "ReportData/population_sizes/8D_NN24-12-3_pop-30_eli-3_mut-p05_two-child_std-eval"
# session = "ReportData/population_sizes/8D_NN24-12-3_pop-60_eli-6_mut-p05_two-child_std-eval"
# session = "ReportData/population_sizes/8D_NN24-12-3_pop-90_eli-9_mut-p05_two-child_std-eval"

# session = "ReportData/mutation_rates/8D_NN24-12-3_pop-100_eli-10_mut-p0_two-child_std-eval"
# session = "ReportData/mutation_rates/8D_NN24-12-3_pop-100_eli-10_mut-p01_two-child_std-eval"
# session = "ReportData/mutation_rates/8D_NN24-12-3_pop-100_eli-10_mut-p10_two-child_std-eval"
# session = "ReportData/mutation_rates/8D_NN24-12-3_pop-100_eli-10_mut-p25_two-child_std-eval"
# session = "ReportData/mutation_rates/8D_NN24-12-3_pop-100_eli-10_mut-p50_two-child_std-eval"

# session = "ReportData/elitism_rates/8D_NN_24-12-3_pop-100_eli-0_mut-p05_two-child_std-eval"
# session = "ReportData/elitism_rates/8D_NN_24-12-3_pop-100_eli-1_mut-p05_two-child_std-eval"
# session = "ReportData/elitism_rates/8D_NN_24-12-3_pop-100_eli-10_mut-p05_two-child_std-eval"

# session = "ReportData/comparative_architecture/8D_NN_24-12-6-3_pop-100_eli-10_mut-p05_two-child_std-eval"

# session = "ReportData/DFSA/DFSA_NN4-8-3_pop-30_eli-3_mut-p05_two-child_std-eval"
# session = "ReportData/DFSA/DFSA_NN4-8-3_pop-100_eli-10_mut-p05_two-child_std-eval"

# session = "ReportData/Adam_Optimizer/8D_NN24-12-3_pop-100_gens-100_minscore-1"
# session = "DFSA_NN4-8-3_pop-30_gens-100_minscore-1"
session = "ReportData/Adam_Optimizer/DFSA_NN4-8-3_pop-100_gens-100_minscore-1"

score_type = "apples"
# score_type = "steps"
# score_type = "utilities"

# plot_type = "all"
plot_type = "top"
# plot_type = "mean"

x_data, y_data = plot_points(session,score_type, plot_type, True)
plt.xlabel("Generations")


filesave = session + "/" + score_type + "-" + plot_type + ".png"
plt.savefig(filesave)
plt.show()









# # ax = fig.add_axes([0, 0, 1, 1])
# # ax.set_xlabel('Energy (eV)', labelpad=10)
# # session="ReportData/DFSA/DFSA_NN4-8-3_pop-30_eli-3_mut-p05_two-child_std-eval"
# session = "ReportData/constant_params/8D_NN24-12-3_pop-100_eli-10_mut-p05_two-child_std-eval_trial1"
# # session = "ReportData/constant_params/8D_NN24-12-3_pop-100_eli-10_mut-p05_two-child_std-eval_trial1/apples.csv"
# # plot_points(session,"apples", "mean")
# score_type = "apples"
# x_data, y_data = plot_points(session,score_type, "all")
# # x_data, y_data = plot_points(session,"apples", "mean")
# # fig.text(.75,-.015,"daslkf", ha="center")
#
# # s = CubicSpline(x_data, y_data)
# # xs = np.linspace(0, 29, 100)
# # xs = np.linspace(0, 100, 200)
#
# # s = UnivariateSpline(x_data, y_data, s=10000)
# # xs = x_data
# # ys = s(xs)
# # plt.plot(xs,ys)
#
#
# # params, params_covariance = optimize.curve_fit(test_func, x_data, y_data, p0=[2, 2])
# # params, params_covariance = optimize.curve_fit(test_func, x_data, y_data, p0=[2, 2])
# # print(params)
# #
# # plt.plot(x_data, test_func(x_data, params[0], params[1]),label='Fitted function')
#
#
# # ax.scatter([1,2,3],[1,2,3])
# # plt.xticks(gens[::10])
# # plt.xticks()
# plt.xlabel("Generations")
# # plt.ylabel("Utility")
#
# # ax.xlabel("Generations")
# # ax.ylabel("Utility")
#
# filesave = session + "/" + score_type + "_graph.png"
# plt.savefig(filesave)
# # plt.show()