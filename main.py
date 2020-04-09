# Import the animations submodule
#from invisoplanet_detection.animations import *
from invisoplanet_detection.statistics import *

if __name__ == "__main__":

	"""
	A NOTE FOR THE PARAMETER FILENAME CONVENTION:
	>>>> data_X_Y_Z.txt
	X: Number of known bodies
	Y: Number of unknown bodies
	Z: Maximum number of unknown bodies we are trying to detect
	NOTE THAT Z >= Y !!!!
	The data file should contain X+Z body entries.
	If Z > Y, zero fill the remaining planets.
	"""

	# Setup the likelihood object
	known_bodies = 1
	unknown_bodies = 2
	parameters_filename = "invisoplanet_detection/data/sat_sun_jup_sat_1_2_2.json"
	max_masses = np.array([1, 9.547919e-4]) * 2  # Actual masses times 2
	surrogate_points = 9
	num_iterations = 200  # 20_000 normally
	time_step = 0.5

	# Construct the likelihood object
	likelihood = Likelihood(known_bodies, unknown_bodies, parameters_filename, max_masses, surrogate_points,
							num_iterations, time_step)

	# Set the eta value
	likelihood.set_eta(1)

	# Plot the posterior
	likelihood.plot_posterior("figures/1_2_2_posterior.pdf", num=20, floor=-10)

	exit()

	"""
	Note, the position data supplied to the planet_creator() must be formatted
	as a list of lists of length 3. That is, the following is an example of 
	proper position data:
	>>>> position_data = [[1, 0, 0], [1, 1, 0], [1, 2, 0], [1, 3, 0]]
	"""

	'''
	in_file = "invisoplanet_detection/data/binary.json"
	out_file = "invisoplanet_detection/data/testx.txt"

	# run the n-body simulation
	space_f = simulate(in_file, 20000, 0.5, out_file)

	# find the final values of pos, vel, and mass
	x, v, all_mass = space_f.arrayvals()

	n_bodies = len(space_f.bodies)

	with open(out_file, "r") as file:
		posdata = np.genfromtxt(file)

	n_steps = int(len(posdata)/n_bodies)

	all_pos = []
	for i in range(n_bodies):
		pi_idx = np.arange(i, len(posdata), step=n_bodies)

		pi_xy_pos = posdata[pi_idx]

		pi_x_pos = pi_xy_pos[:, 0]
		pi_y_pos = pi_xy_pos[:, 1]
		pi_z_pos = np.zeros(n_steps)

		pi_pos = []
		for x, y, z in zip(pi_x_pos, pi_y_pos, pi_z_pos):
			pi_pos.append([x, y, z])

		all_pos.append(pi_pos)

	all_colours = ["red", "blue", "yellow", "green"]


	# all_planets = []
	for i in range(n_bodies):
		pi = planet_creator(all_pos[i], all_mass[i], all_colours[i])

	#
	# # jupiter orbital parameters
	# a = 5.20336301  # semimajor axis, in AU
	# ecc = 0.04839266  # orbital eccentricity
	# T = T = 4330.595 # tropical orbital period in days
	# jupiter_orbit_params = [a, ecc]
	#
	# kep_x_pos, kep_y_pos, delta = kepler_test(20000, 0.5, jupiter_orbit_params, all_pos[1])
	# kep_z_pos = np.zeros(n_steps)
	#
	# kep_pos = []
	# for x, y, z in zip(kep_x_pos, kep_y_pos, kep_z_pos):
	# 	kep_pos.append([x, y, z])
	#
	# pkep = planet_creator(kep_pos, all_mass[1]*1.1, all_colours[3])
	#
	# print(delta)

	# Run the animation!
	pyglet.app.run()
	'''
