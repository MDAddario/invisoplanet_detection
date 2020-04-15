# Import the animations submodule
from invisoplanet_detection.animations import *
from invisoplanet_detection.simulations import *

if __name__ == "__main__":

	"""
	Note, the position data supplied to the planet_creator() must be formatted
	as a list of lists of length 3. That is, the following is an example of 
	proper position data:
	>>>> position_data = [[1, 0, 0], [1, 1, 0], [1, 2, 0], [1, 3, 0]]
	"""

	in_file = "invisoplanet_detection/data/jup_jupmoon_sun_2_1_1.json"
	out_file = "invisoplanet_detection/data/testx.txt"

	# run the n-body simulation
	space_f = simulate(in_file, 20_000, 0.01, out_file)

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
			pi_pos.append(np.asarray([x, y, z]))

		all_pos.append(pi_pos)

	all_colours = ["red", "blue", "yellow", "green", "purple", "red", "blue", "yellow", "green", "purple",]

	all_mass = np.divide([0.95, 1, 2, 1, 3, 2.5, 2, 2, 4], 30)

	for i in range(n_bodies):
		pi = planet_creator(position_data=all_pos[i], radius=all_mass[i], path_radius=0.005,
							color=all_colours[i], num=100)


	# # jupiter orbital parameters
	# a = 5.20336301  # semimajor axis, in AU
	# ecc = 0.04839266  # orbital eccentricity
	# T = T = 4330.595 # tropical orbital period in days
	# jupiter_orbit_params = [a, ecc, T]
	#
	# kep_x_pos, kep_y_pos, delta = kepler_check(10000, 1., jupiter_orbit_params, all_pos[1])
	# kep_z_pos = np.zeros(n_steps)
	#
	# kep_pos = []
	# for x, y, z in zip(kep_x_pos, kep_y_pos, kep_z_pos):
	# 	kep_pos.append(np.asarray([x, y, z]))
	#
	# pkep = planet_creator(position_data=kep_pos, radius=all_mass[1] / 2, path_radius=0.05,
	# 					  color=all_colours[3], num=100)
	#
	# print(delta)

	# Run the animation!
	pyglet.app.run()
