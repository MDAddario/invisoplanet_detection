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

	in_file = "invisoplanet_detection/data/sun_jupiter.json"
	out_file = "invisoplanet_detection/data/testx.txt"

	# run the n-body simulation
	space_f = simulate(in_file, 10000, 0.5, out_file)

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

	all_colours = ["red", "blue", "yellow"]


	# all_planets = []
	for i in range(n_bodies):
		pi = planet_creator(all_pos[i], all_mass[i], all_colours[i])

	# jupiter orbital parameters
	a = 5.20336301  # semimajor axis, in AU
	ecc = 0.04839266  # orbital eccentricity
	T = 4332.589  # sidereal orbital period in days
	omega = 2 * np.pi / T  # angular velocity

	# find the time after t=0 at which jupiter's orbit is at perihelion
	r_jupiter = np.sqrt(np.sum(np.power(all_pos[1], 2), axis=1))
	# perihelion_t = np.where(r_jupiter == np.min(r_jupiter))[0][0]
	perihelion_t = 870
	perihelion_theta = omega * perihelion_t
	# jupiter hits perihelion on 19 jan 2023, at 16:44EST (1020 days away)

	kep_x_pos, kep_y_pos = kepler_jupiter(10000, 0.5, perihelion_theta)
	kep_z_pos = np.zeros(n_steps)

	kep_pos = []
	for x, y, z in zip(kep_x_pos, kep_y_pos, kep_z_pos):
		kep_pos.append([x, y, z])

	pkep = planet_creator(kep_pos, all_mass[1]*1.1, all_colours[2])

	# Run the animation!
	pyglet.app.run()
