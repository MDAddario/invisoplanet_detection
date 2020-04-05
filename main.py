# Import the animations submodule
from invisoplanet_detection.animations import *

if __name__ == "__main__":

	"""
	Note, the position data supplied to the planet_creator() must be formatted
	as a list of lists of length 3. That is, the following is an example of 
	proper position data:
	>>>> position_data = [[1, 0, 0], [1, 1, 0], [1, 2, 0], [1, 3, 0]]
	"""

	# Create sample position data
	parameter = np.linspace(0, 1, num=120)
	p1_x_pos = 10 * np.cos(2 * np.pi * parameter - np.pi / 4) - 5
	p1_y_pos = 10 * np.sin(2 * np.pi * parameter - np.pi / 4) - 5
	p1_z_pos = np.zeros(len(parameter))

	p1_pos = []
	for x, y, z in zip(p1_x_pos, p1_y_pos, p1_z_pos):
		p1_pos.append([x, y, z])

	p2_x_pos = -10 * np.sin(2 * np.pi * parameter + np.pi) + 5
	p2_y_pos = -10 * np.cos(2 * np.pi * parameter + np.pi) + 5
	p2_z_pos = np.zeros(len(parameter))

	p2_pos = []
	for x, y, z in zip(p2_x_pos, p2_y_pos, p2_z_pos):
		p2_pos.append([x, y, z])

	# Create sample masses
	p1_mass = 4
	p2_mass = p1_mass

	# load in the position data from the simulator
	p1_mass = 1
	p2_mass = 0.5
	p3_mass = 0.3

	with open("invisoplanet_detection/simulations/testx.txt", "r") as file:
		posdata = np.genfromtxt(file)

	n_steps = int(len(posdata)/3)

	p1_idx = np.arange(len(posdata), step=3)
	p2_idx = np.arange(1, len(posdata), step=3)
	p3_idx = np.arange(2, len(posdata), step=3)

	p1_xy_pos = posdata[p1_idx]
	p2_xy_pos = posdata[p2_idx]
	p3_xy_pos = posdata[p3_idx]

	p1_x_pos = p1_xy_pos[:, 0]
	p1_y_pos = p1_xy_pos[:, 1]
	p2_x_pos = p2_xy_pos[:, 0]
	p2_y_pos = p2_xy_pos[:, 1]
	p3_x_pos = p3_xy_pos[:, 0]
	p3_y_pos = p3_xy_pos[:, 1]

	p1_z_pos = np.zeros(n_steps)
	p2_z_pos = p1_z_pos
	p3_z_pos = p1_z_pos

	p1_pos = []
	for x, y, z in zip(p1_x_pos, p1_y_pos, p1_z_pos):
		p1_pos.append([x, y, z])

	p2_pos = []
	for x, y, z in zip(p2_x_pos, p2_y_pos, p2_z_pos):
		p2_pos.append([x, y, z])

	p3_pos = []
	for x, y, z in zip(p3_x_pos, p3_y_pos, p3_z_pos):
		p3_pos.append([x, y, z])

	# Create planet models
	p1 = planet_creator(p1_pos, p1_mass, "red")
	p2 = planet_creator(p2_pos, p2_mass, "blue")
	p3 = planet_creator(p3_pos, p3_mass, "blue")

	# Run the animation!
	pyglet.app.run()
