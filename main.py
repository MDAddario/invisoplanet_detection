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

	# Create planet models
	p1 = planet_creator(p1_pos, p1_mass, "red")
	p2 = planet_creator(p2_pos, p2_mass, "blue")

	# Run the animation!
	pyglet.app.run()
