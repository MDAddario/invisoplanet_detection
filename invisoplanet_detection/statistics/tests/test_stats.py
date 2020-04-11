import unittest
import nose.tools as nt
from invisoplanet_detection.statistics.stats import *

"""
Example uses of nosetests:

	- nt.assert_raises(ValueError, function_name, argument_0, argument_1)
	- nt.assert_almost_equal(number_one, number_two, places=3)
	- nt.assert_equal(number_one, number_two)
"""


class TestStats(unittest.TestCase):
	"""
	Raise value errors for:
		- Num bodies != num bodies in IC
		- Forgetting to set eta
		- Num unknown = 0, 3
		- last_n > 100, <= 0

	For each of the following systems:

		- 1_0_1: A
		- 1_1_1: B
		- 2_0_1: C
		- 2_1_1: D      <- My favorite system

		- 1_1_2: E
		- 1_2_2: F
		- 2_1_2: G

			- Ensure that the interpolated posteriors correspond to the surrogate posteriors
			- Ensure that, for eta=1, the interpolated posterior with real masses is zero
			- Ensure that, for eta=1, all other interpolated posteriors are less than zero

	Ensure log_posterior provides -np.inf when given a mass outside of the mass bounds
		- One mass below and one mass above

	Ensure constructor for Likelihood() validates EVERY input properly
		- Consider every other possibility
		- Also code this in the constructor

	Ensure TrajectoryInformation.structure_information() returns the appropriate size of arrays
		- last_n = 100, 50, 10, ...
	"""

	def test_value_errors(self):

		# IC_file body number != known_bodies + unknown_bodies
		nt.assert_raises(ValueError, Likelihood, 2, 2, "ic_files/D_sun_jup_sat_2_1_1.json", [1], 1)

		# Forgetting to set eta
		likelihood = Likelihood(known_bodies=2, unknown_bodies=1,
								parameters_filename="ic_files/D_sun_jup_sat_2_1_1.json", max_masses=[2 * 0.000285802],
								surrogate_points=1, num_iterations=100, time_step=0.5, last_n=100)
		nt.assert_raises(ValueError, likelihood.log_likelihood, [1])

		# Illegal number of unknowns
		nt.assert_raises(ValueError, Likelihood, 3, 0, "ic_files/D_sun_jup_sat_2_1_1.json", [1], 1)

		# Illegal value for last_n
		nt.assert_raises(ValueError, Likelihood, 2, 1, "ic_files/D_sun_jup_sat_2_1_1.json", [1], 1, 1, last_n=-20)

	def test_statistics(self):

		# Construct the likelihood object
		likelihood = Likelihood(known_bodies=2, unknown_bodies=1,
								parameters_filename="ic_files/D_sun_jup_sat_2_1_1.json", max_masses=[2 * 0.000285802],
								surrogate_points=1, num_iterations=100, time_step=0.5, last_n=100)

		# Set the eta value
		likelihood.set_eta(1)

	def print_statement(self):
		print('wiwza')


if __name__ == '__main__':

	unittest.main()
