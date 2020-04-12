import unittest
import nose.tools as nt
from invisoplanet_detection.statistics.stats import *

"""
Example uses of nosetests:

	- nt.assert_raises(ValueError, function_name, argument_0, argument_1)
	- nt.assert_almost_equal(number_one, number_two, places=3)
	- nt.assert_equal(number_one, number_two)
	- nt.assert_true(boolean_expression)
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

	Ensure TrajectoryInformation.structure_information() returns the appropriate size of arrays
		- dimensions are appropriate
		- last_n = 100, 50, 10, ...
	"""

	def test_trajectory_information(self):

		# Generate some pos_data
		known_bodies = 2
		unknown_bodies = 1
		num_steps = 100
		pos_data = np.ones(((known_bodies + unknown_bodies) * num_steps, 2))

		# Create full sized data set
		trajectory = TrajectoryInformation.structure_information(pos_data, known_bodies, unknown_bodies, last_n=100)

		# Ensure trajectory dimensions are appropriate
		nt.assert_equal(trajectory.shape[0], num_steps)
		nt.assert_equal(trajectory.shape[1], 2)
		nt.assert_equal(trajectory.shape[2], known_bodies)

		# Ensure truncation works
		for divisor in [1, 2, 4, 5, 10]:

			last_n = 100 // divisor
			trajectory = TrajectoryInformation.structure_information(pos_data, known_bodies, unknown_bodies, last_n)

			nt.assert_equal(trajectory.shape[0], num_steps // divisor)

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

		# System A
		a = Likelihood(known_bodies=1, unknown_bodies=1, parameters_filename="ic_files/A_sun_1_0_1.json",
					max_masses=[2 * 9.547919e-4], surrogate_points=5, num_iterations=100, time_step=0.5, last_n=100)
		TestStats.validate_1D_system(a, True)

		# System B
		b = Likelihood(known_bodies=1, unknown_bodies=1, parameters_filename="ic_files/B_sun_jup_1_1_1.json",
					max_masses=[2 * 9.547919e-4], surrogate_points=5, num_iterations=100, time_step=0.5, last_n=100)
		TestStats.validate_1D_system(b, False)

		# System C
		c = Likelihood(known_bodies=2, unknown_bodies=1, parameters_filename="ic_files/C_sun_jup_2_0_1.json",
					max_masses=[2 * 9.547919e-4], surrogate_points=5, num_iterations=100, time_step=0.5, last_n=100)
		TestStats.validate_1D_system(c, True)

		# System D
		d = Likelihood(known_bodies=2, unknown_bodies=1, parameters_filename="ic_files/D_sun_jup_sat_2_1_1.json",
					max_masses=[2 * 0.000285802], surrogate_points=5, num_iterations=100, time_step=0.5, last_n=100)
		TestStats.validate_1D_system(d, False)

		# System E
		e = Likelihood(known_bodies=1, unknown_bodies=2, parameters_filename="ic_files/E_sun_jup_1_1_2.json",
					max_masses=[2 * 9.547919e-4, 2 * 9.547919e-4], surrogate_points=5, num_iterations=100)
		TestStats.validate_2D_system(e, True)

		# System F
		f = Likelihood(known_bodies=1, unknown_bodies=2, parameters_filename="ic_files/F_sat_sun_jup_sat_1_2_2.json",
					max_masses=[2 * 1, 2 * 9.547919e-4], surrogate_points=5, num_iterations=100)
		TestStats.validate_2D_system(f, False)

		# System G
		g = Likelihood(known_bodies=2, unknown_bodies=2, parameters_filename="ic_files/G_sun_jup_sat_2_1_2.json",
					max_masses=[2 * 0.000285802, 2 * 9.547919e-4], surrogate_points=5, num_iterations=100)
		TestStats.validate_2D_system(g, True)

	@staticmethod
	def validate_1D_system(likelihood, ignore_zero=False):

		likelihood.set_eta(1)

		# Ensure equivalence between surrogate model and interpolation
		# Note the number of surrogate points MUST be odd to contain the actual mass
		surrogate_logs = likelihood.surrogate_values()
		_, posterior_logs = likelihood.linspace(num=likelihood.surrogate_points, floor=None)
		for i in range(likelihood.surrogate_points):
			nt.assert_almost_equal(surrogate_logs[i], posterior_logs[i])

		# Ensure the interpolated point with the actual masses has posterior zero
		true_masses = extract_unknown_ic_masses(likelihood.parameters_filename, likelihood.known_bodies)
		posterior = likelihood.log_posterior(true_masses)
		nt.assert_almost_equal(posterior, 0)

		# Ensure the posterior at the actual masses is the global maximum
		# Note the count MUST be even to avoid the actual mass
		count = 100
		_, posterior_logs = likelihood.linspace(num=count, floor=None)
		for i in range(count):
			if ignore_zero and i == 0:
				continue
			nt.assert_true(posterior_logs[i] < 0)

	@staticmethod
	def validate_2D_system(likelihood, ignore_zero=False):

		likelihood.set_eta(1)

		# Ensure equivalence between surrogate model and interpolation
		# Note the number of surrogate points MUST be odd to contain the actual mass
		surrogate_logs = likelihood.surrogate_values()
		_, _, posterior_logs = likelihood.linspace(num=likelihood.surrogate_points, floor=None)
		for i in range(likelihood.surrogate_points):
			for j in range(likelihood.surrogate_points):
				nt.assert_almost_equal(surrogate_logs[i, j], posterior_logs[i, j])

		# Ensure the interpolated point with the actual masses has posterior zero
		true_masses = extract_unknown_ic_masses(likelihood.parameters_filename, likelihood.known_bodies)
		posterior = likelihood.log_posterior(true_masses)
		nt.assert_almost_equal(posterior, 0)

		# Ensure the posterior at the actual masses is the global maximum
		# Note the count MUST be even to avoid the actual mass
		count = 100
		_, _, posterior_logs = likelihood.linspace(num=count, floor=None)
		for i in range(count):
			for j in range(count):
				if ignore_zero and (i == 0 or j == 0):
					continue
				nt.assert_true(posterior_logs[i, j] < 0)

	def test_logarithm(self):

		# Ensure posterior is -np.inf when given an invalid mass
		likelihood = Likelihood(known_bodies=2, unknown_bodies=1,
								parameters_filename="ic_files/D_sun_jup_sat_2_1_1.json", max_masses=[2 * 0.000285802],
								surrogate_points=1, num_iterations=100, time_step=0.5, last_n=100)
		likelihood.set_eta(1)
		nt.assert_false(np.isfinite(likelihood.log_posterior([-1])))
		nt.assert_false(np.isfinite(likelihood.log_posterior([1])))
		nt.assert_true(np.isfinite(likelihood.log_posterior([0])))
		nt.assert_true(np.isfinite(likelihood.log_posterior([0.00028])))


if __name__ == '__main__':

	unittest.main()
