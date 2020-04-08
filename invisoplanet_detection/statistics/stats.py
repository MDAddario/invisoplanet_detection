import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from invisoplanet_detection.simulations import *


class TrajectoryInformation:
	"""
	Reduces the complete time-ordered phase space information into a smaller data set that
	will be used to compute the likelihood function.

	For example, extract the position data for every body at 10 evenly spaced times
	from the last 10 % of the trajectory data.
	"""

	def __init__(self, posdata, known_bodies, unknown_bodies):
		"""
		self.pos_data:
		axis 0 = time_step
		axis 1 = position_coordinate
		axis 2 = body index
		"""
		# Aggregate the position data into a 3D matrix
		all_pos = []

		for i in range(known_bodies):
			pi_idx = np.arange(i, len(posdata), step=known_bodies + unknown_bodies)
			all_pos.append(posdata[pi_idx])

		# Store the data as the only attribute
		self.pos_data = np.stack(all_pos, axis=2)

		# Reduce the information
		self.reduce_information()

	def reduce_information(self):
		"""
		Truncate the data set of position data
		"""
		# CURRENTLY NO REDUCTION
		self.pos_data = self.pos_data

	@staticmethod
	def interpolate_information(likelihood, guess_masses):
		"""
		Uses interpolation of surrogate model to determine the trajectory_information of a set of guess masses
		"""
		# Treat cases differently depending on the number of unknown bodies
		if likelihood.unknown_bodies == 1:

			return TrajectoryInformation.interpolate(guess_masses[0], likelihood.mass_1_arr, likelihood.surrogate_model)

		elif likelihood.unknown_bodies == 2:

			# Interpolate for mass 1
			horizontal = TrajectoryInformation.interpolate(guess_masses[0], likelihood.mass_1_arr,
															likelihood.surrogate_model)

			# Interpolate for mass 2
			vertical = TrajectoryInformation.interpolate(guess_masses[1], likelihood.mass_2_arr,
															likelihood.surrogate_model)

			# Average them out
			trajectory = TrajectoryInformation
			trajectory.pos_data = (horizontal.pos_data + vertical.pos_data) / 2
			return trajectory

		else:
			Likelihood.hardcoded_error_message()

	@staticmethod
	def interpolate(guess_mass, mass_array, surrogate_model):

		# Create the new trajectory object (purposely no constructor call)
		trajectory = TrajectoryInformation

		# Determine indices of nearest interpolation objects
		index = np.interp(guess_mass, mass_array, np.arange(len(mass_array)))
		left_index = int(index)
		right_index = left_index + 1
		weight = index - left_index

		# Handle right edge case!
		if right_index == len(mass_array):
			right_index -= 1

		# Interpolate!
		trajectory.pos_data = surrogate_model[left_index].pos_data * (1 - weight) \
							+ surrogate_model[right_index].pos_data * weight

		return trajectory

	@staticmethod
	def log_gaussian_difference(cur_trajectory, true_trajectory, eta):
		"""
		Computes the gaussian difference between the TrajectoryInformation corresponding to the current guess
		of unknown masses and the true masses.
		In reality, returns the log of the gaussian differences
		Difference should only be computed for the trajectories of the KNOWN bodies
		"""
		difference = -0.5 * np.sum(np.square(cur_trajectory.pos_data - true_trajectory.pos_data) / eta + np.log(eta))

		# Handle nans
		if np.isnan(difference):
			return -np.inf
		return difference


class Likelihood:
	"""
	# USER SET ATTRIBUTES

	- self.known_bodies | int
		Number of planets of known mass

	- self.unknown_bodies | int
		Max number of planets of unknown mass

	- self.parameters_filename | string
		Filename used to retrieve the initial conditions for the bodies

	- self.num_iterations | int
		Number of iterations of the simulation to perform

	- self.time_step | float
		The value for dt used when conducting the simulations

	- self.max_masses | list of floats
		Maximum possible masses for the unknown bodies

	- self.eta | float
		Scale parameter used in exponential likelihood function
		Controls the size of steps in MCMC iterations

	- self.surrogate_points | integer
		Number of mass samples to use when computing surrogate model
		This number will correspond to each dimension of mass, i.e. for a model with 2 additional hidden
		bodies, there will be self.surrogate_points**2 points in interpolation space

	# CLASS SET ATTRIBUTES

	- self.mass_1_arr | array of floats
		Mass values for unknown body 1 corresponding to the surrogate model points
		Used for interpolation

	- self.mass_2_arr | array of floats
		Mass values for unknown body 2 corresponding to the surrogate model points
		Used for interpolation

	- self.surrogate_model | n-dimensional array of type TrajectoryInformation
		Array containing the trajectory information for the planetary trajectories
		Array will have self.unknown_bodies indices
		Each axis will have self.surrogate_points entries

	- self.true_unknown_masses | list of floats
		True masses of the unknown bodies

	- self.true_trajectory_information | TrajectoryInformation object
		Information corresponding to the true planet trajectories
	"""

	def __init__(self, known_bodies, unknown_bodies, parameters_filename, num_iterations, time_step,
	             max_masses, eta, surrogate_points):
		"""
		Sets the default parameters for the various user set attributes, and computes class set attributes
		"""

		# User set parameters
		self.known_bodies = known_bodies
		self.unknown_bodies = unknown_bodies
		self.parameters_filename = parameters_filename
		self.num_iterations = num_iterations
		self.time_step = time_step
		self.max_masses = max_masses
		self.eta = eta
		self.surrogate_points = surrogate_points

		# Let everyone know that you hard coded the fact that you're only checking for one or two additional planets
		if self.unknown_bodies not in [1, 2]:
			Likelihood.hardcoded_error_message()

		# Check the specified number of bodies matches the initial conditions file
		if count_ic_bodies(self.parameters_filename) != self.known_bodies + self.unknown_bodies:
			raise ValueError(
				"The number of bodies in the initial conditions file does not match the number of bodies"
				"specified by the sum of known and unknown bodies. Please make sure that there are bodies"
				"in the initial conditions file even for bodies with zero mass. That is, if the system truly"
				"consists of 2_0_1 (2 known bodies, 0 unknown, detecting up to 1 additional planet), then the"
				"initial conditions file should contain 3 bodies, with the third one being of zero mass."
			)

		# Class set parameters
		self.mass_1_arr = None
		self.mass_2_arr = None
		self.surrogate_model = None
		self.construct_surrogate_model()

		self.true_unknown_masses = None
		self.true_trajectory_information = None
		self.configure_true_trajectory()

	@staticmethod
	def hardcoded_error_message():
		raise ValueError(
			"Maximum number of unknown bodies cannot exceed 2 due to hard-coding considerations."
			"Please set the unknown_bodies field to either 1 or 2."
		)

	def extract_trajectory_information(self, unknown_masses):
		"""
		For a given set of guess masses, run the simulation and return the trajectory information
		"""

		# Run the n-body simulation
		os.chdir(os.path.dirname(os.path.abspath(__file__)))
		out_file = "../data/likelihood_output.txt"
		simulate(self.parameters_filename, self.num_iterations, self.time_step, out_file, unknown_masses)

		# Extract the position data
		with open(out_file, "r") as file:
			posdata = np.genfromtxt(file)

		# Delete the file
		os.remove(out_file)

		# Return a reduced and formatted object
		return TrajectoryInformation(posdata, self.known_bodies, self.unknown_bodies)

	def construct_surrogate_model(self):
		"""
		Loop over the entire sample space of unknown_masses and configure the surrogate model at each point
		"""

		# Treat cases differently depending on the number of unknown bodies
		if self.unknown_bodies == 1:

			# Allocate memory for surrogate model
			self.surrogate_model = np.empty(self.surrogate_points, dtype=object)

			# Determine mass values to simulate
			self.mass_1_arr = np.linspace(0, self.max_masses[0], num=self.surrogate_points)

			# Construct the surrogate model
			for index, mass_1 in enumerate(tqdm(self.mass_1_arr, desc='Mass 1 list')):
				self.surrogate_model[index] = self.extract_trajectory_information([mass_1])

		elif self.unknown_bodies == 2:

			# Allocate memory for surrogate model
			self.surrogate_model = np.empty((self.surrogate_points, self.surrogate_points), dtype=object)

			# Determine mass values to simulate
			self.mass_1_arr = np.linspace(0, self.max_masses[0], num=self.surrogate_points)
			self.mass_2_arr = np.linspace(0, self.max_masses[1], num=self.surrogate_points)

			# Construct the surrogate model
			for i_1, mass_1 in enumerate(tqdm(self.mass_1_arr, desc='Mass 1 list')):
				for i_2, mass_2 in enumerate(tqdm(self.mass_2_arr, desc='Mass 2 list')):
					self.surrogate_model[i_1, i_2] = self.extract_trajectory_information([mass_1, mass_2])

		else:
			Likelihood.hardcoded_error_message()

	def configure_true_trajectory(self):
		"""
		Run the simulation that corresponds to the true masses of the unknown bodies and save the trajectory information
		"""
		self.true_unknown_masses = extract_unknown_ic_masses(self.parameters_filename, self.known_bodies)
		self.true_trajectory_information = self.extract_trajectory_information(self.true_unknown_masses)

	def interpolate_trajectory_information(self, guess_masses):
		"""
		Uses interpolation of surrogate model to determine the trajectory_information of a set of guess masses
		Leave the details up to the representation of the TrajectoryInformation object
		"""
		return TrajectoryInformation.interpolate_information(self, guess_masses)

	def log_likelihood(self, guess_masses):
		"""
		Computes the gaussian difference between the TrajectoryInformation corresponding to the current guess
		of unknown masses and the true masses.
		In reality, returns the log of the gaussian differences
		Difference should only be computed for the trajectories of the KNOWN bodies
		"""
		# Determine the trajectory associated with the guess masses
		trajectory = self.interpolate_trajectory_information(guess_masses)

		# Compute the log gaussian difference between the trajectories
		return TrajectoryInformation.log_gaussian_difference(trajectory, self.true_trajectory_information, self.eta)

	def log_posterior(self, guess_masses, x=None, y=None, y_err=None):
		"""
		Serve as the function that will be passed to the MCMC iterator
		Will incorporate the prior: i.e., if masses are negative or greater than max_masses, this function
		will return -np.inf
		"""
		# Check masses are within the bounds
		for guess_mass, max_mass in zip(guess_masses, self.max_masses):
			if not 0 <= guess_mass <= max_mass:
				return -np.inf

		# Else return the likelihood
		return self.log_likelihood(guess_masses)

	def plot_posterior(self, filename=None, num=100):
		"""
		Plot the posterior associated with the optimization problem
		"""
		# Compute the surrogate model gaussian differences
		surrogate_logs = []
		for i in range(self.surrogate_points):
			surrogate_logs.append(TrajectoryInformation.log_gaussian_difference(self.surrogate_model[i],
									self.true_trajectory_information, self.eta))

		# Compute the interpolated model gaussian differences
		posterior_logs = []
		interp_masses = np.linspace(0, self.max_masses[0], num=num)
		for mass_1 in interp_masses:
			posterior_logs.append(self.log_posterior([mass_1]))

		"""
		Note that we expect there to be a local maximum at the true mass of planet 3, with a value
		of 0 for the logarithm when eta=1
		"""

		# Create MPL figure
		fig = plt.figure(figsize=(16, 9))
		ax = fig.add_subplot(111)
		font = 20

		ax.scatter(self.mass_1_arr, surrogate_logs, c='blue', label="Surrogate model posterior")
		ax.plot(interp_masses, posterior_logs, c='blue', label="Interpolated posterior")
		ax.axvline(self.true_unknown_masses[0], c='red', label="True mass")
		ax.legend(fontsize=font)
		ax.set_xlim([0, self.max_masses[0]])
		ax.set_xlabel(r'Mass of 1st invisible body $m_1$', fontsize=font)
		ax.set_ylabel(r'Log. posterior probability $\log[P(m_1)]$', fontsize=font)
		ax.tick_params(axis='both', which='major', labelsize=font)
		ax.tick_params(axis='both', which='minor', labelsize=font)
		if filename is not None:
			plt.savefig(filename)
		plt.show()


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
	known_bodies = 2
	unknown_bodies = 1
	parameters_filename = "../data/sun_jupiter_saturn_2_1_1.json"
	num_iterations = 20000
	time_step = 0.5
	max_masses = [0.000285802 * 2]
	eta = 1
	surrogate_points = 3

	# Construct the likelihood object
	likelihood = Likelihood(known_bodies, unknown_bodies, parameters_filename, num_iterations, time_step,
							max_masses, eta, surrogate_points)

	# Plot the posterior
	likelihood.plot_posterior("posterior.pdf", num=200)

"""
GOALS:
	- Add parameters first_n, last_n to deal with either the first n% or last n% of the position data
	- Consider changing zero masses to epsilon masses
	- Add unit tests
"""
