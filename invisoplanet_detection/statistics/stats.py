from tqdm import tqdm
import os
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

			# We only have one body to worry about
			guess_mass_1 = guess_masses[0]

			# Create the new trajectory object (purposely no constructor call)
			trajectory = TrajectoryInformation

			# Determine indices of nearest interpolation objects
			index = np.interp(guess_mass_1, likelihood.mass_1_arr, np.arange(likelihood.surrogate_points))
			left_index = int(index)
			right_index = left_index + 1
			weight = index - left_index

			# Interpolate!
			trajectory.pos_data = likelihood.surrogate_model[left_index] * weight \
								+ likelihood.surrogate_model[right_index] * (1 - weight)

			return trajectory

		else:
			raise ValueError(
				"Number of unknown bodies cannot exceed 1 due to hard-coding considerations."
			)

	@staticmethod
	def log_gaussian_difference(cur_trajectory, true_trajectory, eta):
		"""
		Computes the gaussian difference between the TrajectoryInformation corresponding to the current guess
		of unknown masses and the true masses.
		In reality, returns the log of the gaussian differences
		Difference should only be computed for the trajectories of the KNOWN bodies
		"""
		return -0.5 * np.sum(np.square(cur_trajectory.pos_data - true_trajectory.pos_data) / eta + np.log(eta))


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

	- self.surrogate_model | n-dimensional array of type TrajectoryInformation
		Array containing the trajectory information for the planetary trajectories
		Array will have self.unknown_bodies indices
		Each axis will have self.surrogate_points entries

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

		# Let everyone know that you hard coded the fact that you're only checking for one additional planey
		if self.unknown_bodies != 1:
			raise ValueError(
				"The statistics submodule currently only supports the detection of exactly one potential unknown"
				"planet. Please specify unknown_bodies=1 or contact the developer to generalize the code."
			)

		# Check the specified number of bodies matches the initial conditions file
		if count_ic_bodies(self.parameters_filename) != self.known_bodies + self.unknown_bodies:
			raise ValueError(
				"The number of bodies in the initial conditions file does not match the number of bodies"
				"specified by the sum of known and unknown bodies. Please make sure that there are bodies"
				"in the initial conditions file even for bodies with zero mass. That is, if the system truly"
				"consists of 2+0 (2 known and 0 unknown bodies) but you are detecting the presence of one"
				"possible additional planet, the initial conditions file should contain 3 bodies, with the"
				"third one being of zero mass."
			)

		# Class set parameters
		self.mass_1_arr = None
		self.surrogate_model = None
		#self.construct_surrogate_model()
		self.true_trajectory_information = None
		self.configure_true_trajectory()

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

		else:
			raise ValueError(
				"Number of unknown bodies cannot exceed 1 due to hard-coding considerations."
			)

	def configure_true_trajectory(self):
		"""
		Run the simulation that corresponds to the true masses of the unknown bodies and save the trajectory information
		"""
		true_unknown_masses = extract_unknown_ic_masses(self.parameters_filename, self.known_bodies)
		print(true_unknown_masses)
		exit()
		self.true_trajectory_information = self.extract_trajectory_information(true_unknown_masses)

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
			if not 0 <= guess_masses <= max_mass:
				return -np.inf

		# Else return the likelihood
		return self.log_likelihood(guess_masses)


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
	max_masses = [1]
	eta = 1e-4
	surrogate_points = 5

	# Construct the likelihood object
	likelihood = Likelihood(known_bodies, unknown_bodies, parameters_filename, num_iterations, time_step,
							max_masses, eta, surrogate_points)

	# Use the posterior function
	likelihood.log_posterior([1e-5])

	exit()
