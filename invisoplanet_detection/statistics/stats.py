from tqdm import tqdm
from invisoplanet_detection.simulations import *


class TrajectoryInformation:
	"""
	Reduces the complete time-ordered phase space information into a smaller data set that
	will be used to compute the likelihood function.

	For example, extract the position data for every body at 10 evenly spaced times
	from the last 10 % of the trajectory data.
	"""

	def __init__(self, posdata, known_bodies):
		"""
		axis 0 = time_step
		axis 1 = position_coordinate
		axis 2 = body index
		"""
		# Aggregate the position data into a 3D matrix
		all_pos = []

		for i in range(known_bodies):
			pi_idx = np.arange(i, len(posdata), step=known_bodies)
			all_pos.append(posdata[pi_idx])

		# Store the data as the only attribute
		self.pos_data = np.stack(all_pos, axis=2)

	def extract_information_from_file(self):
		pass

	def compute_log_gaussian_difference(self):
		pass

	def interpolate_information(self):
		pass


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
		If unset, program will initialize these to double the mass of the largest body in the system

	- self.eta | float
		Scale parameter used in exponential likelihood function
		Controls the size of steps in MCMC iterations

	- self.surrogate_points | integer
		Number of mass samples to use when computing surrogate model
		This number will correspond to each dimension of mass, i.e. for a model with 2 additional hidden
		bodies, there will be self.surrogate_points**2 points in interpolation space

	# CLASS SET ATTRIBUTES

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
		self.surrogate_model = None
		self.construct_surrogate_model()
		self.true_trajectory_information = None
		self.configure_true_trajectory()

	def extract_trajectory_information(self, unknown_masses):
		"""
		For a given set of guess masses, run the simulation and return the trajectory information
		"""

		# Run the n-body simulation
		out_file = "invisoplanet_detection/data/likelihood_output.txt"
		simulate(self.parameters_filename, self.num_iterations, self.time_step, out_file, unknown_masses)

		# Extract the position data
		with open(out_file, "r") as file:
			posdata = np.genfromtxt(file)

		# Return a reduced and formatted object
		return TrajectoryInformation(posdata, self.known_bodies)

	def construct_surrogate_model(self):
		"""
		Loop over the entire sample space of unknown_masses and configure the surrogate model at each point
		"""

		# Treat cases differently depending on the number of unknown bodies
		if self.unknown_bodies == 1:

			# Allocate memory for surrogate model
			self.surrogate_model = np.empty(self.surrogate_points, dtype=object)

			# Determine mass values to simulate
			mass_1_list = np.linspace(0, self.max_masses[0], num=self.surrogate_points)

			# Construct the surrogate model
			for index, mass_1 in enumerate(tqdm(mass_1_list, desc='Mass 1 list')):
				self.surrogate_model[index] = self.extract_trajectory_information([mass_1])

		elif self.unknown_bodies == 2:

			# Allocate memory for surrogate model
			self.surrogate_model = np.empty((self.surrogate_points, self.surrogate_points), dtype=object)

			# Determine mass values to simulate
			mass_1_list = np.linspace(0, self.max_masses[0], num=self.surrogate_points)
			mass_2_list = np.linspace(0, self.max_masses[1], num=self.surrogate_points)

			# Construct the surrogate model
			for i_1, mass_1 in enumerate(tqdm(mass_1_list, desc='Mass 1 list')):
				for i_2, mass_2 in enumerate(tqdm(mass_2_list, desc='Mass 2 list')):
					self.surrogate_model[i_1, i_2] = self.extract_trajectory_information([mass_1, mass_2])

		else:
			raise ValueError(
				"Number of unknown bodies cannot exceed 2 due to hard-coding considerations."
			)

	def configure_true_trajectory(self):
		"""
		Run the simulation that corresponds to the true masses of the unknown bodies and save the trajectory information
		"""
		true_unknown_masses = extract_unknown_ic_masses(self.parameters_filename, self.known_bodies)
		self.true_trajectory_information = self.extract_trajectory_information(true_unknown_masses)

	def interpolate_trajectory_information(self):
		pass

	"""
	Uses interpolation of surrogate model to determine the trajectory_information of a set of masses
	
	Inputs:
		- Guess of unknown masses
	Outputs:
		- Interpolated TrajectoryInformation
	"""

	def MCMC_log_likelihood(self):
		pass

	"""
	Computes the gaussian difference between the TrajectoryInformation corresponding to the current guess
	of unknown masses and the true masses.
	In reality, returns the log of the gaussian differences
	Difference should only be computed for the trajectories of the KNOWN bodies
	
	Inputs:
		- Guess of unknown masses
	Outputs:
		- Float (the difference evaluation)
	"""

	def MCMC_log_posterior(self):
		pass

	"""
	Serve as the function that will be passed to the MCMC iterator
	Will incorcorate the prior: i.e., if masses are negative or greater than max_masses, this function
	will return -np.inf

	Inputs:
		- theta
		- x
		- y
		- yerr
	Outputs:
		- Float (posterior log)
	"""
