class TrajectoryInformation:

    """
    Reduces the complete time-ordered phase space information into a smaller data set that
    will be used to compute the likelihood function.

    For example, extract the position data for every body at 10 evenly spaced times
    from the last 10 % of the trajectory data.
    """

    extract_information_from_file():
        pass

    compute_log_gaussian_difference():
        pass

    interpolate_information():
        pass


class Likelihood:

    """
    Attributes:

        # USER SET ATTRIBUTES

        - self.known_bodies | int
            Number of planets of known mass

        - self.unknown_bodies | int
            Max number of planets of unknown mass

        - self.true_init_parameters | unsure of dtype
            True initial parameters, including masses, positions, and velocities for both
            the known and unknown bodies

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

        - self.num_iter | integer
            Number of iterations to perform in MCMC

        - self.num_walkers | integer
            Number of walkers in MCMC

        # CLASS SET ATTRIBUTES

        - self.surrogate_model | n-dimensional array of type TrajectoryInformation
            Array containing the trajectory information for the planetary trajectories
            Array will have self.unknown_bodies indices
            Each axis will have self.surrogate_points entries

        - self.true_trajectory_information | TrajectoryInformation object
            Information corresponding to the true planet trajectories
    """

    __init__():

        """
        Sets the default parameters for the various user set attributes

        Inputs:
            - known_bodies
            - unknown_bodies
            - true_init_parameters
            - max_masses
            - eta
            - surrogate_points
            - num_iter
            - num_walkers
        Outputs:
            - Updates the self. copy of all the inputs
        """

    extract_trajectory_information():

        """
        For a given set of hidden masses, update the surrogate_model with trajectory information

        Inputs:
            - masses of hidden bodies
        Outputs:
            - updates self.surrogate_model
        """

    construct_surrogate_model():

        """
        Loop over the entire sample space of unknown_masses and configure the surrogate model at each point
        Requires simulation to be run and the information to be extracted
        Use TQDM for this function

        Inputs:
            - Mass space to be searched
        Outputs:
            - constructs self.surrogate_model
        """

    configure_true_trajectory():

        """
        Run the simulation that corresponds to the true masses of the unknown bodies and save the trajectory information

        Inputs:
            - True system information
        Outputs:
            - Updates self.true_information
        """

    interpolate_trajectory_information():

        """
        Uses interpolation of surrogate model to determine the trajectory_information of a set of masses

        Inputs:
            - Guess of unknown masses
        Outputs:
            - Interpolated TrajectoryInformation
        """

    MCMC_log_likelihood():

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

    MCMC_log_posterior():

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
