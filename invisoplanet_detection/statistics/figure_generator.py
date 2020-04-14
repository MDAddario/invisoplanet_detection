from invisoplanet_detection.statistics.stats import *


if __name__ == "__main__":
	"""
	Generate figures for the stats portion of the manuscript
	"""
	# 1D example
	one_dimension = Likelihood(known_bodies=2, unknown_bodies=1,
							parameters_filename="tests/ic_files/D_sun_jup_sat_2_1_1.json",
							max_masses=[2.2 * 0.000285802], surrogate_points=7, num_iterations=200)
	one_dimension.set_eta(1)
	one_dimension.plot_posterior("figures/one_dimensional.pdf", num=100, floor=None)
