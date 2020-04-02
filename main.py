# Import the entire repository
import invisoplanet_detection as invis


if __name__ == "__main__":

	# Example of how to call functions from the subpackages
	invis.statistics.statistics_print()
	invis.simulations.simulations_print()

	# If the names get too long, can do this
	stats = invis.statistics
	simuls = invis.simulations

	stats.statistics_print()
	simuls.simulations_print()
