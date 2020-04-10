import unittest
import nose.tools as nt
from invisoplanet_detection.statistics.stats import *

"""
Example uses of nosetests:

	- nt.assert_raises(ValueError, function_name, argument_0, argument_1)
	- nt.assert_almost_equal(number_one, number_two, places=3)
	- nt.assert_equal(number_one, number_two)
"""


class TestStats:

	def test_statistics(self):
		pass

		"""
		Raise value errors for:
			- Num bodies != num bodies in IC
			- Forgetting to set eta
			- Num unknown = 0, 3
			- last_n > 100, <= 0 

		For each of the following systems:
		
			- 1_0_1
			- 1_1_1
			- 2_0_1
			- 2_1_1
			
			- 1_1_2
			- 1_2_2
			- 2_1_2

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


if __name__ == '__main__':

	unittest.main()
