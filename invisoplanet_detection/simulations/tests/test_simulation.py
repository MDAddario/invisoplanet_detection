import unittest
import nose.tools as nt
import numpy as np
from invisoplanet_detection.simulations.simulation import *
import json
import os


class TestSimulation(unittest.TestCase):

    def setUp(self):
        # setup ***
        data = {}
        data['bodies'] = []
        data['bodies'].append({
            'mass': 1,
            "init_pos": {
                "x": 1,
                "y": 1
            },
            "init_vel": {
                "x": -1,
                "y": -1
            }
        })
        data['bodies'].append({
            'mass': 1,
            "init_pos": {
                "x": -1,
                "y": -1
            },
            "init_vel": {
                "x": -1,
                "y": -1
            }
        })

        testfile = 'testdata.json'
        testout = 'testoutput.txt'

        with open(testfile, 'w') as f:
            json.dump(data, f)

        return testfile, testout


    # total energy of a PhaseSpace object
    def totalenergy(self, space):
        x, v, m = space.arrayvals()
        kinetic = np.sum(0.5 * v ** 2 * m)

        potential = - space.bodies[0].G * np.product(m) / space.bodies[0].scalardistance(space.bodies[1])
        return kinetic + potential


    # total angular momentum of a PhaseSpace object
    def angularmomentum(self, space):
        x, v, m = space.arrayvals()
        return np.cross(x, m * v)

    @nt.with_setup(setUp)
    def test_initialize(self):

        testfile, testout = self.setUp()

        space, outfile = initialize(testfile, testout)

        nt.assert_equal(len(space.bodies), 2)

        x, v, m = space.arrayvals()

        # check for NaNs
        nt.assert_equal(np.any(np.isnan(x)), False)
        nt.assert_equal(np.any(np.isnan(v)), False)
        nt.assert_equal(np.any(np.isnan(m)), False)

        # check the outputs are the expected shape
        nt.assert_equal(x.shape, (2, 2))
        nt.assert_equal(v.shape, (2, 2))
        nt.assert_equal(m.shape, (2, 1))

        # check the center of mass recentering works for these known values
        com = space.findCoM()
        for i in range(2):
            nt.assert_almost_equal(com.pos[i], 0)
            nt.assert_almost_equal(com.vel[i], 0)
        nt.assert_almost_equal(com.mass, 2)

        # check a file is created
        nt.assert_equal(True, os.path.isfile('testoutput.txt'))


    @nt.with_setup(setUp)
    def test_iterate(self):

        testfile, testout = self.setUp()

        space_i, outfile = initialize(testfile, testout)
        space_f = iterate(space_i, 1)

        xi, vi, mi = space_i.arrayvals()
        xf, vf, mf = space_f.arrayvals()

        # check for NaNs
        nt.assert_equal(np.any(np.isnan(xf)), False)
        nt.assert_equal(np.any(np.isnan(vf)), False)
        nt.assert_equal(np.any(np.isnan(mf)), False)

        # make sure none of the bodies are stacked on top of each other
        nt.assert_greater(space_f.bodies[0].scalardistance(space_f.bodies[1]), 1e-6)

        # make sure none of the masses have changed
        for i in range(2):
            nt.assert_equal(mi[i], mf[i])

        # check that the CoM has not moved significantly
        com_i = space_i.findCoM()
        com_f = space_f.findCoM()
        for i in range(2):
            nt.assert_almost_equal(com_i.pos[i], com_f.pos[i])
            nt.assert_almost_equal(com_i.vel[i], com_f.vel[i])
        nt.assert_almost_equal(com_i.mass, com_f.mass)

        # check the total energy before and after iterating, make sure it hasn't changed significantly
        # *** right now only works for a 2-body system
        nt.assert_almost_equal(self.totalenergy(space_i), self.totalenergy(space_f))

        # calculate angular momentum before and after iterating, make sure it hasn't changed significantly

        Li = self.angularmomentum(space_i)
        Lf = self.angularmomentum(space_f)
        for i in range(2):
            nt.assert_almost_equal(Li[i], Lf[i])


    @nt.with_setup(setUp)
    def test_simulate(self):

        testfile, testout = self.setUp()

        space_i, outfile = initialize(testfile, testout)
        space_f = simulate(testfile, 100, 0.5, testout)

        xi, vi, mi = space_i.arrayvals()
        xf, vf, mf = space_f.arrayvals()

        # check for NaNs
        nt.assert_equal(np.any(np.isnan(xf)), False)
        nt.assert_equal(np.any(np.isnan(vf)), False)
        nt.assert_equal(np.any(np.isnan(mf)), False)

        # make sure none of the bodies are stacked on top of each other
        nt.assert_greater(space_f.bodies[0].scalardistance(space_f.bodies[1]), 1e-6)

        # check a file is created
        nt.assert_equal(True, os.path.isfile('testoutput.txt'))

        # make sure none of the masses have changed
        for i in range(2):
            nt.assert_equal(mi[i], mf[i])

        # make sure the CoM has not moved
        com_i = space_i.findCoM()
        com_f = space_f.findCoM()

        for i in range(2):
            nt.assert_equal(com_i.pos[i], com_f.pos[i])
            nt.assert_equal(com_i.vel[i], com_f.vel[i])

        # calculate total energy before and after iterating, make sure it hasn't changed significantly
        # check the total energy before and after iterating, make sure it hasn't changed significantly
        # *** right now only works for a 2-body system
        nt.assert_almost_equal(self.totalenergy(space_i), self.totalenergy(space_f))

        # calculate total angular momentum before and after iterating, make sure it hasn't changed significantly
        Li = self.angularmomentum(space_i)
        Lf = self.angularmomentum(space_f)
        for i in range(2):
            nt.assert_almost_equal(Li[i], Lf[i])

        # kepler problem as a overall test of the physics is included as an actual function in
        # simulation.py - the function's name is kepler_check



if __name__ == '__main__':
    unittest.main()
