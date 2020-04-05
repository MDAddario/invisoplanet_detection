import unittest
import nose.tools as nt
import numpy as np
from invisoplanet_detection.simulations.simulation import *
import json


class TestSimulation():


    def test_initialize(self):

        # setup ***
        data = {}
        data['bodies'] = []
        data['bodies'].append({
            'mass': '1',
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
            'mass': '1',
            "init_pos": {
                "x": -1,
                "y": -11
            },
            "init_vel": {
                "x": -1,
                "y": -1
            }
        })

        testfile = 'invisoplanet_detection/simulation/tests/testdata.json'
        testout = 'invisoplanet_detection/simulation/tests/testoutput.txt'

        with open(testfile, 'w') as f:
            json.dump(data, f)

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

        # make sure none of the bodies are wildly far away from all of the other ones
        distances = []
        for bodyi in space.bodies:
            for bodyj in space.bodies:
                distances.extend(bodyi.scalardistance(bodyj))
        nt.assert_less(np.all(distances), 1000)

        # check the center of mass calculation works for these known values
        com = space.findCoM()
        for i in range(2):
            nt.assert_almost_equal(com.pos[i], 0)
            nt.assert_almost_equal(com.vel[i], -2)
        nt.assert_almost_equal(com.mass, 2)

        # check that the center of mass recentering works
        space_centered = space.CoMrecenter()

        com_centered = space_centered.findCoM()
        for i in range(2):
            nt.assert_almost_equal(com.pos[i], 0)
            nt.assert_almost_equal(com.vel[i], 0)
        nt.assert_almost_equal(com.mass, 2)


    def test_iterate(self):

        # setup ***
        data = {}
        data['bodies'] = []
        data['bodies'].append({
            'mass': '1',
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
            'mass': '1',
            "init_pos": {
                "x": -1,
                "y": -11
            },
            "init_vel": {
                "x": -1,
                "y": -1
            }
        })

        testfile = 'invisoplanet_detection/simulation/tests/testdata.json'
        testout = 'invisoplanet_detection/simulation/tests/testoutput.txt'

        with open(testfile, 'w') as f:
            json.dump(data, f)

        space_i, outfile = initialize(testfile, testout)
        space_f = iterate(space_i, 1)

        xi, vi, mi = space_i.arrayvals()
        xf, vf, mf = space_f.arrayvals()

        # check for NaNs
        nt.assert_equal(np.any(np.isnan(xf)), False)
        nt.assert_equal(np.any(np.isnan(vf)), False)
        nt.assert_equal(np.any(np.isnan(mf)), False)

        # make sure none of the bodies are stacked on top of each other
        distances = []
        for bodyi in space_f.bodies:
            for bodyj in space_f.bodies:
                distances.extend(bodyi.scalardistance(bodyj))
        nt.assert_greater(np.all(distances), 0.000001)

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
        def totalenergy(space):
            x, v, m = space.arrayvals()
            kinetic = np.sum(0.5 * v**2 * m)

            potential = - space.G * np.product(m) / space.bodies[0].scalardistance(space.bodies[1])
            return kinetic + potential

        nt.assert_almost_equal(totalenergy(space_i), totalenergy(space_f))

        # calculate angular momentum before and after iterating, make sure it hasn't changed significantly
        def angularmomentum(space):
            x, v, m = space.arrayvals()
            return np.cross(x, m*v)

        nt.assert_almost_equal(angularmomentum(space_i), angularmomentum(space_f))


    def test_simulate(self):

        pass


if __name__ == '__main__':
    unittest.main()
