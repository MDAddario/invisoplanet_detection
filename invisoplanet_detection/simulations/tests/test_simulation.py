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

        # make sure information is stored in every phase space coordinate and mass
        nt.assert_equal(len(space.bodies), 2)

        x, v, m = space.arrayvals()
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

        pass

    def test_simulate(self):

        pass


if __name__ == '__main__':
    unittest.main()
