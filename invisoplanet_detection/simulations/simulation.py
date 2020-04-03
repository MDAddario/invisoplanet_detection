import numpy as np

# define classes


class Vector:

    # construct
    def __init__(self, x, y):

        # store the instance attributes
        self.x = x
        self.y = y
        self.vector = np.array((x,y))

    # change the origin of the coordinate system in which the vector is defined
    # newcenter should be a member of the vector class
    def recenter(self, newcenter):
        pass

    # return the scalar distance between the vector and another
    def scalardistance(self, v2):
        diff = self.vector - v2.vector
        return np.sqrt(np.sum(diff ** 2))






class Body:

    # construct
    def __init__(self, q, v, m):

        # store the instance attributes
        self.position = q
        self.velocity = v
        self.mass = m

    # find the pair gravitational force acting on this body from a single other body
    def __pairforce__(self):
        pass

    # find the total force acting on this body from all other bodies
    def __totalforce__(self):
        pass


class PhaseSpace:

    # construct
    def __init__(self):

        # store the instance attributes
        pass

    # find the center of mass of the system at the given instant in time
    def __findCoM__(self):
        pass

    # redefine all position and velocity coordinates for all bodies in the system in terms of the
    # center of mass
    def __CoMrecenter__(self):
        pass

    # calculate the matrix of all pair forces for the system at this instant
    def __forcematrix__(self):
        pass


# set up the system to begin the iterations
def initialize_simulation(icfile):

    # read in initial conditions from an external file

    # store them in body classes

    # read in masses from external file

    # store them in body classes

    # define the initial phase space class

    # recenter all coordinates with respect to the center of mass

    pass


def iterate():

    pass


def simulate():

    pass

    #