import numpy as np

# define classes


class Vector:

    # construct
    def __init__(self, x, y):

        # store the instance attributes
        self.x = x
        self.y = y
        self.vector = np.array((x, y))

    # return the scalar distance between the vector and another
    def scalardistance(self, vec2):

        diff = self.vector - vec2.vector
        return np.sqrt(np.sum(diff ** 2))

    # change the origin of the coordinate system in which the vector is defined
    # newcenter should be a member of the vector class - vector pointing to origin of the
    # new coordinate system from the origin of the old system
    def recenter(self, newcenter):

        return self.vector - newcenter.vector





class Body:

    # units involved
    m_earth = 5.972e24 # kg
    AU = 1.495979e11 # m
    # gravitational constant in these units
    G = 6.67408e-11 * AU ** 3 / m_earth

    # construct
    def __init__(self, q, v, m):

        # store the instance attributes (q and v should be vectors)
        self.position = q
        self.velocity = v
        self.mass = m


    # find the pair gravitational force acting on this body from a single other body
    def pairforce(self, b2):

        F = Body.G * self.mass * b2.mass * (self.position.vector - b2.position.vector)
        F = F / self.position.scalardistance(b2.position) ** 3
        return F

    # find the total force acting on this body from all other bodies
    def totalforce(self):
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