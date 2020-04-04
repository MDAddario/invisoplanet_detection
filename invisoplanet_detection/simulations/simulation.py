import numpy as np

# define classes


class Body:
    # units involved
    m_earth = 5.972e24  # kg
    AU = 1.495979e11  # m
    # gravitational constant in these units
    G = 6.67408e-11 * AU ** 3 / m_earth

    # construct
    def __init__(self, q, v, m):
        # store the instance attributes (q and v should be vectors stored as np arrays)
        self.pos = q
        self.vel = v
        self.mass = m

    def pos(self):
        return self.pos

    # find the magnitude of the separation between two bodies
    def scalardistance(self, body2):
        diff = self.pos - body2.pos

        return np.sqrt(np.sum(diff ** 2))

    # find the pair gravitational force acting on this body from a single other body
    def pairforce(self, body2):
        F = Body.G * self.mass * body2.mass * (self.pos - body2.pos)
        F = F / self.scalardistance(body2) ** 3
        return F

    # find the total force acting on this body from all other bodies
    def totalforce(self):
        pass


class PhaseSpace:

    # construct
    def __init__(self, bodies, t):
        # store the instance attributes
        # bodies works just like any other list - it can be indexed from zero
        self.bodies = bodies
        self.time = t

        # read in all positions, velocities, and masses into single arrays for easy access
        pos_arr = []
        vel_arr = []
        mass_arr = []

        # ***
        for body in bodies:
            pos = [body.pos[0], body.pos[1]]
            vel = [body.vel[0], body.vel[1]]
            mass = [body.mass]
            pos_arr.append(pos)
            vel_arr.append(vel)
            mass_arr.append(mass)

        self.pos = np.array(pos_arr)
        self.vel = np.array(vel_arr)
        self.mass = np.array(mass_arr)

        # find the center of mass of the system at the given instant in time
        def findCoM(self):
            return np.sum(self.mass * self.pos, axis=0) / np.sum(self.mass)

    # redefine all pos and vel coordinates for all bodies in the system in terms of the
    # center of mass
    def CoMrecenter(self):
        pass

    # calculate the matrix of all pair forces for the system at this instant
    def forcematrix(self):
        pass

    # print the pos and time information to an external file
    def printphasespace(self):
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