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

    # find the magnitude of the separation between two bodies
    def scalardistance(self, body2):
        diff = self.pos - body2.pos
        return np.sqrt(np.sum(diff ** 2))

    # find the pair gravitational force acting on this body from a single other body
    def pairforce(self, body2):
        F = Body.G * self.mass * body2.mass * (self.pos - body2.pos)
        F = F / self.scalardistance(body2) ** 3
        return F

    # find the total force acting on this body from all other bodies (EXCLUDING those at the same position)
    def totalforce(self, bodies):
        F = np.sum([self.pairforce(body) for body in bodies if np.all(body.pos != self.pos)], axis=0)
        return F


class PhaseSpace:

    # ***FLIP THIS? READ IN THE WHOLE PHASE SPACE FIRST AND THEN STORE THE INFO AS INDIVIDUAL BODIES WHEN
    # NEEDED

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
    # store it as a body object with mass the total mass of the system
    def findCoM(self):

        m_tot = np.sum(self.mass)
        com_pos = np.sum(self.mass * self.pos, axis=0) / m_tot
        com_vel = np.sum(self.mass * self.vel, axis=0) / m_tot
        return Body(com_pos, com_vel, m_tot)

    # redefine all pos and vel coordinates for all bodies in the system in terms of the
    # center of mass
    def CoMrecenter(self):

        # first, find the CoM of the system in relation to the current origin
        com_f = self.findCoM()

        # adjust the position and velocity of all bodies in the space
        self.pos = self.pos - com_f.pos
        self.vel = self.vel - com_f.vel
        return

    # calculate the matrix of all pair forces for the system at this instant
    def forces(self):

        F_arr = []

        for body in self.bodies:
            F_arr.append(body.totalforce(self.bodies))

        return np.array(F_arr)

    # print the pos and time information to an external file
    # tfile and posfile both have to be file pointers to write files
    def psprint(self, tfile, posfile):

        tfile.write(str(self.time))
        tfile.write("\n")

        np.savetxt(posfile, self.pos)
        posfile.write("\n")

        return


# set up the system to begin the iterations
def initialize_simulation(icfile):
    # read ic information from the file into a list of body objects
    with open(icfile, "r") as file:
        ics = np.genfromtxt(icfile)

    body_list = []

    for i in np.arange(len(ics), step=5):
        m = ics[i]
        pos = np.array([ics[i + 1], ics[i + 2]])
        vel = np.array([ics[i + 3], ics[i + 4]])

        body_list.extend([Body(pos, vel, m)])

    # define an initial phase space object
    init_space = PhaseSpace(body_list, 0)

    # recenter the phase space around its CoM
    init_space.CoMrecenter()

    # define output files and print the initial positions and times to them
    t_outfile = open("testt.txt", "w")
    x_outfile = open("testx.txt", "w")

    init_space.psprint(t_outfile, x_outfile)

    return init_space


def iterate():

    pass


def simulate():

    # CLOSE files

    pass

    #