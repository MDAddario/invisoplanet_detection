import numpy as np
import json

# define classes


# a single point mass in space, with an associated position and velocity
class Body:
    # units involved
    m_sun = 1.989e30  # kg
    AU = 1.495979e11  # m
    day = 86400  # s
    # gravitational constant in these units
    G = 6.67408e-11 * m_sun * day ** 2 / AU ** 3

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

        # change the origin of the coordinate system

    def recenter(self, rnew):
        self.pos = self.pos - rnew.pos
        self.vel = self.vel - rnew.vel
        return

    # find the pair gravitational force acting on this body from a single other body
    def pairforce(self, body2):
        tol = 1e-6

        F = Body.G * self.mass * body2.mass * (self.pos - body2.pos)
        F = F / (self.scalardistance(body2) ** 3 + tol)
        return F

    # find the total force acting on this body from all other bodies (EXCLUDING those at the same position)
    def totalforce(self, bodies):
        F = np.sum([self.pairforce(body) for body in bodies if np.any(body.pos != self.pos)], axis=0)
        return F


class PhaseSpace:

    # construct
    def __init__(self, bodies, t):

        # store the instance attributes
        # bodies works just like any other list - it can be indexed from zero
        self.bodies = bodies
        self.time = t

    # read in all positions, velocities, and masses into single arrays for easy access
    def arrayvals(self):

        pos_arr = []
        vel_arr = []
        mass_arr = []

        for body in self.bodies:
            pos = [body.pos[0], body.pos[1]]
            vel = [body.vel[0], body.vel[1]]
            mass = [body.mass]
            pos_arr.append(pos)
            vel_arr.append(vel)
            mass_arr.append(mass)

        return np.array(pos_arr), np.array(vel_arr), np.array(mass_arr)


    def polararrayvals(self):

        x, v, m = self.arrayvals()

        r = np.sqrt(np.sum(x**2, axis=1))
        phi = np.arctan2(x[:, 1], x[:, 0])

        vr = np.sqrt(np.sum(v**2, axis=1))
        vphi = np.arctan2(v[:, 1], v[:, 0])

        pol_pos = []
        pol_vel = []

        for i in range(len(m)):
            pol_pos.append([r[i], phi[i]])
            pol_vel.append([vr[i], vphi[i]])

        return np.array(pol_pos), np.array(pol_vel), m


    # find the center of mass of the system at the given instant in time
    # store it as a body object with mass the total mass of the system
    def findCoM(self):

        x, v, m = self.arrayvals()

        m_tot = np.sum(m)
        com_pos = np.sum(m * x, axis=0) / m_tot
        com_vel = np.sum(m * v, axis=0) / m_tot

        return Body(com_pos, com_vel, m_tot)

    # redefine all pos and vel coordinates for all bodies in the system in terms of the
    # center of mass
    def CoMrecenter(self):

        # first, find the CoM of the system in relation to the current origin
        com_f = self.findCoM()

        for body in self.bodies:
            body.recenter(com_f)

        return

    # print the pos and time information to an external file
    # tfile and posfile both have to be file pointers to write files
    def psprint(self, posfile):

        x, v, m = self.arrayvals()

        np.savetxt(posfile, x)
        posfile.write("\n")

        return


# set up the system to begin the iterations
# both icfile and filenames should be str objects containing the names of the associated files
def initialize(icfile, filename):
    # read ic information from the file into a list of body objects
    with open(icfile, "r") as file:
        ics = json.load(file)

    body_list = []

    for body in ics["bodies"]:
        m = body["mass"]
        pos = np.array([body["init_pos"]["x"], body["init_pos"]["y"]])
        vel = np.array([body["init_vel"]["x"], body["init_vel"]["y"]])

        body_list.extend([Body(pos, vel, m)])

    # define an initial phase space object
    init_space = PhaseSpace(body_list, 0)

    # recenter the phase space around its CoM
    init_space.CoMrecenter()

    # define output files and print the initial positions and times to them
    x_outfile = open(filename, "w")

    init_space.psprint(x_outfile)

    return init_space, x_outfile


# progress the simulation by a single time interval dt
def iterate(space_i, dt):
    bodies_f = []

    # calculate the acceleration of each of the bodies in space_i from the grav force of all other bodies:
    for body in space_i.bodies:
        a = - body.totalforce(space_i.bodies) / body.mass

        xf = body.pos + body.vel * dt + 0.5 * a * dt ** 2
        vf = body.vel + a * dt

        bodies_f.extend([Body(xf, vf, body.mass)])

    # time at the end of the iteration
    tf = space_i.time + dt

    return PhaseSpace(bodies_f, tf)


# initialize the simulation from the icfile and run it n_iter times, progressing by time interval
# dt each time
def simulate(icfile, Niter, dt, filename):
    # call initialize to prepare the simulation
    space_i, outfile = initialize(icfile, filename)

    # loop through iterate Niter times, each time progressing by a timestep dt and printing the results after
    # each step
    for i in range(Niter):
        space_i = iterate(space_i, dt)
        space_i.psprint(outfile)

    # close the files
    outfile.close()

    # return the final phasespace object
    return space_i
