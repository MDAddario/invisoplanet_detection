import numpy as np
import json

# define classes


# a single point mass in space, with an associated position and velocity
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

        # change the origin of the coordinate system

    def recenter(self, rnew):
        self.pos = self.pos - rnew.pos
        self.vel = self.vel - rnew.vel
        return

    # find the pair gravitational force acting on this body from a single other body
    def pairforce(self, body2):
        F = Body.G * self.mass * body2.mass * (self.pos - body2.pos)
        F = F / self.scalardistance(body2) ** 3
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
    def psprint(self, tfile, posfile):

        x, v, m = self.arrayvals()

        tfile.write(str(self.time))
        tfile.write("\n")

        np.savetxt(posfile, x)
        posfile.write("\n")

        return


# set up the system to begin the iterations
# both icfile and filenames should be str objects containing the names of the associated files,
# filenames should be of the format (tfilename, xfilename)
def initialize(icfile, filenames):
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
    t_outfile = open(filenames[0], "w")
    x_outfile = open(filenames[1], "w")

    init_space.psprint(t_outfile, x_outfile)

    return init_space, [t_outfile, x_outfile]


# progress the simulation by a single time interval dt
def iterate(space_i, dt):

    bodies_f = []

    # calculate the acceleration of each of the bodies in space_i from the grav force of all other bodies:
    for body in space_i.bodies:
        a = body.totalforce(space_i.bodies) / body.mass

        xf = body.pos + body.vel * dt + 0.5 * a * dt ** 2
        vf = body.vel + a * dt

        bodies_f.extend([Body(xf, vf, body.mass)])

    # time at the end of the iteration
    tf = space_i.time + dt

    return PhaseSpace(bodies_f, tf)


# initialize the simulation from the icfile and run it n_iter times, progressing by time interval
# dt each time
def simulate(icfile, n_iter, dt, filenames):
    # call initialize to prepare the simulation
    space_i, outfiles = initialize(icfile, filenames)

    # loop through iterate Niter times, each time progressing by a timestep dt and printing the results after
    # each step
    for i in range(n_iter):
        space_i = iterate(space_i, dt)
        space_i.psprint(*outfiles)

    # close the files
    outfiles[0].close()
    outfiles[1].close()

    # return the final phasespace object
    return space_i
