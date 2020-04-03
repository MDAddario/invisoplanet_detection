import pyglet
from pyglet.gl import *
from pyglet.window import key
from pyglet.window import mouse
import numpy as np
from scipy.spatial.transform import Rotation as R

# Setup window
try:
	# Try and create a window with multisampling (antialiasing)
	config = Config(sample_buffers=1, samples=4, depth_size=16, double_buffer=True)
	window = pyglet.window.Window(resizable=True, config=config)
except pyglet.window.NoSuchConfigException:
	# Fall back to no multisampling for old hardware
	window = pyglet.window.Window(resizable=True)

# Set window projection to 3D
window.projection = pyglet.window.Projection3D()


# Automatically called each frame
@window.event
def on_draw():
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
	glLoadIdentity()
	glTranslatef(dx, dy, dz)
	glRotatef(rz, 0, 0, 1)
	glRotatef(ry, 0, 1, 0)
	glRotatef(rx, 1, 0, 0)
	batch.draw()


# One time setup
def setup():
	# Background color and rendering optimizations
	glClearColor(0.1, 0.1, 0.1, 1)
	glColor3f(1, 0, 0)
	glEnable(GL_DEPTH_TEST)
	glEnable(GL_CULL_FACE)

	# Simple light setup
	glEnable(GL_LIGHTING)
	glEnable(GL_LIGHT0)
	glEnable(GL_LIGHT1)


# Create a planet as a box
def planet_creator(batch, position_data, mass, color='grey'):
	# Create the vertex and normal arrays
	vertices = []
	normals = []

	# Have some standard normals
	x_normal = np.array([0.98, 0.1, 0.1])
	y_normal = np.array([0.1, 0.98, 0.1])
	z_normal = np.array([0.1, 0.1, 0.98])

	# Front face
	vertices.extend(np.array([0, 0, 1]))
	vertices.extend(np.array([1, 0, 1]))
	vertices.extend(np.array([1, 1, 1]))
	vertices.extend(np.array([0, 1, 1]))

	for i in range(4):
		normals.extend(z_normal)

	# Back face
	vertices.extend(np.array([0, 0, 0]))
	vertices.extend(np.array([0, 1, 0]))
	vertices.extend(np.array([1, 1, 0]))
	vertices.extend(np.array([1, 0, 0]))

	for i in range(4):
		normals.extend(-z_normal)

	# Top face
	vertices.extend(np.array([0, 1, 0]))
	vertices.extend(np.array([0, 1, 1]))
	vertices.extend(np.array([1, 1, 1]))
	vertices.extend(np.array([1, 1, 0]))

	for i in range(4):
		normals.extend(y_normal)

	# Bot face
	vertices.extend(np.array([0, 0, 0]))
	vertices.extend(np.array([1, 0, 0]))
	vertices.extend(np.array([1, 0, 1]))
	vertices.extend(np.array([0, 0, 1]))

	for i in range(4):
		normals.extend(-y_normal)

	# Right face
	vertices.extend(np.array([1, 0, 0]))
	vertices.extend(np.array([1, 1, 0]))
	vertices.extend(np.array([1, 1, 1]))
	vertices.extend(np.array([1, 0, 1]))

	for i in range(4):
		normals.extend(x_normal)

	# Left face
	vertices.extend(np.array([0, 0, 0]))
	vertices.extend(np.array([0, 0, 1]))
	vertices.extend(np.array([0, 1, 1]))
	vertices.extend(np.array([0, 1, 0]))

	for i in range(4):
		normals.extend(-x_normal)

	# Do the indices too
	indices = np.arange(len(vertices))

	# Select color
	if color == 'red':
		diffuse = [1.0, 0.0, 0.0, 1.0]
	elif color == 'green':
		diffuse = [0.0, 1.0, 0.0, 1.0]
	elif color == 'blue':
		diffuse = [0.0, 0.0, 1.0, 1.0]
	elif color == 'yellow':
		diffuse = [1.0, 1.0, 0.0, 1.0]
	elif color == 'purple':
		diffuse = [0.5, 0.0, 0.3, 1.0]
	else:
		diffuse = [0.0, 0.0, 0.0, 1.0]

	# Create a material and group for the model
	ambient = [0.5, 0.0, 0.3, 1.0]
	specular = [1.0, 1.0, 1.0, 1.0]
	emission = [0.0, 0.0, 0.0, 1.0]
	shininess = 50
	material = pyglet.model.Material("", diffuse, ambient, specular, emission, shininess)
	group = pyglet.model.MaterialGroup(material=material)

	# Create vertex list
	vertex_list = batch.add_indexed(len(vertices) // 3, GL_QUADS, group, indices, ("v3f/dynamic", vertices), ("n3f/dynamic", normals))

	# Create object
	return Planet(batch, vertex_list, position_data, mass, color)


# Planet object that holds position data and mass data
class Planet:

	mass_ratio = 1
	spin_ratio = 100

	# Constructor
	def __init__(self, batch, vertex_list, position_data, mass, color):
		# Store the instance attributes
		self.batch = batch
		self.vertex_list = vertex_list
		self.position_data = position_data
		self.mass = mass
		self.color = color

		# Extract a deepcopy of vertices formatted as Nx3 array
		self.vertices = []
		self.vertices.extend(self.vertex_list.vertices)
		self.vertices = np.reshape(self.vertices, (-1, 3))

		# Extract a deepcopy of normals formatted as Nx3 array
		self.normals = []
		self.normals.extend(self.vertex_list.normals)
		self.normals = np.reshape(self.normals, (-1, 3))

		# Compute transformation center
		self.center = self._compute_center()
		self.frame = 0

		# Initialize position and size
		self.update(dt=0)

		# Trace the trajectory
		if self.mass > 0:
			self.rescale(self.mass * self.mass_ratio)
			self._trace_trajectory()

	# Destructor
	def __del__(self):
		self.vertex_list.delete()

	# Plot n little boxes along the way of the trajectory
	def _trace_trajectory(self, n=30, mini_mass=0.3, spin_ratio=-3):

		# Determine the indices of the locations
		indices = np.linspace(0, len(self.position_data), num=n, endpoint=False)
		self.trajectory_points = []

		# At each position, make a baby cube
		for index in np.rint(indices).astype(int):
			position = self.position_data[index]
			self.trajectory_points.append(planet_creator(batch, [position], -1, self.color))
			self.trajectory_points[-1].rescale(mini_mass)
			self.trajectory_points[-1].spin_ratio *= spin_ratio

	# Determine model center from the vertices
	def _compute_center(self):
		return (np.max(self.vertices, axis=0) + np.min(self.vertices, axis=0)) / 2

	# Update the real vertices from the object's local copy
	def _update_vertices(self):
		self.vertex_list.vertices = np.copy(np.ravel(self.vertices))

	# Update the real normals from the object's local copy
	def _update_normals(self):
		self.vertex_list.normals = np.copy(np.ravel(self.normals))

	# Scale both the real, local vertices, and the ecb dimensions
	def _scale_vertices(self, scaling):
		self.vertices *= scaling
		self._update_vertices()

	# Rotate both the real, local vertices, the ecb dimensions, and the normals
	def _rotate_vertices(self, rotation):
		self.vertices = rotation.apply(self.vertices)
		self.normals = rotation.apply(self.normals)
		self._update_vertices()
		self._update_normals()

	# Translate both the real, local vertices, and the transformation center
	def _translate_vertices(self, translation):
		self.vertices += translation
		self._update_vertices()
		self.center += translation

	# Rescale total object size about center
	def rescale(self, new_scale):
		# Determine maximal distance of vertices from center
		distances = np.sqrt(np.sum(np.square(self.vertices - self.center), axis=1))
		max_dist = np.max(distances)
		scaling = new_scale / max_dist

		# Keep track of old center position
		old_center = np.copy(self.center)

		# Slide to origin, rescale, slide back
		self._translate_vertices(-old_center)
		self._scale_vertices(scaling)
		self._translate_vertices(old_center)

	# Rotate object about center
	def rotate_degrees(self, axis, angle):

		# Build rotation object
		r = R.from_euler(axis, angle, degrees=True)

		# Keep track of old center position
		old_center = np.copy(self.center)

		# Slide to origin, rescale, slide back
		self._translate_vertices(-old_center)
		self._rotate_vertices(r)
		self._translate_vertices(old_center)

	# Set the position of the model
	def set_position(self, position):
		self._translate_vertices(position - self.center)

	# Update the model every frame
	def update(self, dt):
		# Update position
		self.set_position(self.position_data[self.frame])
		self.frame += 1

		# Wrap frame counter
		if self.frame == len(self.position_data):
			self.frame = 0

		# Rotate body
		self.rotate_degrees('z', dt * self.spin_ratio)


# Take care of camera movement
def translate_camera_x(dt, rate):
	global dx
	dx += dt * rate * dz


def translate_camera_y(dt, rate):
	global dy
	dy += dt * rate * dz


@window.event
def on_key_press(symbol, modifiers):
	# Translate the camera
	if symbol == key.UP:
		if keys[key.DOWN]:
			pyglet.clock.unschedule(translate_camera_y)
		pyglet.clock.schedule(translate_camera_y, rate=cam_rate)

	elif symbol == key.DOWN:
		if keys[key.UP]:
			pyglet.clock.unschedule(translate_camera_y)
		pyglet.clock.schedule(translate_camera_y, rate=-cam_rate)

	elif symbol == key.RIGHT:
		if keys[key.LEFT]:
			pyglet.clock.unschedule(translate_camera_x)
		pyglet.clock.schedule(translate_camera_x, rate=cam_rate)

	elif symbol == key.LEFT:
		if keys[key.RIGHT]:
			pyglet.clock.unschedule(translate_camera_x)
		pyglet.clock.schedule(translate_camera_x, rate=-cam_rate)


@window.event
def on_key_release(symbol, modifiers):
	# Reset the camera
	if symbol == key.UP:
		if not keys[key.DOWN]:
			pyglet.clock.unschedule(translate_camera_y)

	elif symbol == key.DOWN:
		if not keys[key.UP]:
			pyglet.clock.unschedule(translate_camera_y)

	elif symbol == key.RIGHT:
		if not keys[key.LEFT]:
			pyglet.clock.unschedule(translate_camera_x)

	elif symbol == key.LEFT:
		if not keys[key.RIGHT]:
			pyglet.clock.unschedule(translate_camera_x)


# Rotate the window
@window.event
def on_mouse_drag(x, y, dx, dy, buttons, modifiers):
	global rx, ry, rz

	if buttons & mouse.LEFT:

		if modifiers & key.MOD_SHIFT:
			rz += dy
			rz %= 360

		else:
			rx += -dy
			ry += dx
			rx %= 360
			ry %= 360


# Zoom the window
@window.event
def on_mouse_scroll(x, y, scroll_x, scroll_y):
	global dz
	dz -= scroll_y * dz * 0.25
	dz = min(dz, 0)


# Update every frame
def update(dt):
	# Update all mobile objects
	for obj in object_list:
		obj.update(dt)


# The main attraction
if __name__ == "__main__":

	# Setup window and the only batch
	setup()
	batch = pyglet.graphics.Batch()

	# Add keystate handler
	keys = key.KeyStateHandler()
	window.push_handlers(keys)

	# Schedule the ever-important update function
	pyglet.clock.schedule(update)

	# Initialize global variables for camera rotation and translation
	ry = rz = dx = 0
	rx = -60
	dy = -2
	dz = -40

	# Camera translation speed
	cam_rate = 0.7

	# Create position data
	parameter = np.linspace(0, 1, num=120)
	p1_x_pos = 10 * np.cos(2 * np.pi * parameter - np.pi/4) - 5
	p1_y_pos = 10 * np.sin(2 * np.pi * parameter - np.pi/4) - 5
	p1_z_pos = np.zeros(len(parameter))

	p1_pos = []
	for x, y, z in zip(p1_x_pos, p1_y_pos, p1_z_pos):
		p1_pos.append([x, y, z])

	p2_x_pos = -10 * np.sin(2 * np.pi * parameter + np.pi) + 5
	p2_y_pos = -10 * np.cos(2 * np.pi * parameter + np.pi) + 5
	p2_z_pos = np.zeros(len(parameter))

	p2_pos = []
	for x, y, z in zip(p2_x_pos, p2_y_pos, p2_z_pos):
		p2_pos.append([x, y, z])

	# Set masses
	p1_mass = 4
	p2_mass = p1_mass

	# Create planet models
	p1 = planet_creator(batch, p1_pos, p1_mass, "red")
	p2 = planet_creator(batch, p2_pos, p2_mass, "blue")

	# Keep track of all the active models
	object_list = []
	object_list.extend([p1, p2])

	# Animate the trajectories
	#object_list.extend(p1.trajectory_points)
	#object_list.extend(p2.trajectory_points)

	# Run the animation!
	pyglet.app.run()
