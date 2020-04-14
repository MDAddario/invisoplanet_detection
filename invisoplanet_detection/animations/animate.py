import pyglet
from pyglet.gl import *
from pyglet.window import key
from pyglet.window import mouse
import numpy as np

# Setup window
config = Config(sample_buffers=1, samples=4, depth_size=16, double_buffer=True)
window = pyglet.window.Window(resizable=True, config=config)

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


# Create vertex list from vertices, normals, indices and all that good stuff
def create_vertex_list(batch, vertices, normals, indices, color, render_mode=GL_TRIANGLES,
						vertex_format="static", normal_format="static"):

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

	# Configure formats
	vertex_format = "v3f/" + vertex_format
	normal_format = "n3f/" + normal_format

	# Create a material and group for the model
	ambient = [0.5, 0.0, 0.3, 1.0]
	specular = [1.0, 1.0, 1.0, 1.0]
	emission = [0.0, 0.0, 0.0, 1.0]
	shininess = 50
	material = pyglet.model.Material("", diffuse, ambient, specular, emission, shininess)
	group = pyglet.model.MaterialGroup(material=material)

	return batch.add_indexed(len(vertices)//3,
							render_mode,
							group,
							indices,
							(vertex_format, vertices),
							(normal_format, normals))


# Create a planet as a sphere
def planet_creator(position_data, radius, color, num=3):

	# In spherical coordinates
	def make_vector(theta, phi, radius=1):

		x = (np.sin(theta) * np.cos(phi)) * radius
		y = (np.sin(theta) * np.sin(phi)) * radius
		z = (np.cos(theta)) * radius

		return [x, y, z]

	# Create the vertex and normal arrays
	vertices = []
	normals = []

	# Span all angles
	for theta in np.linspace(0, np.pi, num=num, endpoint=True):

		# North and south poles
		if theta == 0 or theta == np.pi:
			vertices.extend(make_vector(theta, 0, radius))
			normals.extend(make_vector(theta, 0))

		else:
			for phi in np.linspace(0, 2*np.pi, num=num, endpoint=False):
				vertices.extend(make_vector(theta, phi, radius))
				normals.extend(make_vector(theta, phi))

	# Count how many vertices we have
	N = len(vertices) // 3

	# Create a list of triangle indices
	indices = []

	# Span all angles
	for t_i in np.arange(num-1):

		# North and south poles
		if t_i == 0:
			for p_i in np.arange(num):
				if p_i == num - 1:
					indices.extend([0, p_i + 1, 1])
				else:
					indices.extend([0, p_i + 1, p_i + 2])

		elif t_i == num-2:
			for p_i in np.arange(num):
				if p_i == num - 1:
					indices.extend([N-1, N-1-p_i-1, N-2])
				else:
					indices.extend([N-1, N-1-p_i-1, N-1-p_i-2])

		else:
			# Determine what vertex we are at
			M = 1 + num * (t_i - 1)

			for p_i in np.arange(num):

				# Make nice triangles
				if p_i == num - 1:
					indices.extend([M + 1 + p_i - num, M + p_i, M + num + p_i])
					indices.extend([M + 1 + p_i - num, M + num + p_i, M + num + p_i + 1 - num])
				else:
					indices.extend([M + 1 + p_i, M + p_i, M + num + p_i])
					indices.extend([M + 1 + p_i, M + num + p_i, M + num + p_i + 1])

	vertex_list = create_vertex_list(batch, vertices, normals, indices, color, GL_TRIANGLES, "static", "static")

	# Create object
	planet = Planet(vertex_list, position_data, color)
	object_list.append(planet)

	return planet


# Planet object that holds position data and mass data
class Planet:

	# Constructor
	def __init__(self, vertex_list, position_data, color):
		# Store the instance attributes
		self.vertex_list = vertex_list
		self.position_data = position_data
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
		self._trace_trajectory()

	# Destructor
	def __del__(self):
		self.vertex_list.delete()

	# Plot n little boxes along the way of the trajectory
	def _trace_trajectory(self, n=50, mini_mass=1e-10):
		pass

	# Determine model center from the vertices
	def _compute_center(self):
		return (np.max(self.vertices, axis=0) + np.min(self.vertices, axis=0)) / 2

	# Update the real vertices from the object's local copy
	def _update_vertices(self):
		self.vertex_list.vertices = np.copy(np.ravel(self.vertices))

	# Scale both the real, local vertices, and the ecb dimensions
	def _scale_vertices(self, scaling):
		self.vertices *= scaling
		self._update_vertices()

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
dy = 2
dz = -20

# Camera translation speed
cam_rate = 0.7

# Keep track of objects
object_list = []
