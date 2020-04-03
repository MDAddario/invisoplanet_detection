import pyglet
from pyglet.gl import *
from pyglet.window import key
from pyglet.window import mouse
import numpy as np
from scipy.spatial.transform import Rotation as R
import os

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
def planet_creator(batch, color="red"):
	# Define center and size
	size = np.array([1, 1, 1])
	center = np.array([0, 0, 0])

	# Offset parameter
	offset = center - size / 2

	# Create the vertex and normal arrays
	vertices = []
	normals = []

	# Have some standard normals
	x_normal = np.array([0.98, 0.1, 0.1])
	y_normal = np.array([0.1, 0.98, 0.1])
	z_normal = np.array([0.1, 0.1, 0.98])

	# Front face
	vertices.extend(np.array([0, 0, 1] * size + offset))
	vertices.extend(np.array([1, 0, 1] * size + offset))
	vertices.extend(np.array([1, 1, 1] * size + offset))
	vertices.extend(np.array([0, 1, 1] * size + offset))

	for i in range(4):
		normals.extend(z_normal)

	# Back face
	vertices.extend(np.array([0, 0, 0] * size + offset))
	vertices.extend(np.array([0, 1, 0] * size + offset))
	vertices.extend(np.array([1, 1, 0] * size + offset))
	vertices.extend(np.array([1, 0, 0] * size + offset))

	for i in range(4):
		normals.extend(-z_normal)

	# Top face
	vertices.extend(np.array([0, 1, 0] * size + offset))
	vertices.extend(np.array([0, 1, 1] * size + offset))
	vertices.extend(np.array([1, 1, 1] * size + offset))
	vertices.extend(np.array([1, 1, 0] * size + offset))

	for i in range(4):
		normals.extend(y_normal)

	# Bot face
	vertices.extend(np.array([0, 0, 0] * size + offset))
	vertices.extend(np.array([1, 0, 0] * size + offset))
	vertices.extend(np.array([1, 0, 1] * size + offset))
	vertices.extend(np.array([0, 0, 1] * size + offset))

	for i in range(4):
		normals.extend(-y_normal)

	# Right face
	vertices.extend(np.array([1, 0, 0] * size + offset))
	vertices.extend(np.array([1, 1, 0] * size + offset))
	vertices.extend(np.array([1, 1, 1] * size + offset))
	vertices.extend(np.array([1, 0, 1] * size + offset))

	for i in range(4):
		normals.extend(x_normal)

	# Left face
	vertices.extend(np.array([0, 0, 0] * size + offset))
	vertices.extend(np.array([0, 0, 1] * size + offset))
	vertices.extend(np.array([0, 1, 1] * size + offset))
	vertices.extend(np.array([0, 1, 0] * size + offset))

	for i in range(4):
		normals.extend(-x_normal)

	# Do the indices too
	indices = []
	for i in range(len(vertices)):
		indices.append(i)

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
	return StaticNoClipModel(vertex_list)


# Static model that does not check for collisions with environment
class StaticNoClipModel:

	# Constructor
	def __init__(self, vertex_list):
		# Store the instance attributes
		self.vertex_list = vertex_list

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

	# Destructor
	def __del__(self):
		self.vertex_list.delete()

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
		# Check axis makes sense
		if axis not in "xyz":
			raise ValueError("Rotation axis must be 'x', 'y', or 'z'.")

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
	pass
	# Update all mobile objects
	#for obj in object_list:
	#	obj.update(dt)


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
	rx = ry = rz = dx = 0
	dy = -2
	dz = -15

	# Camera translation speed
	cam_rate = 0.7

	# Create planet models
	p1 = planet_creator(batch, "red")
	p2 = planet_creator(batch, "blue")

	# Rescale and reposition the planets
	p1.rescale(14)
	p2.rescale(5)
	p1.set_position([-10, -10, -10])
	p2.set_position([5, 5, 5])

	# Keep track of all the active models
	object_list = []
	object_list.extend([p1, p2])

	# Run the animation!
	pyglet.app.run()
