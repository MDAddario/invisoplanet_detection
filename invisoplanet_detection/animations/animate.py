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
	glClearColor(1, 1, 1, 1)
	glColor3f(1, 0, 0)
	glEnable(GL_DEPTH_TEST)
	glEnable(GL_CULL_FACE)

	# Enable wireframe view
	# glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)

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

	return batch.add_indexed(len(vertices) // 3,
	                         render_mode,
	                         group,
	                         indices,
	                         (vertex_format, vertices),
	                         (normal_format, normals))


# Create two floating triangles
def triangle_practice(batch):
	size = 1.5

	# Create the vertex and normal arrays
	vertices = []
	normals = []

	# Populate the vertices array
	vertices.extend([-size, 0, 0])
	vertices.extend([-size, -size / 5, 0])
	vertices.extend([-0, -0, 0])

	vertices.extend([+size, -size / 5, 0])
	vertices.extend([+size, 0, 0])
	vertices.extend([+0, +0, 0])

	# Populate the normals array
	z_normal = [0.1, 0.1, 0.98]

	for i in range(6):
		normals.extend(z_normal)

	# Create a list of triangle indices
	indices = []
	indices.extend([0, 1, 2])
	indices.extend([3, 4, 5])
	indices.extend([0, 2, 1])
	indices.extend([3, 5, 4])

	return create_vertex_list(batch, vertices, normals, indices, "red",
	                          GL_TRIANGLES, "static", "static")


# Create floating rectangles
def box_creator(batch, size, center, color, vertex_format, normal_format):
	# Convert to array objects
	size = np.asarray(size)
	center = np.asarray(center)

	# Error check
	if np.any(size <= 0):
		raise ValueError("Size values must be strictly positive")
	if size.shape != (3,):
		raise ValueError("Size must be a 1D array, 3 in length")
	if center.shape != (3,):
		raise ValueError("Center must be a 1D array, 3 in length")

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

	return create_vertex_list(batch, vertices, normals, indices, color,
	                          GL_QUADS, vertex_format, normal_format)


# Create a set of vertex lists for battlefield stage
def battlefield_creator(batch, color="blue"):
	# Keep track of all models
	model_list = []

	# Platform dimensions
	base_size = [16.0, 0.8, 3.0]
	base_center = [0.0, 0.0, 0.0]
	plat_size = [4.0, 0.3, 2.5]
	left_center = [-5.0, 3.0, 0.0]
	right_center = [5.0, 3.0, 0.0]
	top_center = [0.0, 5.0, 0.0]

	# Create all vertex_lists and use them to create models
	model_list.append(StaticNoClipModel(box_creator(batch, base_size, base_center,
	                                                color, "static", "static"),
	                                    is_platform=False))
	model_list.append(StaticNoClipModel(box_creator(batch, plat_size, left_center,
	                                                color, "static", "static"),
	                                    is_platform=True))
	model_list.append(StaticNoClipModel(box_creator(batch, plat_size, right_center,
	                                                color, "static", "static"),
	                                    is_platform=True))
	model_list.append(StaticNoClipModel(box_creator(batch, plat_size, top_center,
	                                                color, "static", "static"),
	                                    is_platform=True))

	return model_list


# Static model that does not check for collisions with environment
class StaticNoClipModel:

	# Constructor
	def __init__(self, vertex_list, is_platform=False):
		# Store the instance attributes
		self.vertex_list = vertex_list
		self.is_platform = is_platform

		# Extract a deepcopy of vertices formatted as Nx3 array
		self.vertices = []
		self.vertices.extend(self.vertex_list.vertices)
		self.vertices = np.reshape(self.vertices, (-1, 3))

		# Extract a deepcopy of normals formatted as Nx3 array
		self.normals = []
		self.normals.extend(self.vertex_list.normals)
		self.normals = np.reshape(self.normals, (-1, 3))

		# Compute transformation center and Environment Collision Box dimensions
		self.center = self._compute_center()
		self.ecb_dims = self._compute_ecb_dims()

	# Destructor
	def __del__(self):
		self.vertex_list.delete()

	# Determine model center from the vertices
	def _compute_center(self):
		return (np.max(self.vertices, axis=0) + np.min(self.vertices, axis=0)) / 2

	# Determine ecb dimensions from the vertices
	def _compute_ecb_dims(self):
		return (np.max(self.vertices, axis=0) - np.min(self.vertices, axis=0)) / 2

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
		self.ecb_dims *= scaling

	# Rotate both the real, local vertices, the ecb dimensions, and the normals
	def _rotate_vertices(self, rotation):
		self.vertices = rotation.apply(self.vertices)
		self.normals = rotation.apply(self.normals)
		self._update_vertices()
		self._update_normals()
		self.ecb_dims = self._compute_ecb_dims()

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


# Static model that does not check for collisions with environment
class DynamicClipModel(StaticNoClipModel):
	# Grounded attributes
	run_force = 100.0
	run_friction = 50.0
	jump_force = 25.0

	# Airbourne attributes
	fall_force = 50.0
	drift_force = 30.0
	gravity_force = 100.0

	# General attributes
	ang_speed = 500.0
	max_speed = 20.0

	# Constructor
	def __init__(self, vertex_list, keys, stage_model_list):

		# Call the parent constructor
		super().__init__(vertex_list)

		# Store the instance attributes
		self.keys = keys
		self.stage_model_list = stage_model_list

		# Start body at rest, in the air
		self.velocity = np.zeros(3)
		self.is_grounded = False

	# Update model position based off keyboard input, existing velocity, and evironment coliisions
	def update(self, dt):

		# Update color depending on status
		size = self.vertex_list.tex_coords.size
		if self.is_grounded:
			self.vertex_list.tex_coords = np.random.random(size) * 1000
		else:
			self.vertex_list.tex_coords = np.ones(size)

		# Rotate the body
		if self.keys[key.Q] and not self.keys[key.E]:
			self.rotate_degrees('y', dt * self.ang_speed)
		elif self.keys[key.E] and not self.keys[key.Q]:
			self.rotate_degrees('y', -dt * self.ang_speed)

		# Grounded motion
		if self.is_grounded:

			# Running
			if self.keys[key.D] and not self.keys[key.A]:
				self.velocity[0] += self.run_force * dt
			elif self.keys[key.A] and not self.keys[key.D]:
				self.velocity[0] -= self.run_force * dt

			# Jumping
			if self.keys[key.SPACE]:
				self.velocity[1] += self.jump_force
				self.is_grounded = False

			# Friction
			else:
				self.velocity[0] -= np.sign(self.velocity[0]) * self.run_friction * dt
				self.velocity[0] = np.where(np.abs(self.velocity[0]) - self.run_friction * dt < 0, 0, self.velocity[0])

				# Max speed
				self.velocity[0] = np.clip(self.velocity[0], -self.max_speed, self.max_speed)

		# Airborne motion
		else:

			# Horizontal drifting
			if self.keys[key.D] and not self.keys[key.A]:
				self.velocity[0] += self.drift_force * dt
			elif self.keys[key.A] and not self.keys[key.D]:
				self.velocity[0] -= self.drift_force * dt

			# Fast falling
			if self.velocity[1] < 0 and self.keys[key.S]:
				self.velocity[1] -= self.fall_force * dt

			# Gravity
			self.velocity[1] -= self.gravity_force * dt

		# Move player for non-zero velocity
		if not np.allclose(self.velocity, 0):
			self._translate_vertices(dt * self.velocity)

		# Clip detection
		for stage_model in self.stage_model_list:

			separation = np.abs(stage_model.center[0:2] - self.center[0:2]) \
			             - (stage_model.ecb_dims[0:2] + self.ecb_dims[0:2])

			# Check for negative separations
			if np.all(separation < 0):

				# Decide along which axis to eject the body
				xi = np.argmax(separation)

				# Determine direction along which to eject body
				sign = np.sign(self.velocity[xi])

				# Reset to grounded if landing on floor
				if xi == 1 and sign < 0:
					self.is_grounded = True

				# Treat platforms differently
				if stage_model.is_platform:

					# Only drop through platform if down is held
					if xi == 0 or sign > 0 or self.keys[key.S]:
						continue

				# Eject body
				displacement = self.ecb_dims[xi] + stage_model.ecb_dims[xi]
				new_position = np.copy(self.center)
				new_position[xi] = stage_model.center[xi] - sign * displacement
				self.set_position(new_position)

				# Set velocity to zero
				self.velocity[xi] = 0


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
	rx = ry = rz = dx = 0
	dy = -2
	dz = -15

	# Camera translation speed
	cam_rate = 0.7

	# Create stage models
	stage_model_list = battlefield_creator(batch)

	# Decorate stage with triangle models
	tri_front_vertex_list = triangle_practice(batch)
	tri_front_model = StaticNoClipModel(tri_front_vertex_list)
	tri_front_model.rescale(8)
	tri_front_model.set_position([0, -1, 1])

	tri_back_vertex_list = triangle_practice(batch)
	tri_back_model = StaticNoClipModel(tri_back_vertex_list)
	tri_back_model.rescale(8)
	tri_back_model.rotate_degrees('y', 180)
	tri_back_model.set_position([0, -1, -1])

	# Load 3D fox model
	os.chdir('fox/')
	fox = pyglet.model.load("low-poly-fox-by-pixelmannen.obj", batch=batch)
	fox_model = DynamicClipModel(fox.vertex_lists[0], keys, stage_model_list)

	# Configure initial conditions for fox model
	fox_model.rescale(2)
	fox_model.set_position([-5, 1.2, 0])
	fox_model.rotate_degrees('y', 90)

	# Keep track of all the active models
	object_list = []
	object_list.append(fox_model)

	# Run the animation!
	pyglet.app.run()
