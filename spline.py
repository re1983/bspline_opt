import numpy as np

# this currently only supports uniform cubic splines
class Spline:
  def __init__(self, control_points, knot_points, degree=3):
    assert(len(knot_points) == len(control_points) + degree + 1)
    self.control_points = control_points
    self.knot_points = knot_points
    self.degree = degree

    # self.B = 1.0/6.0 * np.array([[6,0,0,0], [5,3,-3,1], [1,3,3,-2], [0,0,0,1]]).T
    self.B = 1/6.0 * np.array([[1,4,1,0], [-3,0,3,0], [3,-6,3,0], [-1,3,-3,1]])
    self.D = np.array([[0,1,0,0], [0,0,2,0], [0,0,0,3], [0,0,0,0]])
    self.dim = np.max(control_points[0].shape)

    self.detect_uniform()

  def detect_uniform(self):
    self.uniform = False
    for i in range(len(self.knot_points)):
      if i == 0:
        continue
      if i == 1:
        dt = self.knot_points[i] - self.knot_points[i-1]
      else:
        if not np.isclose(self.knot_points[i] - self.knot_points[i-1], dt):
          break

      if i == (len(self.knot_points) - 1):
        self.uniform = True
        self.dt = dt

  def evaluate(self, t, derivative=0):
    assert(derivative <= self.degree)
    assert(t >= self.knot_points[self.degree] and t <= self.knot_points[-4])

    i = 1
    while t >= self.knot_points[i+1]:
      i += 1

    Phid = self.calculate_Phid(t, i, derivative)

    p_bar = np.vstack([self.control_points[i-3], self.control_points[i-2], self.control_points[i-1], self.control_points[i]])
    return Phid @ p_bar

  def calculate_Phid(self, t, i, derivative=0, is_dt_jac=False, set_du_ddeltat_zero=False):
    denom = self.dt if self.uniform else (self.knot_points[i+1] - self.knot_points[i])
    u_i = (t - self.knot_points[i])/denom
    u = np.array([[1, u_i, u_i**2, u_i**3]])

    if is_dt_jac:
      du_ddeltat = np.multiply((u + np.array([[0, i-3, (i-3)*u_i, (i-3)*u_i**2]])), np.array([0, -1/self.dt, -2/self.dt, -3/self.dt])) if not set_du_ddeltat_zero else np.zeros((1,4))
      u = du_ddeltat - float(derivative)/self.dt * u

    D = 1/denom * self.D
    u_d = u 
    for j in range(derivative):
      u_d = u_d @ D

    return np.block([u_d.dot(self.B[:,0]) * np.eye(self.dim), u_d.dot(self.B[:,1]) * np.eye(self.dim), u_d.dot(self.B[:,2]) * np.eye(self.dim), u_d.dot(self.B[:,3]) * np.eye(self.dim)])

  def update_control_points(self, delta):
    self.control_points = [control + delta[i] for i, control in enumerate(self.control_points)]

  def update_knot_spacing(self, dt):
    assert(self.uniform)
    for i in range(1,len(self.knot_points)):
      self.knot_points[i] = self.knot_points[i-1] + dt
    self.dt = dt

  # Calculate the jacobian of the spline derivative evaluated at time t with respect
  # to a specific control point
  def get_cp_jacobian(self, cp_ind, t, derivative=0):
    assert(derivative <= self.degree)
    assert(t >= self.knot_points[self.degree] and t <= self.knot_points[-4])
    assert(cp_ind > 0 and cp_ind < len(self.control_points))

    i = 1
    while t >= self.knot_points[i+1]:
      i += 1

    if cp_ind < (i - 3) or cp_ind > i:
      return np.zeros((self.dim, self.dim))

    Phid = self.calculate_Phid(t, i, derivative)
    pos = 3-(i-cp_ind)
    return Phid[:, self.dim*pos:self.dim*(pos+1)]

  # Calculate the jacobian of the spline derivative evaluated at time t with respect
  # to the uniform knot spacing. This is needed e.g. to optimize minimum time trajectories
  def get_dt_jacobian(self, t, derivative=0, set_du_ddeltat_zero=False):
    assert(self.uniform)
    assert(derivative <= self.degree)
    assert(t >= self.knot_points[self.degree] and t <= self.knot_points[-4])

    i = 1
    while t >= self.knot_points[i+1]:
      i += 1

    Phid = self.calculate_Phid(t, i, derivative, True, set_du_ddeltat_zero)

    p_bar = np.vstack([self.control_points[i-3], self.control_points[i-2], self.control_points[i-1], self.control_points[i]])
    return Phid @ p_bar