import numpy as np
from se2 import SE2
from so2 import SO2

def unicycle_flat_states(pd, vd):
  return SE2(np.atan2(vd.item(1), vd.item(0)), pd.item(0), pd.item(1))

def unicycle_flat_inputs(vd, ad, jac=False):
  vd_norm = np.linalg.norm(vd)
  u = np.array([[np.linalg.norm(vd)],
                [(-1.0/np.linalg.norm(vd)**2 * vd.T @ SO2.hat(1) @ ad).item(0)]])
  if not jac:
    return u if not np.isclose(vd_norm, 0.0) else np.zeros((2,1))

  jac_v = np.vstack([1.0/np.linalg.norm(vd) * vd.T,
                    1.0/np.linalg.norm(vd)**2 * ad.T @ SO2.hat(1) - 2.0/np.linalg.norm(vd)**4 * vd.T * (ad.T @ SO2.hat(1) @ vd)])
  jac_a = np.vstack([np.zeros((1,2)),
                    -1.0/np.linalg.norm(vd)**2 * vd.T @ SO2.hat(1)])

  return (u, jac_v, jac_a) if not np.isclose(vd_norm, 0.0) else (np.zeros((2,1)), np.array([[0.0, 0.0], [0.0, 0.0]]), np.zeros((2,2)))