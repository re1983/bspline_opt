import numpy as np
from unicycle_model import *
from spline import *
import sys

def test_unicycle_jacs():
  eval_pts = np.linspace(-10, 10, 10).tolist()
  x = []
  for a in eval_pts:
    for b in eval_pts:
      for c in eval_pts:
        for d in eval_pts:
          x.append((np.array([[a,b]]).T, np.array([[c,d]]).T))

  h = 1e-6

  fun = unicycle_flat_inputs

  for x1, x2 in x:
    jac_v = np.zeros((2,2))
    jac_a = np.zeros((2,2))

    for i in range(len(x1.tolist())):
      e = np.zeros(x1.shape)
      e[i] = 1.0
      jac_v[:,[i]] = (fun(x1 + h*e, x2) - fun(x1, x2))/h

    for i in range(len(x2.tolist())):
      e = np.zeros(x2.shape)
      e[i] = 1.0
      jac_a[:,[i]] = (fun(x1, x2 + h*e) - fun(x1, x2))/h

    _, jac_v_eval, jac_a_eval = fun(x1, x2, jac=True)

    if np.linalg.norm(jac_v - jac_v_eval) > 1e-3 and not np.isclose(np.linalg.norm(x1), 0.0):
      print("Jac v failed: ")
      print(x1, x2)
      print(jac_v)
      print(jac_v_eval)

    if np.linalg.norm(jac_a - jac_a_eval) > 1e-3:
      print("Jac a failed: ")
      print(x1, x2)
      print(jac_a)
      print(jac_a_eval)


def test_spline_cp_jacs():
  # ctrl_pts = [np.array([[0,0]]).T, np.array([[0,0]]).T, np.array([[0,0]]).T, 
  #                 np.array([[1,1]]).T, 
  #                 np.array([[2,2]]).T, 
  #                 np.array([[3,3]]).T, 
  #                 np.array([[4,4]]).T, 
  #                 np.array([[5,5]]).T, 
  #                 np.array([[6,6]]).T, 
  #                 np.array([[7,7]]).T, 
  #                 np.array([[8,8]]).T, 
  #                 np.array([[9,9]]).T, 
  #                 np.array([[10,10]]).T, np.array([[10,10]]).T, np.array([[10,10]]).T]

  ctrl_pts = [np.array([[0],[0]]),
              np.array([[0],[0]]),
              np.array([[0],[0]]),
              np.array([[0.73129424],[1.26942609]]),
              np.array([[1.83259101],[2.20305541]]),
              np.array([[2.62263369],[3.41106366]]),
              np.array([[3.72159126],[4.35300428]]),
              np.array([[5.00036494],[4.99929259]]),
              np.array([[6.27810694],[5.64760749]]),
              np.array([[7.37770837],[6.58832444]]),
              np.array([[8.17232273],[7.79305542]]),
              np.array([[9.2763986 ],[8.72394108]]),
              np.array([[10],[10]]),
              np.array([[10],[10]]),
              np.array([[10],[10]])]
          
  knot_space = 1.0
  knot_pts = (np.array([-3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])*knot_space).tolist() # evaluatable from 0 to 12

  spline = Spline(ctrl_pts, knot_pts)

  h = 1e-6

  tv = np.linspace(knot_pts[3], knot_pts[-4]-h, 100).tolist()
  dv = [0, 1, 2, 3]

  for d in dv:
    for t in tv:
      for i in range(len(ctrl_pts)-6):
        cp_jac = np.zeros((2,2))
        cp_jac_eval = spline.get_cp_jacobian(i+3, t, derivative=d)
        for j in range(ctrl_pts[i].shape[0]):
          p1 = spline.evaluate(t, derivative=d)
          delta = [np.zeros((2,1)) for k in range(len(ctrl_pts))]
          delta[i+3][j] = h
          spline.update_control_points(delta)
          p2 = spline.evaluate(t, derivative=d)
          cp_jac[:,[j]] = (p2 - p1)/h
          spline.control_points = ctrl_pts
        # print(i, ": ", cp_jac)
        # print(cp_jac_eval)
        if np.linalg.norm(cp_jac - cp_jac_eval) > 1e-5:
          print("Failed for i = ", i, ", t = ", t)
          print(cp_jac)
          print(cp_jac_eval)


def test_spline_dt_jacs():
  # ctrl_pts = [np.array([[0,0]]).T, np.array([[0,0]]).T, np.array([[0,0]]).T, 
  #             np.array([[1,1]]).T, 
  #             np.array([[2,2]]).T, 
  #             np.array([[3,3]]).T, 
  #             np.array([[4,4]]).T, 
  #             np.array([[5,5]]).T, 
  #             np.array([[6,6]]).T, 
  #             np.array([[7,7]]).T, 
  #             np.array([[8,8]]).T, 
  #             np.array([[9,9]]).T, 
  #             np.array([[10,10]]).T, np.array([[10,10]]).T, np.array([[10,10]]).T]

  ctrl_pts = [np.array([[0],[0]]),
              np.array([[0],[0]]),
              np.array([[0],[0]]),
              np.array([[0.73129424],[1.26942609]]),
              np.array([[1.83259101],[2.20305541]]),
              np.array([[2.62263369],[3.41106366]]),
              np.array([[3.72159126],[4.35300428]]),
              np.array([[5.00036494],[4.99929259]]),
              np.array([[6.27810694],[5.64760749]]),
              np.array([[7.37770837],[6.58832444]]),
              np.array([[8.17232273],[7.79305542]]),
              np.array([[9.2763986 ],[8.72394108]]),
              np.array([[10],[10]]),
              np.array([[10],[10]]),
              np.array([[10],[10]])]
          
  knot_space = 1.0
  knot_pts = (np.array([-3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])*knot_space).tolist() # evaluatable from 0 to 12

  spline = Spline(ctrl_pts, knot_pts)

  h = 1e-8

  tv = np.linspace(knot_pts[3], knot_pts[-4]-h, 100).tolist()
  dv = [0, 1, 2, 3]
  
  for d in dv:
    for t in tv:
      dt_jac_eval = spline.get_dt_jacobian(t, derivative=d)
      p1 = spline.evaluate(t, derivative=d)
      spline.knot_points = (np.array(knot_pts)/knot_space*(knot_space+h)).tolist()
      spline.dt = knot_space+h
      p2 = spline.evaluate(t, derivative=d)
      dt_jac = (p2 - p1)/h
      spline.knot_points = knot_pts
      spline.dt = knot_space
      # print(dt_jac)
      # print(dt_jac_eval)
      if np.linalg.norm(dt_jac - dt_jac_eval) > 1e-5:
        print("Failed for d = ", d, ", t = ", t)
        print(dt_jac)
        print(dt_jac_eval)


def test_obs_cons_jacs():
  obs = [np.array([[4,4.25]]).T, np.array([[6,6.75]]).T]

  # endpoints will be kept fixed
  init_ctrl_pts = [np.array([[0,0]]).T, np.array([[0,0]]).T, np.array([[0,0]]).T, 
                    np.array([[1,1]]).T, 
                    np.array([[2,2]]).T, 
                    np.array([[3,3]]).T, 
                    np.array([[4,4]]).T, 
                    np.array([[5,5]]).T, 
                    np.array([[6,6]]).T, 
                    np.array([[7,7]]).T, 
                    np.array([[8,8]]).T, 
                    np.array([[9,9]]).T, 
                    np.array([[10,10]]).T, np.array([[10,10]]).T, np.array([[10,10]]).T]

  init_knot_space = 1.0
  init_knot_pts = (np.array([-3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])*init_knot_space).tolist() # evaluatable from 0 to 12
  cons_eval_pts = np.linspace(0, 1-1e-6, 5).tolist()

  x0 = []
  for i in range(len(init_ctrl_pts)-6):
    x0 += init_ctrl_pts[i+3].flatten().tolist().copy()
  x0.append(init_knot_space)

  def obs_cons(x):
    con = []

    ctrl_pts = init_ctrl_pts.copy()
    ctrl_pts[3:len(init_ctrl_pts)-3] = [np.array([[x[2*i], x[2*i+1]]]).T for i in range((len(x)-1)//2)]
    knot_pts = (np.array(init_knot_pts)/init_knot_space * x[-1]).tolist()
    spline = Spline(ctrl_pts, knot_pts)

    tf = knot_pts[-4]
    for i in range(len(obs)):
      for j in range(len(cons_eval_pts)):
        con.append(np.linalg.norm(spline.evaluate(cons_eval_pts[j]*tf) - obs[i])**2)

    return con

  def obs_cons_jac(x):
    ctrl_pts = init_ctrl_pts.copy()
    ctrl_pts[3:len(init_ctrl_pts)-3] = [np.array([[x[2*i], x[2*i+1]]]).T for i in range((len(x)-1)//2)]
    knot_pts = (np.array(init_knot_pts)/init_knot_space * x[-1]).tolist()
    spline = Spline(ctrl_pts, knot_pts)

    tf = knot_pts[-4]
    jac = np.zeros((len(obs)*len(cons_eval_pts), len(x)))
    for i in range(len(obs)):
      for j in range(len(cons_eval_pts)):
        p_t = spline.evaluate(cons_eval_pts[j]*tf)
        for k in range(len(ctrl_pts)-6):
          spl_jac = spline.get_cp_jacobian(k+3, cons_eval_pts[j]*tf)
          jac[i*len(cons_eval_pts)+j, 2*k:2*k+2] = 2*(p_t - obs[i]).T @ spl_jac
        
        # dt_ddeltat = cons_eval_pts[j]*(len(knot_pts)-6)
        jac[i*len(cons_eval_pts)+j, -1] = 0.0#2*(p_t - obs[i]).T @ (spline.get_dt_jacobian(cons_eval_pts[j]*tf) + spline.evaluate(cons_eval_pts[j]*tf, 1)*dt_ddeltat)

    return jac

  h = 1e-6
  obs_jac = np.zeros((len(obs)*len(cons_eval_pts), len(x0)))
  obs_jac_eval = obs_cons_jac(x0)
  for i in range(len(x0)):
    x = x0
    C1 = obs_cons(x)
    x[i] += h
    C2 = obs_cons(x)
    obs_jac[:,[i]] = (np.array([C2]).T - np.array([C1]).T)/h
  
  print(obs_jac - obs_jac_eval)
  # print(obs_jac_eval)
  

def test_inp_cons_jacs():
  # endpoints will be kept fixed
  init_ctrl_pts = [np.array([[0,0]]).T, np.array([[0,0]]).T, np.array([[0,0]]).T, 
                    np.array([[1,1]]).T, 
                    np.array([[2,2]]).T, 
                    np.array([[3,3]]).T, 
                    np.array([[4,4]]).T, 
                    np.array([[5,5]]).T, 
                    np.array([[6,6]]).T, 
                    np.array([[7,7]]).T, 
                    np.array([[8,8]]).T, 
                    np.array([[9,9]]).T, 
                    np.array([[10,10]]).T, np.array([[10,10]]).T, np.array([[10,10]]).T]

  # init_ctrl_pts = [np.array([[0],[0]]),
  #             np.array([[0],[0]]),
  #             np.array([[0],[0]]),
  #             np.array([[0.73129424],[1.26942609]]),
  #             np.array([[1.83259101],[2.20305541]]),
  #             np.array([[2.62263369],[3.41106366]]),
  #             np.array([[3.72159126],[4.35300428]]),
  #             np.array([[5.00036494],[4.99929259]]),
  #             np.array([[6.27810694],[5.64760749]]),
  #             np.array([[7.37770837],[6.58832444]]),
  #             np.array([[8.17232273],[7.79305542]]),
  #             np.array([[9.2763986 ],[8.72394108]]),
  #             np.array([[10],[10]]),
  #             np.array([[10],[10]]),
  #             np.array([[10],[10]])]

  init_knot_space = 0.2
  init_knot_pts = (np.array([-3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])*init_knot_space).tolist() # evaluatable from 0 to 12
  cons_eval_pts = np.linspace(0, 1-1e-6, 5).tolist()

  x0 = []
  for i in range(len(init_ctrl_pts)-6):
    x0 += init_ctrl_pts[i+3].flatten().tolist().copy()
  x0.append(init_knot_space)

  def inp_cons(x):
    con = []

    ctrl_pts = init_ctrl_pts.copy()
    ctrl_pts[3:len(init_ctrl_pts)-3] = [np.array([[x[2*i], x[2*i+1]]]).T for i in range((len(x)-1)//2)]
    knot_pts = (np.array(init_knot_pts)/init_knot_space * x[-1]).tolist()
    spline = Spline(ctrl_pts, knot_pts)

    tf = knot_pts[-4]
    for i in range(len(cons_eval_pts)):
      inp = unicycle_flat_inputs(spline.evaluate(cons_eval_pts[i]*tf, 1), spline.evaluate(cons_eval_pts[i]*tf, 2))
      con += inp.flatten().tolist()

    return con

  def inp_cons_jac(x):
    ctrl_pts = init_ctrl_pts.copy()
    ctrl_pts[3:len(init_ctrl_pts)-3] = [np.array([[x[2*i], x[2*i+1]]]).T for i in range((len(x)-1)//2)]
    knot_pts = (np.array(init_knot_pts)/init_knot_space * x[-1]).tolist()
    spline = Spline(ctrl_pts, knot_pts)

    tf = knot_pts[-4]
    jac = np.zeros((2*len(cons_eval_pts), len(x)))
    for i in range(len(cons_eval_pts)):
      vel = spline.evaluate(cons_eval_pts[i]*tf, 1)
      acc = spline.evaluate(cons_eval_pts[i]*tf, 2)
      jer = spline.evaluate(cons_eval_pts[i]*tf, 3)
      _, jac_v, jac_a = unicycle_flat_inputs(vel, acc, jac=True)
      for j in range(len(ctrl_pts)-6):
        spl_jac_v = spline.get_cp_jacobian(j+3, cons_eval_pts[i]*tf, 1)
        spl_jac_a = spline.get_cp_jacobian(j+3, cons_eval_pts[i]*tf, 2)
        jac[2*i:2*i+2, 2*j:2*j+2] = jac_v @ spl_jac_v + jac_a @ spl_jac_a

      # dt_ddeltat = cons_eval_pts[i]*(len(knot_pts)-6)
      # jac[2*i:2*i+2, [-1]] = jac_v @ (spline.get_dt_jacobian(cons_eval_pts[i]*tf, 1) + acc*dt_ddeltat) + jac_a @ (spline.get_dt_jacobian(cons_eval_pts[i]*tf, 2) + jer*dt_ddeltat)
      jac[2*i:2*i+2, [-1]] = jac_v @ spline.get_dt_jacobian(cons_eval_pts[i]*tf, 1, set_du_ddeltat_zero=True) + jac_a @ spline.get_dt_jacobian(cons_eval_pts[i]*tf, 2, set_du_ddeltat_zero=True)


    return jac

  h = 1e-8
  inp_jac = np.zeros((2*len(cons_eval_pts), len(x0)))
  inp_jac_eval = inp_cons_jac(x0)
  for i in range(len(x0)):
    x = x0
    C1 = inp_cons(x)
    x[i] += h
    C2 = inp_cons(x)
    inp_jac[:,[i]] = (np.array([C2]).T - np.array([C1]).T)/h
  
  print(np.linalg.norm(inp_jac-inp_jac_eval))
  # np.set_printoptions(threshold=sys.maxsize)
  # print(inp_jac - inp_jac_eval)
  # for i in range(2*len(cons_eval_pts)):
  #   print(inp_jac[i,:])
  #   print(inp_jac_eval[i,:])


def main():
  # test_unicycle_jacs()
  # test_spline_cp_jacs()
  # test_spline_dt_jacs()
  # test_obs_cons_jacs()
  test_inp_cons_jacs()


if __name__=="__main__":
  main()