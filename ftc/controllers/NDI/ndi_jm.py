"""References
[1] A. Das, K. Subbarao and F. Lewis, "Dynamic inversion of quadrotor with zero-dynamics stabilization," 2008 IEEE International Conference on Control Applications, San Antonio, TX, USA, 2008, pp. 1189-1194, doi: 10.1109/CCA.2008.4629582.
"""
import fym
import numpy as np
from fym.utils.rot import quat2angle


class NDIController(fym.BaseEnv):
    def __init__(self, env):
        super().__init__()
        

    def get_control(self, t, env):
        pos, vel, quat, ang_dot = env.plant.observe_list()
        ang = np.vstack(quat2angle(quat)[::-1])

        posd, posd_dot = env.get_ref(t, "posd", "posd_dot")
        angd_dot = np.zeros((3,1))
        
        """
            x, y position control
        """
        pos_e = pos - posd
        vel_e = vel - posd_dot
        
        Kpos = np.diag((1,1))
        Kvel = np.diag((1,1))
        
        accd = (-Kpos @ pos_e[0:2] - Kvel @ vel_e[0:2]) / env.plant.g
        accd_mag = np.linalg.norm(accd)
        
        angd = np.vstack((accd[1], -accd[0], 0))
        
        """
            y1 (z, phi, theta, psi) control
        """
        ang_e = ang - angd
        ang_dot_e = ang_dot - angd_dot
        
        K1 = np.diag((10, 5, 5, 5))
        K2 = np.diag((10, 1, 1, 1))
        
        nu_h = np.vstack((-K1[0,0]*vel_e[2] - K2[0,0]*pos_e[2],
                          -K1[1:4,1:] @ ang_e - K2[1:4,1:] @ ang_dot_e))
        
        E_h = np.diag([float(-np.cos(ang[1])*np.cos(ang[0])/env.plant.m), env.plant.d/env.plant.J[0,0], env.plant.d/env.plant.J[1,1], env.plant.d/env.plant.J[2,2]])
                        
        M_h = np.vstack((env.plant.g, 
                         ang_dot[1]*ang_dot[2]*((env.plant.J[1,1]-env.plant.J[2,2])/env.plant.J[0,0]),
                         ang_dot[0]*ang_dot[2]*((env.plant.J[2,2]-env.plant.J[0,0])/env.plant.J[1,1]),
                         ang_dot[0]*ang_dot[1]*((env.plant.J[0,0]-env.plant.J[1,1])/env.plant.J[2,2])))
        
        forces = np.linalg.inv(E_h) @ (-M_h + nu_h)
        ctrls = np.linalg.inv(env.plant.mixer.B) @ forces

        controller_info = {
            "angd": angd,
            "ang": ang,
            "posd": posd,
        }

        return ctrls, controller_info
