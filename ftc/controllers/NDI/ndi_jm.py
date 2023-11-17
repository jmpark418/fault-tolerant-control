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
        
        """
            x, y position control
        """
        pos_e = pos - posd
        vel_e = vel - posd_dot
        
        Kpos = np.diag((1,1))
        Kvel = np.diag((1,1))
        
        nuo = (-Kpos @ pos_e[0:2] - Kvel @ vel_e[0:2]) / env.plant.g
        angd = np.vstack((nuo[1], -nuo[0], 0))
        
        """
            y1 (z, phi, theta, psi) control
        """
                        
        M_h = np.vstack((-env.plant.g, 
                         ang_dot[1]*ang_dot[2]))

        controller_info = {
            "angd": angd,
            "ang": ang,
            "posd": posd,
        }

        return ctrls, controller_info
