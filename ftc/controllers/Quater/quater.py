"""References
[1]J. Cariño, H. Abaunza and P. Castillo, "Quadrotor quaternion control," 2015 International Conference on Unmanned Aircraft Systems (ICUAS), Denver, CO, USA, 2015, pp. 825-831, doi: 10.1109/ICUAS.2015.7152367.
"""

import fym
import control
import numpy as np

from fym.utils.rot import quat2angle

class QuaterController(fym.BaseEnv):
    def __init__(self, env):
        super().__init__()
        m, g, Jinv = env.plant.m, env.plant.g, env.plant.Jinv
        # self.trim_forces = np.vstack([m * g, 0, 0, 0])
        
        Aatt = np.array(
            [
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ]
        )
        Batt = np.vstack((np.zeros((3,3)), np.identity(3)))
        Qatt = np.diag((1, 1, 1, 1, 1, 1))
        Ratt = np.diag((1, 1, 1))
        
        Apos = np.array(
            [
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ]
        )
        Bpos = np.vstack((np.zeros((3,3)), np.identity(3)))
        Qpos = np.diag((1, 1, 1, 1, 1, 1))
        Rpos = np.diag((1, 1, 1))
        
        self.Katt, *_ = control.lqr(Aatt, Batt, Qatt, Ratt)
        self.Kpos, *_ = control.lqr(Apos, Bpos, Qpos, Rpos)
        
        
    def get_control(self, t, env):
        pos, vel, quat, omega = env.plant.observe_list()
        ang = np.vstack(quat2angle(quat)[::-1])
        
        posd = np.vstack((1, 2, 0))
        veld = np.zeros((3,1))
        omegad = np.zeros((3,1))
        angd = np.deg2rad(np.vstack((0, 0, 0)))
        
        q0 = quat[0]
        qbar = quat[1::]
        qnorm = np.linalg.norm(quat)
        qbarnorm = np.linalg.norm(qbar)   
         
        if qbarnorm == 0:
            lnq = np.zeros((3, 1))
        else:
            lnq = qbar / qbarnorm * np.arccos(q0)
            
        theta = 2 * lnq
        
        xatt = np.vstack((theta, omega))
        xattd = np.vstack((angd, omegad))
        
        xpos = np.vstack((pos, vel))
        xposd = np.vstack((posd, veld))
        
        # x = np.vstack((xpos, xatt))
        # xd = np.vstack((xposd, xattd))
        # K = np.hstack((self.Katt, self.Kpos))
        
        # torques = -K @ (x - xd)        
        torques = -self.Katt @ (xatt - xattd) - self.Kpos @ (xpos - xposd)
        forces = np.vstack((env.plant.m * env.plant.g, torques))
        ctrl = np.linalg.pinv(env.plant.mixer.B) @ forces
        
        controller_info = {
            "angd": angd,
            "ang": ang,
            "posd": posd,
        }
        
        return ctrl, controller_info
        
        