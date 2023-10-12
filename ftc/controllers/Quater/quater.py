"""References
[1]J. Cariño, H. Abaunza and P. Castillo, "Quadrotor quaternion control," 2015 International Conference on Unmanned Aircraft Systems (ICUAS), Denver, CO, USA, 2015, pp. 825-831, doi: 10.1109/ICUAS.2015.7152367.
"""

import control
import fym
import numpy as np
from fym.utils.rot import quat2angle


class QuaterController(fym.BaseEnv):
    def __init__(self, env):
        super().__init__()
        m, g, Jinv = env.plant.m, env.plant.g, env.plant.Jinv

        self.thetadf = fym.BaseSystem(np.zeros((3, 1)))
        self.tau = 0.05

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
        Bpos = np.vstack((np.zeros((3, 3)), np.diag((1, 1, 1))))
        Qpos = np.diag((1, 1, 1, 1, 1, 1))
        Rpos = np.diag((1, 1, 1))

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
        Batt = np.vstack((np.zeros((3, 3)), np.diag((1, 1, 1))))
        Qatt = np.diag((100, 100, 100, 20, 20, 20))
        Ratt = 0.1 * np.diag((1, 1, 1))

        self.Kpos, *_ = control.lqr(Apos, Bpos, Qpos, Rpos)
        self.Katt, *_ = control.lqr(Aatt, Batt, Qatt, Ratt)

    def get_control(self, t, env):
        pos, vel, quat, omega = env.plant.observe_list()

        ang = np.vstack(quat2angle(quat)[::-1])
        angd = np.zeros((3, 1))

        posd = np.vstack((0, 0, 0))
        veld = np.vstack((0, 0, 0))
        quatd = np.vstack((1, 0, 0, 0))
        # omegad = np.vstack((0, 0, 0))

        gbar = np.vstack((0, 0, env.plant.g))

        qnorm = np.linalg.norm(quat)
        q = quat / qnorm
        q0 = q[0]
        qbar = q[1::]
        qbarnorm = np.linalg.norm(qbar)

        if qbarnorm != 0:
            lnq = qbar / qbarnorm * np.arccos(q0)
        else:
            lnq = np.zeros((3, 1))

        thetabar = 2 * lnq
        xatt = np.vstack((thetabar, omega))

        xpos = np.vstack((pos, vel))
        xposd = np.vstack((posd, veld))

        upos = -self.Kpos @ (xpos - xposd)
        up = upos - gbar
        b = np.vstack((0, 0, -1))

        qpd = np.vstack(
            (np.dot(np.transpose(b), up) + np.linalg.norm(up), np.cross(b, up, axis=0))
        )
        qd = qpd / np.linalg.norm(qpd)

        qd0 = qd[0]
        qdbar = qd[1::]
        qdbarnorm = np.linalg.norm(qdbar)

        if qdbarnorm != 0:
            lnqd = qdbar / qdbarnorm * np.arccos(qd0)
        else:
            lnqd = np.zeros((3, 1))

        thetadbar = 2 * lnqd

        thetadf = self.thetadf.state
        thetadf_dot = (thetadbar - thetadf) / self.tau
        omegad = (thetadbar - thetadf) / self.tau
        self.thetadf.dot = thetadf_dot

        xattd = np.vstack((thetadbar, omegad))

        uatt = -self.Katt @ (xatt - xattd)

        F_th = np.linalg.norm(up) * env.plant.m
        forces = np.vstack((F_th, uatt))

        ctrl = np.linalg.pinv(env.plant.mixer.B) @ forces

        q_ypr = np.vstack((quat2angle(quat)[::-1]))
        qd_ypr = np.vstack((quat2angle(qd)[::-1]))

        controller_info = {
            "angd": angd,
            "q": q,
            # "qd": qd,
            "theta": thetabar,
            # "thetad": thetad,
            "ang": ang,
            "posd": posd,
            "q_ypr": q_ypr,
            "qd_ypr": qd_ypr,
            "theta_f": thetadf,
        }

        return ctrl, controller_info
