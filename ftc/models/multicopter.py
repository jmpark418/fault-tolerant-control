"""References
[1] T. Lee, M. Leok, and N. H. McClamroch, “Nonlinear Robust Tracking Control of a Quadrotor UAV on SE(3),” Asian J. Control, vol. 15, no. 2, pp. 391–408, Mar. 2013, doi: 10.1002/asjc.567.
[2] V. S. Akkinapalli, G. P. Falconí, and F. Holzapfel, “Attitude control of a multicopter using L1 augmented quaternion based backstepping,” Proceeding - ICARES 2014 2014 IEEE Int. Conf. Aerosp. Electron. Remote Sens. Technol., no. November, pp. 170–178, 2014.
[3] M. C. Achtelik, K. M. Doth, D. Gurdan, and J. Stumpf, “Design of a multi rotor MAV with regard to efficiency, dynamics and redundancy,” AIAA Guid. Navig. Control Conf. 2012, no. August, pp. 1–17, 2012.
[4] https://kr.mathworks.com/help/aeroblks/6dofquaternion.html#mw_f692de78-a895-4edc-a4a7-118228165a58
[5] M. C. Achtelik, K. M. Doth, D. Gurdan, and J. Stumpf, “Design of a multi rotor MAV with regard to efficiency, dynamics and redundancy,” AIAA Guid. Navig. Control Conf. 2014, no. August, pp. 1–17, 2012, doi: 10.2514/6.2012-4779.
[6] M. Faessler, A. Franchi, and D. Scaramuzza, “Differential Flatness of Quadrotor Dynamics Subject to Rotor Drag for Accurate Tracking of High-Speed Trajectories,” IEEE Robot. Autom. Lett., vol. 3, no. 2, pp. 620–626, Apr. 2018, doi: 10.1109/LRA.2017.2776353.
"""

import numpy as np

import fym
from fym.utils.rot import quat2dcm

from ftc.utils import safeupdate


class Mixer:
    """Definition:
    Mixer takes force commands and translate them to actuator commands.
    Actuator commands here mean the force generated by each rotor.
    """

    def __init__(self, d, c, b, rtype):

        if rtype == "quad":
            B = np.array(
                [
                    [1, 1, 1, 1],
                    [0, -d, 0, d],
                    [d, 0, -d, 0],
                    [-c, c, -c, c]
                ]
            )

        elif rtype == "hexa-x":
            B = np.array(
                [
                    [b, b, b, b, b, b],
                    [-b * d, b * d, b * d / 2, -b * d / 2, -b * d / 2, b * d / 2],
                    [
                        0,
                        0,
                        b * d * np.sqrt(3) / 2,
                        -b * d * np.sqrt(3) / 2,
                        b * d * np.sqrt(3) / 2,
                        -b * d * np.sqrt(3) / 2,
                    ],
                    [-c, c, -c, c, c, -c],
                ]
            )

        elif rtype == "hexa-+":
            B = np.array(
                [
                    [b, b, b, b, b, b],
                    [
                        0,
                        0,
                        b * d * np.sqrt(3) / 2,
                        -b * d * np.sqrt(3) / 2,
                        b * d * np.sqrt(3) / 2,
                        -b * d * np.sqrt(3) / 2,
                    ],
                    [-b * d, b * d, b * d / 2, -b * d / 2, -b * d / 2, b * d / 2],
                    [-c, c, -c, c, c, -c],
                ]
            )
            self.b_gyro = np.vstack((1, -1, 1, -1, 1, -1))

            s2 = 1 / 2
            s3 = np.sqrt(3) / 2
            self.d_rotor = np.array(
                [
                    [d, 0, 0],
                    [d * s2, -d * s3, 0],
                    [-d * s2, -d * s3, 0],
                    [-d, 0, 0],
                    [-d * s2, d * s3, 0],
                    [d * s2, d * s3, 0],
                ]
            )

        elif rtype == "hexa-falconi":
            B = np.array(
                [
                    [b, b, b, b, b, b],
                    [
                        0.5 * d * b,
                        d * b,
                        0.5 * d * b,
                        -0.5 * d * b,
                        -d * b,
                        -0.5 * d * b,
                    ],
                    [
                        0.5 * np.sqrt(3) * d * b,
                        0,
                        -0.5 * np.sqrt(3) * d * b,
                        -0.5 * np.sqrt(3) * d * b,
                        0,
                        0.5 * np.sqrt(3) * d * b,
                    ],
                    [c, -c, c, -c, c, -c],
                ]
            )

        else:
            B = np.eye(4)

        self.b = b
        self.B = B
        self.Binv = np.linalg.pinv(B)

    def inverse(self, rotors):
        return self.B.dot(rotors)

    def __call__(self, forces):
        return self.Binv.dot(forces)


class Multicopter(fym.BaseEnv):
    """Multicopter Model
    Variables:
        pos: position in I-coord
        vel: velocity in I-coord
        quat: unit quaternion.
            Corresponding to the rotation matrix from I- to B-coord.

    Drag model [6]:
        self.M_gyroscopic
        self.A_drag
        self.B_drag
        self.D_drag
        - Note: orientation `R` in the paper is `dcm.T`

    """

    # Model parameters
    g = 9.81
    rho = 1.225
    modelfrom = "TLee"
    if modelfrom == "TLee":  # Taeyoung Lee's model for quadrotor UAV [1]
        m = 4.34
        J = np.diag([0.0820, 0.0845, 0.1377])
        d = 0.315
        c = 8.004e-4
        b = 1
    elif modelfrom == "Falconi":  # G. P. Falconi's multicopter model [2-4]
        m = 0.64
        J = np.diag([0.010007, 0.0102335, 0.0081])
        d = 0.215
        c = 1.2864e-7
        b = 6.546e-6

    rotor_min = 0
    rotor_max = m * g * 0.6371  # maximum thrust for each  rotor [5]

    ENV_CONFIG = {
        "init": {
            "pos": np.zeros((3, 1)),
            "vel": np.zeros((3, 1)),
            "quat": np.vstack((1, 0, 0, 0)),
            "omega": np.zeros((3, 1)),
        },
    }

    def __init__(self, env_config={}, dx=0.0, dy=0.0, dz=0.0, rtype="hexa-x"):
        env_config = safeupdate(self.ENV_CONFIG, env_config)
        super().__init__()
        self.pos = fym.BaseSystem(env_config["init"]["pos"])
        self.vel = fym.BaseSystem(env_config["init"]["vel"])
        self.quat = fym.BaseSystem(env_config["init"]["quat"])
        self.omega = fym.BaseSystem(env_config["init"]["omega"])

        self.Jinv = np.linalg.inv(self.J)
        self.M_gyroscopic = np.zeros((3, 1))
        self.A_drag = np.diag(np.zeros(3))  # currently ignored
        self.B_drag = np.diag(np.zeros(3))  # currently ignored
        self.D_drag = np.diag([dx, dy, dz])
        self.mixer = Mixer(d=self.d, c=self.c, b=self.b, rtype=rtype)

    def deriv(self, pos, vel, quat, omega, rotors):
        F, M1, M2, M3 = self.mixer.inverse(rotors)

        M = np.vstack((M1, M2, M3))

        m, g, J = self.m, self.g, self.J
        e3 = np.vstack((0, 0, 1))

        dpos = vel
        dcm = quat2dcm(quat)
        dvel = g * e3 - F * dcm.T.dot(e3) / m - dcm.T.dot(self.D_drag).dot(dcm).dot(vel)
        # DCM integration (Note: dcm; I to B) [1]
        p, q, r = np.ravel(omega)
        # unit quaternion integration [4]
        dquat = 0.5 * np.array(
            [[0.0, -p, -q, -r], [p, 0.0, r, -q], [q, -r, 0.0, p], [r, q, -p, 0.0]]
        ).dot(quat)
        eps = 1 - (quat[0] ** 2 + quat[1] ** 2 + quat[2] ** 2 + quat[3] ** 2)
        k = 1
        dquat = dquat + k * eps * quat
        domega = self.Jinv.dot(
            M
            - np.cross(omega, J.dot(omega), axis=0)
            - self.M_gyroscopic
            - self.A_drag.dot(dcm).dot(vel)
            - self.B_drag.dot(omega)
        )

        return dpos, dvel, dquat, domega

    def set_dot(self, t, rotors):
        states = self.observe_list()
        dots = self.deriv(*states, rotors)
        self.pos.dot, self.vel.dot, self.quat.dot, self.omega.dot = dots

    def saturate(self, rotors):
        _rotors = np.zeros((rotors.shape))
        _rotors = np.clip(rotors, self.rotor_min, self.rotor_max)
        return _rotors


if __name__ == "__main__":
    system = Multicopter()
    system.set_dot(t=0, rotors=np.zeros((6, 1)))
    print(repr(system))
