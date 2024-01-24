"""References
[1] Mueller MW, D'Andrea R. Relaxed hover solutions for multicopters: Application to algorithmic redundancy and novel vehicles. The International Journal of Robotics Research. 2016;35(8):873-889. doi:10.1177/0278364915596233
[2] M. W. Mueller and R. D'Andrea, "Stability and control of a quadrocopter despite the complete loss of one, two, or three propellers," 2014 IEEE International Conference on Robotics and Automation (ICRA), Hong Kong, China, 2014, pp. 45-52, doi: 10.1109/ICRA.2014.6906588.
"""
import fym
import numpy as np
from fym.utils.rot import quat2angle
import control


class ReducedAttitudeController(fym.BaseEnv):
    def __init__(self, env):
        super().__init__()
        m, g, J, l= env.plant.m, env.plant.g, env.plant.J, env.plant.d
        
        Ixx = J[0,0]
        Iyy = J[1,1]
        Izz = J[2,2]
        
        n_bar = np.vstack((0,0,1))
        k_tau = env.plant.c
        gamma = 1.00e-3 # arbitrary
        rho = 0 # thrust ratio between f_bar_2 and f_bar_1
        
        self.f_bar_sigma = m*g/float(n_bar[2])
        self.f_bar_1 = self.f_bar_sigma / (2+rho)
        self.f_bar_2 = rho*self.f_bar_1
        self.f_bar_3 = self.f_bar_1
        self.f_bar_4 = 0 # rotor 4 failure
        
        r_bar = k_tau/gamma*(self.f_bar_1-self.f_bar_2+self.f_bar_3-self.f_bar_4)
        
        A = np.array(
            [
                [0, -(Izz-Iyy)/Ixx*r_bar, 0, 0],
                [(Izz-Ixx)/Iyy*r_bar, 0, 0 ,0],
                [0, -float(n_bar[2]), 0, r_bar],
                [float(n_bar[2]), 0, r_bar, 0],
            ]
        )
        
        B = l/Ixx * np.array(
            [
                [1, 0],
                [0, 1],
                [0, 0],
                [0, 0],
            ]
        )
        
        Q = np.diag((10,10,10,10))
        R = np.diag((1,1))
        
        self.Katt, *_ = control.lqr(A,B,Q,R)
        

    def get_control(self, t, env):
        pos, vel, quat, omega = env.plant.observe_list()
        
        # extract p,q,r values
        p = omega[0]
        q = omega[1]
        r = omega[2]
        omega_norm = np.linalg.norm(np.vstack((p,q,r)))
        
        n_x = p
        n_y = q
        n_z = r
        
        # normalize if n is not a unit vector
        if omega_norm != 0:
            n_x = n_x / omega_norm
            n_y = n_y / omega_norm
            n_z = n_z / omega_norm
        
        # create s_tilde and s_tilde desired
        s_tilde = np.vstack((p, q, n_x, n_y))
        s_tilde_des = np.zeros((4,1))
        
        # obtain control inputs using control gains from LQR controller
        u_att = -self.Katt @ (s_tilde - s_tilde_des)
        forces = np.vstack((self.f_bar_sigma, u_att,0)) # [total thrust to be generated; u_1; u_2] (3x1)

        # distribute corresponding thrust values for each motor
        # ?????? 4번 로터에 추력제한을 걸어야 함! HOW??????
        ctrls = np.linalg.pinv(env.plant.mixer.B) @ forces
        ctrls = ctrls + np.vstack((self.f_bar_1, self.f_bar_2, self.f_bar_3, self.f_bar_4))
        
        # print("==============")
        # print(ctrls)
        # print("==============")
                
        controller_info = {
            # "angd": angd,
            # "n": n,
            "omega": omega,
            "s_tilde": s_tilde,
            "s_tilde_des": s_tilde_des
        }

        return ctrls, controller_info
