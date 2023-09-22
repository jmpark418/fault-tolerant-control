import argparse

import fym
import matplotlib.pyplot as plt
import numpy as np

import ftc
from ftc.models.multicopter import Multicopter
from ftc.utils import safeupdate
from fym.utils.rot import angle2quat

np.seterr(all="raise")

class MyEnv(fym.BaseEnv):
    ang = np.deg2rad((0, 0, 0))
    
    ENV_CONFIG = {
        "fkw": {
            "dt": 0.01,
            "max_t": 20,
        },
        "plant": {
            "init": {
                "pos": np.vstack((1, 2, 2)),
                "vel": np.vstack((0, 0, 0)),
                "quat": angle2quat(ang[2], ang[1], ang[0]),
                # "quat": np.vstack((1, 0, 0, 0)),
                "omega": np.vstack((0, 0, 0)),
            },
        },
    }

    def __init__(self, env_config={}):
        env_config = safeupdate(self.ENV_CONFIG, env_config)
        super().__init__(**env_config["fkw"])
        self.plant = Multicopter(env_config["plant"], rtype = "quad")
        self.controller = ftc.make("Quater", self)

    def step(self):
        env_info, done = self.update()
        return done, env_info

    def observation(self):
        return self.observe_flat()

    def get_ref(self, t, *args):
        posd = np.vstack((0, 0, 0))
        posd_dot = np.vstack((0, 0, 0))
        refs = {"posd": posd, "posd_dot": posd_dot}
        return [refs[key] for key in args]

    def set_dot(self, t):
                
        ctrl, controller_info = self.controller.get_control(t,self)
        
        self.plant.set_dot(t, ctrl)
        env_info = {
            "t": t,
            **self.observe_dict(),
            **controller_info,
            "ctrl": ctrl,
            # "forces": forces,
            # "rotors0": rotors0,
            # "rotors": rotors,
        }

        return env_info


def run():
    env = MyEnv()
    flogger = fym.Logger("data.h5")

    env.reset()
    try:
        while True:
            env.render()

            done, env_info = env.step()
            flogger.record(env=env_info)

            if done:
                break

    finally:
        flogger.close()
        plot()

def plot():
    data = fym.load("data.h5")["env"]

    """ Figure 1 - States """
    fig, axes = plt.subplots(3, 4, figsize=(18, 5), squeeze=False, sharex=True)

    """ Column 1 - States: Position """
    ax = axes[0, 0]
    ax.plot(data["t"], data["plant"]["pos"][:, 0].squeeze(-1), "k-")
    ax.plot(data["t"], data["posd"][:, 0].squeeze(-1), "r--")
    ax.set_ylabel(r"$x$, m")
    # ax.legend(["Response", "Ref"], loc="upper right")
    ax.set_xlim(data["t"][0], data["t"][-1])

    ax = axes[1, 0]
    ax.plot(data["t"], data["plant"]["pos"][:, 1].squeeze(-1), "k-")
    ax.plot(data["t"], data["posd"][:, 1].squeeze(-1), "r--")
    ax.set_ylabel(r"$y$, m")

    ax = axes[2, 0]
    ax.plot(data["t"], data["plant"]["pos"][:, 2].squeeze(-1), "k-")
    ax.plot(data["t"], data["posd"][:, 2].squeeze(-1), "r--")
    ax.set_ylabel(r"$z$, m")

    ax.set_xlabel("Time, sec")

    """ Column 2 - States: Velocity """
    ax = axes[0, 1]
    ax.plot(data["t"], data["plant"]["vel"][:, 0].squeeze(-1), "k-")
    ax.set_ylabel(r"$v_x$, m/s")
    # ax.legend(["Response", "Ref"], loc="upper right")

    ax = axes[1, 1]
    ax.plot(data["t"], data["plant"]["vel"][:, 1].squeeze(-1), "k-")
    ax.set_ylabel(r"$v_y$, m/s")

    ax = axes[2, 1]
    ax.plot(data["t"], data["plant"]["vel"][:, 2].squeeze(-1), "k-")
    ax.set_ylabel(r"$v_z$, m/s")

    ax.set_xlabel("Time, sec")

    """ Column 3 - States: Quternion angles """
    ax = axes[0, 2]
    ax.plot(data["t"], np.rad2deg(data["q_ypr"][:, 0].squeeze(-1)), "k-")
    ax.plot(data["t"], np.rad2deg(data["qd_ypr"][:, 0].squeeze(-1)), "r--")
    ax.set_ylabel(r"$\phi$, deg")
    # ax.legend(["Response", "Ref"], loc="upper right")

    ax = axes[1, 2]
    ax.plot(data["t"], np.rad2deg(data["q_ypr"][:, 1].squeeze(-1)), "k-")
    ax.plot(data["t"], np.rad2deg(data["qd_ypr"][:, 1].squeeze(-1)), "r--")
    ax.set_ylabel(r"$\theta$, deg")

    ax = axes[2, 2]
    ax.plot(data["t"], np.rad2deg(data["q_ypr"][:, 2].squeeze(-1)), "k-")
    ax.plot(data["t"], np.rad2deg(data["qd_ypr"][:, 2].squeeze(-1)), "r--")
    ax.set_ylabel(r"$\psi$, deg")

    ax.set_xlabel("Time, sec")

    """ Column 4 - States: Angular rates """
    ax = axes[0, 3]
    ax.plot(data["t"], np.rad2deg(data["plant"]["omega"][:, 0].squeeze(-1)), "k-")
    ax.set_ylabel(r"$p$, deg/s")
    ax.legend(["Response", "Ref"], loc="upper right")

    ax = axes[1, 3]
    ax.plot(data["t"], np.rad2deg(data["plant"]["omega"][:, 1].squeeze(-1)), "k-")
    ax.set_ylabel(r"$q$, deg/s")

    ax = axes[2, 3]
    ax.plot(data["t"], np.rad2deg(data["plant"]["omega"][:, 2].squeeze(-1)), "k-")
    ax.set_ylabel(r"$r$, deg/s")

    ax.set_xlabel("Time, sec")

    fig.tight_layout()
    fig.subplots_adjust(wspace=0.3)
    fig.align_ylabels(axes)

    # """ Figure 2 - Generalized forces """
    # fig, axs = plt.subplots(4, 1)
    # for i, _ylabel in enumerate(["F", "Mx", "My", "Mz"]):
    #     ax = axs[i]
    #     ax.plot(data["t"], data["forces"].squeeze(-1)[:, i], "k-", label="Response")
    #     # ax.plot(data["t"], data["forces0"].squeeze(-1)[:, i], "r--", label="Command")
    #     ax.grid()
    #     plt.setp(ax, ylabel=_ylabel)
    #     if i == 0:
    #         ax.legend(loc="upper right")
    # plt.gcf().supxlabel("Time, sec")
    # plt.gcf().supylabel("Generalized Forces")

    # fig.tight_layout()
    # fig.subplots_adjust(wspace=0.5)
    # fig.align_ylabels(axs)

    """ Figure 3 - Rotor forces """
    fig, axes = plt.subplots(2, 2)
    # ylabels = np.array((["R1", "R2"], ["R3", "R4"], ["R5", "R6"]))
    # for i, _ylabel in np.ndenumerate(ylabels):
    #     ax = axes[i]
    #     ax.plot(
    #         data["t"], data["ctrl"].squeeze(-1)[:, sum(i)], "k-", label="Response"
    #     )
    #     # ax.plot(
    #     #     data["t"], data["rotors0"].squeeze(-1)[:, sum(i)], "r--", label="Command"
    #     # )
    #     ax.grid()
    #     plt.setp(ax, ylabel=_ylabel)
    #     if i == (0, 1):
    #         ax.legend(loc="upper right")
    # plt.gcf().supxlabel("Time, sec")
    # plt.gcf().supylabel("Rotor Thrusts")

    # fig.tight_layout()
    # fig.subplots_adjust(wspace=0.5)
    # fig.align_ylabels(axes)
    
    ax = axes[0, 0]
    ax.plot(data["t"], data["ctrl"][:, 0], "k-")
    ax.set_ylabel("R1")

    ax = axes[0, 1]
    ax.plot(data["t"], data["ctrl"][:, 1], "k-")
    ax.set_ylabel("R2")

    ax = axes[1, 0]
    ax.plot(data["t"], data["ctrl"][:, 2], "k-")
    ax.set_ylabel("R3")
    
    ax = axes[1, 1]
    ax.plot(data["t"], data["ctrl"][:, 3], "k-")
    ax.set_ylabel("R4")

    # ax.set_xlabel("Time, sec")
    plt.gcf().supxlabel("Time, sec")
    plt.gcf().supylabel("Rotor Thrusts")
    
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.5)
    fig.align_ylabels(axes)


    """ Figure 4 - Quadcopter attitude """
    plt.figure()
    plt.plot(data["t"], data["q"][:, 0], "k-", label = 'q0')
    plt.plot(data["t"], data["q"][:, 1], "r-", label = 'q1')
    plt.plot(data["t"], data["q"][:, 2], "g-", label = 'q2')
    plt.plot(data["t"], data["q"][:, 3], "b-", label = 'q3')
    
    ax.legend(loc="upper right")
    
    plt.gcf().supxlabel("Time, sec")
    plt.gcf().supylabel("Quaternion")
    
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.5)
    fig.align_ylabels(axes)
    
    """ Figure 5 - LPF Filter """
    plt.figure()
    plt.plot(data["t"], data["theta"][:, 0], "k-", label = 'theta')
    plt.plot(data["t"], data["theta_f"][:, 0], "r-", label = 'theta_filtered')
    
    ax.legend(loc="upper right")
    plt.legend()
    
    plt.gcf().supxlabel("Time, sec")
    plt.gcf().supylabel(r"$\theta$, deg")
    
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.5)
    fig.align_ylabels(axes)

    plt.show()

def main(args):
    if args.only_plot:
        plot()
        return
    else:
        run()

        if args.plot:
            plot()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--plot", action="store_true")
    parser.add_argument("-P", "--only-plot", action="store_true")
    args = parser.parse_args()
    main(args)
