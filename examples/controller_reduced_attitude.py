import argparse

import fym
import matplotlib.pyplot as plt
import numpy as np
from fym.utils.rot import angle2quat

import ftc
from ftc.models.multicopter import Multicopter
from ftc.utils import safeupdate

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
                "pos": np.vstack((5, 5, 10)),
                "vel": np.zeros((3, 1)),
                "quat": angle2quat(ang[2], ang[1], ang[0]),
                "omega": np.vstack((0.05, 0.05, 0.05)),
            },
        },
    }

    def __init__(self, env_config={}):
        env_config = safeupdate(self.ENV_CONFIG, env_config)
        super().__init__(**env_config["fkw"])
        self.plant = Multicopter(env_config["plant"], rtype="quad")
        self.controller = ftc.make("Reduced-Attitude", self)

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
        ctrl, controller_info = self.controller.get_control(t, self)
        ctrl2 = self.set_Lambda(
            t, ctrl
        )  ###################################################
        self.plant.set_dot(t, ctrl2)

        env_info = {
            "t": t,
            **self.observe_dict(),
            **controller_info,
            "ctrl": ctrl2,
            # "forces": forces,
            # "rotors0": rotors0,
            # "rotors": rotors,
        }

        return env_info

    def get_Lambda(self, t):
        """Lambda function"""

        Lambda = np.ones(4)
        Lambda[-1] = 0
        return Lambda

    def set_Lambda(self, t, ctrls):
        Lambda = self.get_Lambda(t)
        ctrls = np.diag(Lambda) @ ctrls
        return ctrls


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
    fig, axes = plt.subplots(4, 3, figsize=(18, 5), squeeze=False, sharex=True)

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

    """ Column 3 - States: Euler angles """
    ax = axes[0, 2]
    ax.plot(data["t"], np.rad2deg(data["s_tilde"][:, 0].squeeze(-1)), "k-")
    ax.plot(data["t"], np.rad2deg(data["s_tilde_des"][:, 0].squeeze(-1)), "r--")
    ax.set_ylabel("p, deg/sec")
    # ax.legend(["Response", "Ref"], loc="upper right")

    ax = axes[1, 2]
    ax.plot(data["t"], np.rad2deg(data["s_tilde"][:, 1].squeeze(-1)), "k-")
    ax.plot(data["t"], np.rad2deg(data["s_tilde_des"][:, 1].squeeze(-1)), "r--")
    ax.set_ylabel("q, deg/sec")

    ax = axes[2, 2]
    ax.plot(data["t"], np.rad2deg(data["s_tilde"][:, 2].squeeze(-1)), "k-")
    ax.plot(data["t"], np.rad2deg(data["s_tilde_des"][:, 2].squeeze(-1)), "r--")
    ax.set_ylabel(r"$n_x$")

    ax = axes[3, 2]
    ax.plot(data["t"], np.rad2deg(data["s_tilde"][:, 3].squeeze(-1)), "k-")
    ax.plot(data["t"], np.rad2deg(data["s_tilde_des"][:, 3].squeeze(-1)), "r--")
    ax.set_ylabel(r"$n_y$")

    ax.set_xlabel("Time, sec")

    # """ Column 4 - States: Angular rates """
    # ax = axes[0, 1]
    # ax.plot(data["t"], np.rad2deg(data["plant"]["omega"][:, 0].squeeze(-1)), "k-")
    # ax.set_ylabel(r"$p$, deg/s")
    # ax.legend(["Response", "Ref"], loc="upper right")

    # ax = axes[1, 1]
    # ax.plot(data["t"], np.rad2deg(data["plant"]["omega"][:, 1].squeeze(-1)), "k-")
    # ax.set_ylabel(r"$q$, deg/s")

    # ax = axes[2, 1]
    # ax.plot(data["t"], np.rad2deg(data["plant"]["omega"][:, 2].squeeze(-1)), "k-")
    # ax.set_ylabel(r"$r$, deg/s")

    # ax.set_xlabel("Time, sec")

    fig.tight_layout()
    fig.subplots_adjust(wspace=0.3)
    fig.align_ylabels(axes)

    """ Figure 2 - Rotor forces """
    fig, axes = plt.subplots(2, 2)

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
