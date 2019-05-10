import typing
import gdf
import numpy as np
from scipy.optimize import newton, fsolve
import matplotlib.pyplot as plt
import matplotlib.figure
import matplotlib.axes


class LavalNozzleSolver:
    def __init__(
            self,
            T_stag, p1_stag, p2,
            area: typing.Callable[[float], float],
            x1, x2,
            k=1.4, R=287.,
            num=200,
            lam_pre_init=1.5
    ):
        self.T_stag = T_stag
        self.p1_stag = p1_stag
        self.p2 = p2
        self.k = k
        self.R = R
        self.area = area
        self.x1 = x1
        self.x2 = x2
        self.num = num
        self.lam_pre_init = lam_pre_init
        self.x_arr = np.linspace(x1, x2, num)
        self.area_arr = np.array([area(x) for x in self.x_arr])
        self.q_arr = np.zeros(num)
        self.lam_arr = np.zeros(num)
        self.lam_rel_arr = np.zeros(num)
        self.T_arr = np.zeros(num)
        self.T_rel_arr = np.zeros(num)
        self.p_arr = np.zeros(num)
        self.p_rel_arr = np.zeros(num)
        self.c_arr = np.zeros(num)
        self.c_rel_arr = np.zeros(num)
        self.p_stag_arr = np.zeros(num)
        self.p_stag_rel_arr = np.zeros(num)
        self.p2_nom = None
        self.lam2 = None
        self.p2_stag = None
        self.lam_pre = None
        self.lam_post = None
        self.area_shock = None

    def _compute_nominal(self):
        # индекс критического сечения
        self.i_cr = list(self.area_arr).index(np.min(self.area_arr))
        self.a_cr = gdf.a_cr(self.T_stag, self.k, self.R)
        self.area_cr = self.area_arr[self.i_cr]
        self.G = gdf.m(self.k) * self.p1_stag * self.area_cr / (self.R * self.T_stag)**0.5
        self.area2 = self.area_arr[self.num - 1]

        for i in range(self.num):
            self.p_stag_arr[i] = self.p1_stag
            if i < self.i_cr:
                self.q_arr[i] = self.G * (self.R * self.T_stag)**0.5 / \
                                (self.p_stag_arr[i] * self.area_arr[i] * gdf.m(self.k))
                self.lam_arr[i] = gdf.lam(self.k, q=self.q_arr[i], kind='subs')
                self.c_arr[i] = self.a_cr * self.lam_arr[i]
                self.T_arr[i] = self.T_stag * gdf.tau_lam(self.lam_arr[i], self.k)
                self.p_arr[i] = self.p_stag_arr[i] * gdf.pi_lam(self.lam_arr[i], self.k)
            elif i == self.i_cr:
                self.q_arr[i] = 1
                self.lam_arr[i] = 1
                self.c_arr[i] = self.a_cr
                self.T_arr[i] = self.T_stag * gdf.tau_lam(1, self.k)
                self.p_arr[i] = self.p_stag_arr[i] * gdf.pi_lam(1, self.k)
            else:
                self.q_arr[i] = self.G * (self.R * self.T_stag) ** 0.5 / \
                                (self.p_stag_arr[i] * self.area_arr[i] * gdf.m(self.k))
                self.lam_arr[i] = gdf.lam(self.k, q=self.q_arr[i], kind='supers')
                self.c_arr[i] = self.a_cr * self.lam_arr[i]
                self.T_arr[i] = self.T_stag * gdf.tau_lam(self.lam_arr[i], self.k)
                self.p_arr[i] = self.p_stag_arr[i] * gdf.pi_lam(self.lam_arr[i], self.k)
        self.p2_nom = self.p_arr[self.num - 1]

    def compute(self):
        self._compute_nominal()
        if self.p2 > self.p2_nom:
            # Режим перерасширения
            self.lam2 = newton(
                lambda lam: (
                        self.G - gdf.m(self.k) * self.area2 / (self.R * self.T_stag)**0.5 *
                        gdf.q(lam, self.k) * self.p2 / gdf.pi_lam(lam, self.k)
                ),
                x0=0.6
            )
            if self.lam2 < 1:
                self.p2_stag = self.p2 / gdf.pi_lam(self.lam2, self.k)
                self.lam_pre = newton(
                    lambda lam: self.p1_stag * gdf.q(lam, self.k) - self.p2_stag * gdf.q(1 / lam, self.k),
                    x0=self.lam_pre_init
                )
                self.lam_post = 1 / self.lam_pre
                self.area_shock = self.area_cr * gdf.q(1, self.k) / gdf.q(self.lam_pre, self.k)
                for i in range(self.i_cr, self.num):
                    if self.area_arr[i] > self.area_shock:
                        # Скачок уплотнения внутри сопла
                        self.p_stag_arr[i] = self.p2_stag
                        self.q_arr[i] = self.G * (self.R * self.T_stag) ** 0.5 / \
                                        (self.p_stag_arr[i] * self.area_arr[i] * gdf.m(self.k))
                        self.lam_arr[i] = gdf.lam(self.k, q=self.q_arr[i], kind='subs')
                        self.c_arr[i] = self.a_cr * self.lam_arr[i]
                        self.T_arr[i] = self.T_stag * gdf.tau_lam(self.lam_arr[i], self.k)
                        self.p_arr[i] = self.p_stag_arr[i] * gdf.pi_lam(self.lam_arr[i], self.k)
        self.T_rel_arr = self.T_arr / self.T_arr[0]
        self.c_rel_arr = self.c_arr / self.c_arr[0]
        self.lam_rel_arr = self.lam_arr / self.lam_arr[0]
        self.p_stag_rel_arr = self.p_stag_arr / self.p_stag_arr[0]
        self.p_rel_arr = self.p_arr / self.p_arr[0]

    @classmethod
    def make_patch_spines_invisible(cls, ax):
        ax.set_frame_on(True)
        ax.patch.set_visible(False)
        for sp in ax.spines.values():
            sp.set_visible(False)

    def plot(self, figsize=(10, 6)):
        subplots = plt.subplots()
        fig: matplotlib.figure.Figure = subplots[0]
        fig.set_size_inches(figsize[0], figsize[1])

        host: matplotlib.axes.Axes = subplots[1]
        fig.subplots_adjust(right=0.68)
        par1: matplotlib.axes.Axes = host.twinx()
        par2: matplotlib.axes.Axes = host.twinx()
        par3: matplotlib.axes.Axes = host.twinx()

        # Offset the right spine of par2.  The ticks and label have already been
        # placed on the right by twinx above.
        par2.spines["right"].set_position(("axes", 1.2))
        par3.spines["right"].set_position(("axes", 1.4))
        # Having been created by twinx, par2 has its frame off, so the line of its
        # detached spine is invisible.  First, activate the frame but make the patch
        # and spines invisible.
        self.make_patch_spines_invisible(par2)
        self.make_patch_spines_invisible(par3)
        # Second, show the right spine.
        par2.spines["right"].set_visible(True)
        par3.spines["right"].set_visible(True)

        p1, = host.plot(self.x_arr, self.p_arr / 1e6, "b", label=r"$p$")
        p2, = par1.plot(self.x_arr, self.T_arr, "r", label=r"$T$")
        p3, = par2.plot(self.x_arr, self.lam_arr, "g", label=r"$\lambda$")
        p4, = par3.plot(self.x_arr, self.p_stag_arr / 1e6, "black", label=r"$p^*$")

        host.set_xlim(self.x_arr.min(), self.x_arr.max())
        host.set_ylim(bottom=0)
        par1.set_ylim(bottom=273)
        par2.set_ylim(bottom=0)
        par3.set_ylim(bottom=0)

        host.set_xlabel(r"$x,\ м$", fontsize=14)
        host.set_ylabel(r"$p,\ МПа$", fontsize=14)
        par1.set_ylabel(r"$T,\ К$", fontsize=14)
        par2.set_ylabel(r"$\lambda$", fontsize=14)
        par3.set_ylabel(r"$p^*,\ Мпа$", fontsize=14)

        host.yaxis.label.set_color(p1.get_color())
        par1.yaxis.label.set_color(p2.get_color())
        par2.yaxis.label.set_color(p3.get_color())
        par3.yaxis.label.set_color(p4.get_color())

        tkw = dict(size=4, width=1.5)
        host.tick_params(axis='y', colors=p1.get_color(), **tkw)
        par1.tick_params(axis='y', colors=p2.get_color(), **tkw)
        par2.tick_params(axis='y', colors=p3.get_color(), **tkw)
        par3.tick_params(axis='y', colors=p4.get_color(), **tkw)
        host.tick_params(axis='x', **tkw)
        host.grid(True)

        lines = [p1, p2, p3, p4]

        host.legend(lines, [l.get_label() for l in lines], fontsize=12, loc=3)

        plt.show()
