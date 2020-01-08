from scipy.optimize import newton, fsolve
from enum import Enum
import matplotlib.pyplot as plt
import numpy as np


class SolutionType(Enum):
    BOTH_SHOCK = 0
    BOTH_RARE = 1
    L_SHOCK_R_RARE = 2
    L_RARE_R_SHOCK = 3


class GodunovRiemannSolver:
    def __init__(self, rho_l, u_l, p_l, rho_r, u_r, p_r, p_star_init_type='TR', k=1.4, tol=1e-6):
        self.rho_l = rho_l
        self.rho_r = rho_r
        self.u_l = u_l
        self.u_r = u_r
        self.p_l = p_l
        self.p_r = p_r
        self.k = k
        self.tol = tol
        self.a_l = (k * p_l / rho_l) ** 0.5
        self.a_r = (k * p_r / rho_r) ** 0.5
        self.A_l = 2 / ((k + 1) * rho_l)
        self.A_r = 2 / ((k + 1) * rho_r)
        self.B_l = (k - 1) / (k + 1) * p_l
        self.B_r = (k - 1) / (k + 1) * p_r
        self.p_star_init_type = p_star_init_type
        self.p_star_init = None
        self.rho_star_l = None
        self.rho_star_r = None
        self.a_star_l = None
        self.a_star_r = None
        self.u_star = None
        self.p_star = None
        self.S_l = None
        self.S_r = None
        self.S_tl = None
        self.S_tr = None
        self.S_hl = None
        self.S_hr = None
        self._sol_type: SolutionType = None

    @property
    def sol_type(self) -> SolutionType:
        return self._sol_type

    def compute_p_star_init(self):
        if self.p_star_init_type == 'TR':
            self.p_star_init = (
                (self.a_l + self.a_r - 0.5 * (self.k - 1) * (self.u_r - self.u_l)) /
                (self.a_l / self.p_l ** ((self.k - 1) / (2 * self.k)) +
                 self.a_r / self.p_r ** ((self.k - 1) / (2 * self.k)))
            ) ** (2 * self.k / (self.k - 1))
        elif self.p_star_init_type == 'PV':
            p_pv = (
                0.5 * (self.p_l + self.p_r) -
                (self.u_r - self.u_l) * (self.rho_l + self.rho_r) * (self.a_l + self.a_r) / 8
            )
            self.p_star_init = max(self.tol, p_pv)
        elif self.p_star_init_type == 'MV':
            self.p_star_init = 0.5 * (self.p_r + self.p_l)

    def _set_solution_type(self):
        if self.p_star <= self.p_l and self.p_star <= self.p_r:
            self._sol_type = SolutionType.BOTH_RARE
        elif self.p_star > self.p_l and self.p_star > self.p_r:
            self._sol_type = SolutionType.BOTH_SHOCK
        elif self.p_l >= self.p_star > self.p_r:
            self._sol_type = SolutionType.L_RARE_R_SHOCK
        elif self.p_l < self.p_star <= self.p_r:
            self._sol_type = SolutionType.L_SHOCK_R_RARE

    def f_left(self, p):
        if p > self.p_l:
            return (p - self.p_l) * (self.A_l / (p + self.B_l)) ** 0.5
        else:
            return 2 * self.a_l / (self.k - 1) * (
                    (p / self.p_l) ** ((self.k - 1) / (2 * self.k)) - 1
            )

    def f_left_d(self, p):
        if p > self.p_l:
            return (self.A_l / (p + self.B_l))**0.5 - \
                   0.5 * p * self.A_l * (self.A_l / (p + self.B_l))**-0.5 / (p + self.B_l)**2 + \
                   0.5 * self.p_l * self.A_l * (self.A_l / (p + self.B_l))**-0.5 / (p + self.B_l)**2
        else:
            return self.a_l / (self.p_l * self.k) * (p / self.p_l) ** (-(self.k + 1) / (2 * self.k))

    def f_right(self, p):
        if p > self.p_r:
            return (p - self.p_r) * (self.A_r / (p + self.B_r)) ** 0.5
        else:
            return 2 * self.a_r / (self.k - 1) * (
                    (p / self.p_r) ** ((self.k - 1) / (2 * self.k)) - 1
            )

    def f_right_d(self, p):
        if p > self.p_r:
            return (self.A_r / (p + self.B_r))**0.5 - \
                   0.5 * p * self.A_r * (self.A_r / (p + self.B_r))**-0.5 / (p + self.B_r)**2 + \
                   0.5 * self.p_r * self.A_r * (self.A_r / (p + self.B_r))**-0.5 / (p + self.B_r)**2
        else:
            return self.a_r / (self.p_r * self.k) * (p / self.p_r) ** (-(self.k + 1) / (2 * self.k))

    def f(self, p):
        return self.f_left(p) + self.f_right(p) + self.u_r - self.u_l

    def f_d(self, p):
        return self.f_left_d(p) + self.f_right_d(p)

    def compute_star(self):
        self.compute_p_star_init()
        # self.p_star = newton(self.f, self.p_star_init, fprime=self.f_d, tol=self.tol)
        self.p_star = fsolve(lambda x: [self.f(x[0])], np.array([self.p_star_init]), xtol=self.tol)[0]
        self.u_star = 0.5 * (self.u_l + self.u_r) + 0.5 * (self.f_right(self.p_star) - self.f_left(self.p_star))
        self._set_solution_type()
        if self._sol_type == SolutionType.BOTH_SHOCK:
            self.rho_star_l = self.rho_l * (
                (self.p_star / self.p_l + (self.k - 1) / (self.k + 1)) /
                ((self.k - 1) / (self.k + 1) * self.p_star / self.p_l + 1)
            )
            self.a_star_l = (self.k * self.p_star / self.rho_star_l) ** 0.5
            self.S_l = self.u_l - self.a_l * (
                (self.k + 1) / (2 * self.k) * self.p_star / self.p_l + (self.k - 1) / (2 * self.k)
            )**0.5

            self.rho_star_r = self.rho_r * (
                (self.p_star / self.p_r + (self.k - 1) / (self.k + 1)) /
                ((self.k - 1) / (self.k + 1) * self.p_star / self.p_r + 1)
            )
            self.a_star_r = (self.k * self.p_star / self.rho_star_r) ** 0.5
            self.S_r = self.u_r + self.a_r * (
                    (self.k + 1) / (2 * self.k) * self.p_star / self.p_r + (self.k - 1) / (2 * self.k)
            ) ** 0.5

        elif self._sol_type == SolutionType.BOTH_RARE:
            self.rho_star_l = self.rho_l * (self.p_star / self.p_l) ** (1 / self.k)
            self.a_star_l = self.a_l * (self.p_star / self.p_l) ** ((self.k - 1) / (2 * self.k))
            self.S_hl = self.u_l - self.a_l
            self.S_tl = self.u_star - self.a_star_l

            self.rho_star_r = self.rho_r * (self.p_star / self.p_r) ** (1 / self.k)
            self.a_star_r = self.a_r * (self.p_star / self.p_r) ** ((self.k - 1) / (2 * self.k))
            self.S_hr = self.u_r + self.a_r
            self.S_tr = self.u_star + self.a_star_r

        elif self._sol_type == SolutionType.L_SHOCK_R_RARE:
            self.rho_star_l = self.rho_l * (
                    (self.p_star / self.p_l + (self.k - 1) / (self.k + 1)) /
                    ((self.k - 1) / (self.k + 1) * self.p_star / self.p_l + 1)
            )
            self.a_star_l = (self.k * self.p_star / self.rho_star_l) ** 0.5
            self.S_l = self.u_l - self.a_l * (
                    (self.k + 1) / (2 * self.k) * self.p_star / self.p_l + (self.k - 1) / (2 * self.k)
            ) ** 0.5

            self.rho_star_r = self.rho_r * (self.p_star / self.p_r) ** (1 / self.k)
            self.a_star_r = self.a_r * (self.p_star / self.p_r) ** ((self.k - 1) / (2 * self.k))
            self.S_hr = self.u_r + self.a_r
            self.S_tr = self.u_star + self.a_star_r

        elif self._sol_type == SolutionType.L_RARE_R_SHOCK:
            self.rho_star_l = self.rho_l * (self.p_star / self.p_l) ** (1 / self.k)
            self.a_star_l = self.a_l * (self.p_star / self.p_l) ** ((self.k - 1) / (2 * self.k))
            self.S_hl = self.u_l - self.a_l
            self.S_tl = self.u_star - self.a_star_l

            self.rho_star_r = self.rho_r * (
                    (self.p_star / self.p_r + (self.k - 1) / (self.k + 1)) /
                    ((self.k - 1) / (self.k + 1) * self.p_star / self.p_r + 1)
            )
            self.a_star_r = (self.k * self.p_star / self.rho_star_r) ** 0.5
            self.S_r = self.u_r + self.a_r * (
                    (self.k + 1) / (2 * self.k) * self.p_star / self.p_r + (self.k - 1) / (2 * self.k)
            ) ** 0.5

    def get_point(self, x, t):
        s = x / t
        if self._sol_type == SolutionType.BOTH_SHOCK:
            if s < self.S_l:
                return self.rho_l, self.u_l, self.p_l
            elif self.S_l <= s < self.u_star:
                return self.rho_star_l, self.u_star, self.p_star
            elif self.u_star <= s < self.S_r:
                return self.rho_star_r, self.u_star, self.p_star
            elif self.S_r <= s:
                return self.rho_r, self.u_r, self.p_r

        elif self._sol_type == SolutionType.BOTH_RARE:
            if s < self.S_hl:
                return self.rho_l, self.u_l, self.p_l
            elif self.S_hl <= s < self.S_tl:
                rho = self.rho_l * (
                        2 / (self.k + 1) + (self.k - 1) / ((self.k + 1) * self.a_l) * (self.u_l - x / t)
                ) ** (2 / (self.k - 1))
                u = 2 / (self.k + 1) * (self.a_l + (self.k - 1) / 2 * self.u_l + x / t)
                p = self.p_l * (
                    2 / (self.k + 1) + (self.k - 1) / ((self.k + 1) * self.a_l) * (self.u_l - x / t)
                ) ** (2 * self.k / (self.k - 1))
                return rho, u, p
            elif self.S_tl <= s < self.u_star:
                return self.rho_star_l, self.u_star, self.p_star
            elif self.u_star <= s < self.S_tr:
                return self.rho_star_r, self.u_star, self.p_star
            elif self.S_tr <= s < self.S_hr:
                rho = self.rho_r * (
                        2 / (self.k + 1) - (self.k - 1) / ((self.k + 1) * self.a_r) * (self.u_r - x / t)
                ) ** (2 / (self.k - 1))
                u = 2 / (self.k + 1) * (-self.a_r + (self.k - 1) / 2 * self.u_r + x / t)
                p = self.p_r * (
                        2 / (self.k + 1) - (self.k - 1) / ((self.k + 1) * self.a_r) * (self.u_r - x / t)
                ) ** (2 * self.k / (self.k - 1))
                return rho, u, p
            elif self.S_hr <= s:
                return self.rho_r, self.u_r, self.p_r

        elif self._sol_type == SolutionType.L_SHOCK_R_RARE:
            if s < self.S_l:
                return self.rho_l, self.u_l, self.p_l
            elif self.S_l <= s < self.u_star:
                return self.rho_star_l, self.u_star, self.p_star
            elif self.u_star <= s < self.S_tr:
                return self.rho_star_r, self.u_star, self.p_star
            elif self.S_tr <= s < self.S_hr:
                rho = self.rho_r * (
                        2 / (self.k + 1) - (self.k - 1) / ((self.k + 1) * self.a_r) * (self.u_r - x / t)
                ) ** (2 / (self.k - 1))
                u = 2 / (self.k + 1) * (-self.a_r + (self.k - 1) / 2 * self.u_r + x / t)
                p = self.p_r * (
                        2 / (self.k + 1) - (self.k - 1) / ((self.k + 1) * self.a_r) * (self.u_r - x / t)
                ) ** (2 * self.k / (self.k - 1))
                return rho, u, p
            elif self.S_hr <= s:
                return self.rho_r, self.u_r, self.p_r

        elif self._sol_type == SolutionType.L_RARE_R_SHOCK:
            if s < self.S_hl:
                return self.rho_l, self.u_l, self.p_l
            elif self.S_hl <= s < self.S_tl:
                rho = self.rho_l * (
                        2 / (self.k + 1) + (self.k - 1) / ((self.k + 1) * self.a_l) * (self.u_l - x / t)
                ) ** (2 / (self.k - 1))
                u = 2 / (self.k + 1) * (self.a_l + (self.k - 1) / 2 * self.u_l + x / t)
                p = self.p_l * (
                    2 / (self.k + 1) + (self.k - 1) / ((self.k + 1) * self.a_l) * (self.u_l - x / t)
                ) ** (2 * self.k / (self.k - 1))
                return rho, u, p
            elif self.S_tl <= s < self.u_star:
                return self.rho_star_l, self.u_star, self.p_star
            elif self.u_star <= s < self.S_r:
                return self.rho_star_r, self.u_star, self.p_star
            elif self.S_r <= s:
                return self.rho_r, self.u_r, self.p_r

    def plot_time_level(self, t, xlim=(-1, 1), figsize=(7, 6), num_pnt = 100):
        x_arr = np.linspace(xlim[0], xlim[1], num_pnt)
        rho_arr, u_arr, p_arr = [], [], []
        for x in x_arr:
            solution = self.get_point(x, t)
            rho_arr.append(solution[0])
            u_arr.append(solution[1])
            p_arr.append(solution[2])

        plt.figure(figsize=figsize)
        plt.title('t = %.3f' % t, fontsize=16)
        plt.plot(x_arr, rho_arr, lw=2, color='red')
        plt.grid()
        plt.xlim(xlim)
        plt.xlabel(r'$x$', fontsize=14)
        plt.ylabel(r'$\rho$', fontsize=14)
        plt.show()

        plt.figure(figsize=figsize)
        plt.title('t = %.3f' % t, fontsize=16)
        plt.plot(x_arr, u_arr, lw=2, color='red')
        plt.grid()
        plt.xlim(xlim)
        plt.xlabel(r'$x$', fontsize=14)
        plt.ylabel(r'$u$', fontsize=14)
        plt.show()

        plt.figure(figsize=figsize)
        plt.title('t = %.3f' % t, fontsize=16)
        plt.plot(x_arr, p_arr, lw=2, color='red')
        plt.grid()
        plt.xlim(xlim)
        plt.xlabel(r'$x$', fontsize=14)
        plt.ylabel(r'$p$', fontsize=14)
        plt.show()


