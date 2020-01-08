from typing import Tuple
import numpy as np
from abc import ABCMeta, abstractmethod
from riemann_solvers.godunov import GodunovRiemannSolver
from constant import SpaceScheme


class ConvectiveFlux(metaclass=ABCMeta):

    @classmethod
    @abstractmethod
    def compute(cls, ls: np.ndarray, rs: np.ndarray, k: float, dt: float) -> Tuple[float, float, float]:
        pass

# for flux vector splitting schemes - F_{i+1/2} = F^{+}_{U_l} + F^{-}_{U_r}


class VanLeer(ConvectiveFlux):

    @classmethod
    def compute(cls, ls: np.ndarray, rs: np.ndarray, k: float, dt: float) -> Tuple[float, float, float]:
        a_l = (ls[2] * k / ls[0]) ** 0.5
        a_r = (rs[2] * k / rs[0]) ** 0.5
        rho_l = ls[0]
        rho_r = rs[0]
        M_l = ls[1] / a_l
        M_r = rs[1] / a_r

        f_p0 = 0.25 * rho_l * a_l * (1 + M_l)**2
        f_p1 = f_p0 * 2 * a_l / k * ((k - 1) / 2 * M_l + 1)
        f_p2 = f_p0 * 2 * a_l**2 / (k**2 - 1) * ((k - 1) / 2 * M_l + 1)**2

        f_m0 = -0.25 * rho_r * a_r * (1 - M_r) ** 2
        f_m1 = f_m0 * 2 * a_r / k * ((k - 1) / 2 * M_r - 1)
        f_m2 = f_m0 * 2 * a_r ** 2 / (k ** 2 - 1) * ((k - 1) / 2 * M_r - 1) ** 2

        f0 = (f_p0 + f_m0)
        f1 = (f_p1 + f_m1)
        f2 = (f_p2 + f_m2)

        return f0, f1, f2


class StegerWarming(ConvectiveFlux):

    @classmethod
    def compute(cls, ls: np.ndarray, rs: np.ndarray, k: float, dt: float) -> Tuple[float, float, float]:
        a_l = (ls[2] * k / ls[0]) ** 0.5
        a_r = (rs[2] * k / rs[0]) ** 0.5
        rho_l = ls[0]
        rho_r = rs[0]
        u_l = ls[1]
        u_r = rs[1]
        H_l = 0.5 * u_l**2 + a_l**2 / (k - 1)
        H_r = 0.5 * u_r**2 + a_r**2 / (k - 1)

        lam0_l = u_l - a_l
        lam1_l = u_l
        lam2_l = u_l + a_l

        lam0_r = u_r - a_r
        lam1_r = u_r
        lam2_r = u_r + a_r

        lam_p0 = 0.5 * (lam0_l + abs(lam0_l))
        lam_p1 = 0.5 * (lam1_l + abs(lam1_l))
        lam_p2 = 0.5 * (lam2_l + abs(lam2_l))

        lam_m0 = 0.5 * (lam0_r - abs(lam0_r))
        lam_m1 = 0.5 * (lam1_r - abs(lam1_r))
        lam_m2 = 0.5 * (lam2_r - abs(lam2_r))

        f_p0 = rho_l / (2 * k) * (lam_p0 + 2 * (k - 1) * lam_p1 + lam_p2)
        f_p1 = rho_l / (2 * k) * (
                (u_l - a_l) * lam_p0 + 2 * (k - 1) * u_l * lam_p1 + (u_l + a_l) * lam_p2
        )
        f_p2 = rho_l / (2 * k) * (
                (H_l - u_l * a_l) * lam_p0 + (k - 1) * u_l**2 * lam_p1 + (H_l + u_l * a_l) * lam_p2
        )

        f_m0 = rho_r / (2 * k) * (lam_m0 + 2 * (k - 1) * lam_m1 + lam_m2)
        f_m1 = rho_r / (2 * k) * (
                (u_r - a_r) * lam_m0 + 2 * (k - 1) * u_r * lam_m1 + (u_r + a_r) * lam_m2
        )
        f_m2 = rho_r / (2 * k) * (
                (H_r - u_r * a_r) * lam_m0 + (k - 1) * u_r ** 2 * lam_m1 + (H_r + u_r * a_r) * lam_m2
        )

        f0 = (f_p0 + f_m0)
        f1 = (f_p1 + f_m1)
        f2 = (f_p2 + f_m2)

        return f0, f1, f2


class Godunov(ConvectiveFlux):

    @classmethod
    def compute(cls, ls: np.ndarray, rs: np.ndarray, k: float, dt: float) -> Tuple[float, float, float]:
        godunov_solver = GodunovRiemannSolver(
            rho_l=ls[0], u_l=ls[1], p_l=ls[2],
            rho_r=rs[0], u_r=rs[1], p_r=rs[2],
            p_star_init_type='PV', k=k
        )
        godunov_solver.compute_star()
        rho, u, p = godunov_solver.get_point(0, dt)

        f0 = rho * u
        f1 = (rho * u * u + p)
        e = 0.5 * u**2 + p / ((k - 1) * rho)
        f2 = u * (rho * e + p)

        return f0, f1, f2


def get_conv_flux(space_scheme: SpaceScheme) -> ConvectiveFlux:
    if space_scheme == SpaceScheme.Godunov:
        return Godunov
    elif space_scheme == SpaceScheme.StegerWarming:
        return StegerWarming
    elif space_scheme == SpaceScheme.VanLeer:
        return VanLeer

