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


class HLLC(ConvectiveFlux):

    @classmethod
    def compute(cls, ls: np.ndarray, rs: np.ndarray, k: float, dt: float) -> Tuple[float, float, float]:
        a_l = (ls[2] * k / ls[0]) ** 0.5
        a_r = (rs[2] * k / rs[0]) ** 0.5
        rho_l = ls[0]
        rho_r = rs[0]
        u_l = ls[1]
        u_r = rs[1]
        p_l = ls[2]
        p_r = rs[2]

        rho_av = 0.5 * (rho_r + rho_l)
        a_av = 0.5 * (a_r + a_l)
        p_pvrs = 0.5 * (p_r + p_l) - 0.5 * (u_r + u_l) * rho_av * a_av
        p_star = max(0, p_pvrs)

        q_l = 1
        if p_star > p_l:
            q_l = (1 + (k + 1) / (2 * k) * (p_star / p_l - 1))**0.5

        q_r = 1
        if p_star > p_l:
            q_r = (1 + (k + 1) / (2 * k) * (p_star / p_r - 1))**0.5

        s_l = u_l - a_l * q_l
        s_r = u_r + a_r * q_r

        s_star = (p_r - p_l + rho_l * u_l * (s_l - u_l) - rho_r * u_r * (s_r - u_r)) / \
                 (rho_l * (s_l - u_l) - rho_r * (s_r - u_r))

        e_l = p_l / (rho_l * (k - 1))
        e_r = p_r / (rho_r * (k - 1))

        f_l0 = rho_l * u_l
        f_l1 = rho_l * u_l**2 + p_l
        f_l2 = (rho_l * (e_l + 0.5 * u_l**2) + p_l) * u_l

        f_r0 = rho_r * u_r
        f_r1 = rho_r * u_r**2 + p_r
        f_r2 = (rho_r * (e_r + 0.5 * u_r**2) + p_r) * u_r

        u_l0 = rho_l
        u_l1 = rho_l * u_l
        u_l2 = rho_l * (e_l + 0.5 * u_l**2)

        u_r0 = rho_r
        u_r1 = rho_r * u_r
        u_r2 = rho_r * (e_r + 0.5 * u_r**2)

        u_l0_star = rho_l * (s_l - u_l) / (s_l - s_star)
        u_l1_star = u_l0_star * s_star
        u_l2_star = u_l0_star * ((e_l + 0.5 * u_l**2) + (s_star - u_l) * (s_star + p_l / (rho_l * (s_l - u_l))))

        u_r0_star = rho_r * (s_r - u_r) / (s_r - s_star)
        u_r1_star = u_r0_star * s_star
        u_r2_star = u_r0_star * ((e_r + 0.5 * u_r**2) + (s_star - u_r) * (s_star + p_r / (rho_r * (s_r - u_r))))

        f_l0_star = f_l0 + s_l * (u_l0_star - u_l0)
        f_l1_star = f_l1 + s_l * (u_l1_star - u_l1)
        f_l2_star = f_l2 + s_l * (u_l2_star - u_l2)

        f_r0_star = f_r0 + s_r * (u_r0_star - u_r0)
        f_r1_star = f_r1 + s_r * (u_r1_star - u_r1)
        f_r2_star = f_r2 + s_r * (u_r2_star - u_r2)

        f0 = f_l0 * (0 <= s_l) + f_l0_star * (s_l < 0 <= s_star) + f_r0_star * (s_star < 0 <= s_r) + f_r0 * (s_r < 0)
        f1 = f_l1 * (0 <= s_l) + f_l1_star * (s_l < 0 <= s_star) + f_r1_star * (s_star < 0 <= s_r) + f_r1 * (s_r < 0)
        f2 = f_l2 * (0 <= s_l) + f_l2_star * (s_l < 0 <= s_star) + f_r2_star * (s_star < 0 <= s_r) + f_r2 * (s_r < 0)

        return f0, f1, f2


class Roe(ConvectiveFlux):

    @classmethod
    def compute(cls, ls: np.ndarray, rs: np.ndarray, k: float, dt: float) -> Tuple[float, float, float]:
        a_l = (ls[2] * k / ls[0]) ** 0.5
        a_r = (rs[2] * k / rs[0]) ** 0.5
        rho_l = ls[0]
        rho_r = rs[0]
        u_l = ls[1]
        u_r = rs[1]
        p_l = ls[2]
        p_r = rs[2]

        e_l = p_l / (rho_l * (k - 1))
        e_r = p_r / (rho_r * (k - 1))
        h_l = (e_l + 0.5 * u_l**2) + p_l / rho_l
        h_r = (e_r + 0.5 * u_r**2) + p_r / rho_r

        f_l0 = rho_l * u_l
        f_l1 = rho_l * u_l**2 + p_l
        f_l2 = (rho_l * (e_l + 0.5 * u_l**2) + p_l) * u_l

        f_r0 = rho_r * u_r
        f_r1 = rho_r * u_r**2 + p_r
        f_r2 = (rho_r * (e_r + 0.5 * u_r**2) + p_r) * u_r

        u_av = (rho_l**0.5 * u_l + rho_r**0.5 * u_r) / (rho_l**0.5 + rho_r**0.5)
        h_av = (rho_l**0.5 * h_l + rho_r**0.5 * h_r) / (rho_l**0.5 + rho_r**0.5)
        a_av = ((k - 1) * (h_av - 0.5 * u_av**2))**0.5

        lam_0_av = u_av - a_av
        lam_1_av = u_av
        lam_2_av = u_av + a_av

        k00 = 1
        k10 = u_av - a_av
        k20 = h_av - u_av * a_av
        k01 = 1
        k11 = u_av
        k21 = 0.5 * u_av**2
        k02 = 1
        k12 = u_av + a_av
        k22 = h_av + u_av * a_av

        du0 = rho_r - rho_l
        du1 = u_r * rho_r - u_l * rho_l
        du2 = rho_r * (e_r + 0.5 * u_r**2) - rho_l * (e_l + 0.5 * u_l**2)

        alpha_1 = (k - 1) / a_av**2 * (du0 * (h_av - u_av**2) + u_av * du1 - du2)
        alpha_0 = 1 / (2 * a_av) * (du0 * (u_av + a_av) - du1 - a_av * alpha_1)
        alpha_2 = du0 - (alpha_0 + alpha_1)

        f0 = 0.5 * (f_l0 + f_r0) - 0.5 * (
                alpha_0 * abs(lam_0_av) * k00 + alpha_1 * abs(lam_1_av) * k01 + alpha_2 * abs(lam_2_av) * k02
        )
        f1 = 0.5 * (f_l1 + f_r1) - 0.5 * (
                alpha_0 * abs(lam_0_av) * k10 + alpha_1 * abs(lam_1_av) * k11 + alpha_2 * abs(lam_2_av) * k12
        )
        f2 = 0.5 * (f_l2 + f_r2) - 0.5 * (
                alpha_0 * abs(lam_0_av) * k20 + alpha_1 * abs(lam_1_av) * k21 + alpha_2 * abs(lam_2_av) * k22
        )

        return f0, f1, f2


class RoeEntropyFix(ConvectiveFlux):

    @classmethod
    def compute(cls, ls: np.ndarray, rs: np.ndarray, k: float, dt: float) -> Tuple[float, float, float]:
        a_l = (ls[2] * k / ls[0]) ** 0.5
        a_r = (rs[2] * k / rs[0]) ** 0.5
        rho_l = ls[0]
        rho_r = rs[0]
        u_l = ls[1]
        u_r = rs[1]
        p_l = ls[2]
        p_r = rs[2]

        e_l = p_l / (rho_l * (k - 1))
        e_r = p_r / (rho_r * (k - 1))
        h_l = (e_l + 0.5 * u_l**2) + p_l / rho_l
        h_r = (e_r + 0.5 * u_r**2) + p_r / rho_r

        f_l0 = rho_l * u_l
        f_l1 = rho_l * u_l**2 + p_l
        f_l2 = (rho_l * (e_l + 0.5 * u_l**2) + p_l) * u_l

        f_r0 = rho_r * u_r
        f_r1 = rho_r * u_r**2 + p_r
        f_r2 = (rho_r * (e_r + 0.5 * u_r**2) + p_r) * u_r

        u_roe_av = (rho_l**0.5 * u_l + rho_r**0.5 * u_r) / (rho_l**0.5 + rho_r**0.5)
        h_roe_av = (rho_l**0.5 * h_l + rho_r**0.5 * h_r) / (rho_l**0.5 + rho_r**0.5)
        a_roe_av = ((k - 1) * (h_roe_av - 0.5 * u_roe_av**2))**0.5

        lam_0_av = u_roe_av - a_roe_av
        lam_1_av = u_roe_av
        lam_2_av = u_roe_av + a_roe_av

        k00 = 1
        k10 = u_roe_av - a_roe_av
        k20 = h_roe_av - u_roe_av * a_roe_av
        k01 = 1
        k11 = u_roe_av
        k21 = 0.5 * u_roe_av**2
        k02 = 1
        k12 = u_roe_av + a_roe_av
        k22 = h_roe_av + u_roe_av * a_roe_av

        du0 = rho_r - rho_l
        du1 = u_r * rho_r - u_l * rho_l
        du2 = rho_r * (e_r + 0.5 * u_r**2) - rho_l * (e_l + 0.5 * u_l**2)

        alpha_1 = (k - 1) / a_roe_av**2 * (du0 * (h_roe_av - u_roe_av**2) + u_roe_av * du1 - du2)
        alpha_0 = 1 / (2 * a_roe_av) * (du0 * (u_roe_av + a_roe_av) - du1 - a_roe_av * alpha_1)
        alpha_2 = du0 - (alpha_0 + alpha_1)

        rho_av = 0.5 * (rho_l + rho_r)
        a_av = 0.5 * (a_l + a_r)
        p_star = 0.5 * (p_l + p_r) + 0.5 * (u_l - u_r) * (rho_av * a_av)
        u_star = 0.5 * (u_l + u_r) + 0.5 * (p_l - p_r) / (rho_av * a_av)
        rho_l_star = rho_l + (u_l - u_star) * rho_av / a_av
        rho_r_star = rho_r + (u_star - u_r) * rho_av / a_av
        a_l_star = (k * p_star / rho_l_star)**0.5
        a_r_star = (k * p_star / rho_r_star)**0.5

        f0 = 0.5 * (f_l0 + f_r0) - 0.5 * (
                alpha_0 * abs(lam_0_av) * k00 + alpha_1 * abs(lam_1_av) * k01 + alpha_2 * abs(lam_2_av) * k02
        )
        f1 = 0.5 * (f_l1 + f_r1) - 0.5 * (
                alpha_0 * abs(lam_0_av) * k10 + alpha_1 * abs(lam_1_av) * k11 + alpha_2 * abs(lam_2_av) * k12
        )
        f2 = 0.5 * (f_l2 + f_r2) - 0.5 * (
                alpha_0 * abs(lam_0_av) * k20 + alpha_1 * abs(lam_1_av) * k21 + alpha_2 * abs(lam_2_av) * k22
        )

        if p_star <= p_l:
            lam_l0 = u_l - a_l
            lam_r0 = u_star - a_l_star
            if lam_l0 < 0 < lam_r0:
                lam_0_av_new = lam_l0 * (lam_r0 - lam_0_av) / (lam_r0 - lam_l0)
                f0 = f_l0 + lam_0_av_new * alpha_0 * k00
                f1 = f_l1 + lam_0_av_new * alpha_0 * k10
                f2 = f_l2 + lam_0_av_new * alpha_0 * k20
        if p_star <= p_r:
            lam_l2 = u_star + a_r_star
            lam_r2 = u_r + a_r
            if lam_l2 < 0 < lam_r2:
                lam_2_av_new = lam_r2 * (lam_2_av - lam_l2) / (lam_r2 - lam_l2)
                f0 = f_r0 + lam_2_av_new * alpha_2 * k02
                f1 = f_r1 + lam_2_av_new * alpha_2 * k12
                f2 = f_r2 + lam_2_av_new * alpha_2 * k22

        return f0, f1, f2


def get_conv_flux(space_scheme: SpaceScheme) -> ConvectiveFlux:
    if space_scheme == SpaceScheme.Godunov:
        return Godunov
    elif space_scheme == SpaceScheme.StegerWarming:
        return StegerWarming
    elif space_scheme == SpaceScheme.VanLeer:
        return VanLeer
    elif space_scheme == SpaceScheme.HLLC:
        return HLLC
    elif space_scheme == SpaceScheme.Roe:
        return Roe
    elif space_scheme == SpaceScheme.RoeEntropyFix:
        return RoeEntropyFix
