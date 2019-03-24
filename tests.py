import unittest
from godunov import GodunovRiemannSolver, SolutionType


class TestRiemannSolver(unittest.TestCase):

    def test_case1(self):
        rho_l = 1
        u_l = 0
        p_l = 1
        rho_r = 0.125
        u_r = 0
        p_r = 0.1

        solver = GodunovRiemannSolver(rho_l, u_l, p_l, rho_r, u_r, p_r, p_star_init_type='MV', k=1.4)
        solver.compute_star()
        solver.plot_time_level(0.25, (-1, 1), num_pnt=200)

    def test_case2(self):
        rho_l = 1
        u_l = -2
        p_l = 0.4
        rho_r = 1
        u_r = 2
        p_r = 0.4
        tol = 1e-7

        solver = GodunovRiemannSolver(rho_l, u_l, p_l, rho_r, u_r, p_r, p_star_init_type='PV', k=1.4, tol=tol)
        solver.compute_star()
        solver.plot_time_level(0.15, (-1, 1), num_pnt=200)

    def test_case3(self):
        rho_l = 1
        u_l = 0
        p_l = 1000
        rho_r = 1
        u_r = 0
        p_r = 0.01

        solver = GodunovRiemannSolver(rho_l, u_l, p_l, rho_r, u_r, p_r, p_star_init_type='TR', k=1.4)
        solver.compute_star()
        solver.plot_time_level(0.012, (-1, 1), num_pnt=200)

    def test_case4(self):
        rho_l = 1
        u_l = 0
        p_l = 0.01
        rho_r = 1
        u_r = 0
        p_r = 100

        solver = GodunovRiemannSolver(rho_l, u_l, p_l, rho_r, u_r, p_r, p_star_init_type='PV', k=1.4)
        solver.compute_star()
        solver.plot_time_level(0.035, (-1, 1), num_pnt=200)

