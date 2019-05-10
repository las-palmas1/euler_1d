import unittest
from godunov import GodunovRiemannSolver, SolutionType
from laval_nozzle import LavalNozzleSolver


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

    def test_case5(self):
        rho_l = 1
        u_l = 0
        p_l = 0.01
        rho_r = 1
        u_r = 0
        p_r = 0.01

        solver = GodunovRiemannSolver(rho_l, u_l, p_l, rho_r, u_r, p_r, p_star_init_type='PV', k=1.4)
        solver.compute_star()
        solver.plot_time_level(0.035, (-1, 1), num_pnt=200)


class LavalNozzleSolverTests(unittest.TestCase):

    def setUp(self):
        self.area = lambda x: 1 + 2 * x**2
        self.x1 = -1 / 3
        self.x2 = 1
        self.k = 1.4
        self.R = 287.
        self.num = 200
        self.lam_pre_init = 1.5

    def test_nominal(self):
        solver = LavalNozzleSolver(
            T_stag=1000,
            p1_stag=500e3,
            p2=60e3,
            x1=self.x1,
            x2=self.x2,
            area=self.area,
            k=self.k,
            R=self.R,
            num=self.num
        )
        solver.compute()
        solver.plot()

    def test_underexpansion(self):
        solver = LavalNozzleSolver(
            T_stag=1000,
            p1_stag=500e3,
            p2=300e3,
            x1=self.x1,
            x2=self.x2,
            area=self.area,
            k=self.k,
            R=self.R,
            num=self.num,
            lam_pre_init=self.lam_pre_init
        )
        solver.compute()
        solver.plot()

