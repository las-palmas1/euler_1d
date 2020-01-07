import unittest
from godunov import GodunovRiemannSolver, SolutionType
from laval_nozzle import LavalNozzleSolver
from solver import BoundCond, Quasi1DBlock, SolverQuasi1D, SubsonicInletRiemann, PressureOutletCharacteristic, \
    SolverData, PressureOutlet, SubsonicInlet, Transmissive
import matplotlib.pyplot as plt
import numpy as np


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
        solver.plot_time_level(0.25, (-0.5, 0.5), num_pnt=200)

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
        solver.plot_time_level(0.15, (-0.5, 0.5), num_pnt=200)

    def test_case3(self):
        rho_l = 1
        u_l = 0
        p_l = 1000
        rho_r = 1
        u_r = 0
        p_r = 0.01

        solver = GodunovRiemannSolver(rho_l, u_l, p_l, rho_r, u_r, p_r, p_star_init_type='TR', k=1.4)
        solver.compute_star()
        solver.plot_time_level(0.012, (-0.5, 0.5), num_pnt=200)

    def test_case4(self):
        rho_l = 1
        u_l = 0
        p_l = 0.01
        rho_r = 1
        u_r = 0
        p_r = 100

        solver = GodunovRiemannSolver(rho_l, u_l, p_l, rho_r, u_r, p_r, p_star_init_type='PV', k=1.4)
        solver.compute_star()
        solver.plot_time_level(0.035, (-0.5, 0.5), num_pnt=200)

    def test_case5(self):
        rho_l = 1
        u_l = 0
        p_l = 0.01
        rho_r = 1
        u_r = 0
        p_r = 0.01

        solver = GodunovRiemannSolver(rho_l, u_l, p_l, rho_r, u_r, p_r, p_star_init_type='PV', k=1.4)
        solver.compute_star()
        solver.plot_time_level(0.035, (-0.5, 0.5), num_pnt=200)


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
            p2=350e3,
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


class Quasi1DBlockTests(unittest.TestCase):

    def setUp(self):
        self.b1 = BoundCond()
        self.b2 = BoundCond()
        self.area = lambda x: x
        self.x1 = 0
        self.x2 = 10
        self.num_real = 10
        self.num_dum = 3
        self.data = SolverData(self.num_real, self.num_dum)

    def test_mesh_init(self):
        block = Quasi1DBlock(
            area=self.area,
            x1=self.x1, x2=self.x2,
            bc1=self.b1, bc2=self.b2,
            num_real=self.num_real,
            num_dum=self.num_dum
        )
        block.init_mesh(self.data)
        self.assertEqual(block.x_c[block.i_start], 0.5)
        self.assertEqual(block.x_face[block.i_start], 0)

        self.assertEqual(block.area_c[block.i_start], self.area(0.5))
        self.assertEqual(block.area_c[block.i_start + 1], self.area(1.5))
        self.assertEqual(block.area_c[block.i_start - 1], self.area(0.5))
        self.assertEqual(block.area_c[block.i_start - 2], self.area(0.5))
        self.assertEqual(block.area_c[block.num_sum - 1], self.area(9.5))

        self.assertEqual(block.area_face[block.i_start], self.area(0))
        self.assertEqual(block.area_face[block.i_start + 1], self.area(1))


def plot_rel(data: SolverData, mesh: Quasi1DBlock):
    u = data.cv[1, :] / data.cv[0, :]
    u = np.where(np.isnan(u), np.zeros(u.shape), u)
    u_rel = u / u.max()
    rho = data.cv[0, :] / mesh.area_c
    rho = np.where(np.isnan(rho), np.zeros(rho.shape), rho)
    rho_rel = rho / rho.max()
    p = np.where(np.isnan(data.p), np.zeros(data.p.shape), data.p)
    p_rel = p / p.max()
    plt.plot(mesh.x_c, u_rel, 'red', ls='-', marker='o', ms=2, label='u')
    plt.plot(mesh.x_c, rho_rel, 'blue', ls='-', marker='o', ms=2, label='rho')
    plt.plot(mesh.x_c, p_rel, 'green', ls='-', marker='o', ms=2, label='p')
    plt.legend()
    plt.ylim(top=1.1)
    plt.xlim(left=mesh.x_c.min())
    plt.xlim(right=mesh.x_c.max())
    plt.grid()
    plt.xlabel('x', fontsize=12)
    plt.show()


class LavalNozzleTests(unittest.TestCase):

    def setUp(self):
        self.area = lambda x: 1 + 2 * x ** 2
        self.x1 = -1 / 3
        self.x2 = 1
        self.k = 1.4
        self.R = 287.
        self.num_real = 50
        self.num_dum = 1
        self.T_stag = 1000
        self.p1_stag = 500e3
        self.p2 = 350e3
        T = self.T_stag * (self.p2 / self.p1_stag) ** ((self.k - 1) / self.k)
        Mach = ((self.T_stag / T - 1) * 2 / (self.k - 1)) ** 0.5
        a = (self.k * self.R * T) ** 0.5
        self.T_ini = lambda x: T
        self.u_ini = lambda x: a * Mach
        self.p_ini = lambda x: self.p2

    def test_inlet_riemann_pressure_outlet_char(self):
        inlet = SubsonicInletRiemann(p_stag=self.p1_stag, T_stag=self.T_stag)
        outlet = PressureOutletCharacteristic(p=self.p2)
        mesh = Quasi1DBlock(
            area=self.area, x1=self.x1, x2=self.x2, bc1=inlet, bc2=outlet,
            num_real=self.num_real, num_dum=self.num_dum
        )
        solver = SolverQuasi1D(
            mesh=mesh, k=self.k, R=self.R,
            T_ini=self.T_ini, u_ini=self.u_ini, p_ini=self.p_ini,
            space_scheme='Godunov', time_scheme='Explicit Euler',
            time_stepping='Local', ts_num=350, log_file='', log_console=True,
            log_level='info', cfl=0.6
        )
        solver.solve()
        plot_rel(solver.data, solver.mesh)

    def test_inlet_pressure_outlet_char(self):
        inlet = SubsonicInlet(p_stag=self.p1_stag, T_stag=self.T_stag)
        outlet = PressureOutletCharacteristic(p=self.p2)
        mesh = Quasi1DBlock(
            area=self.area, x1=self.x1, x2=self.x2, bc1=inlet, bc2=outlet,
            num_real=self.num_real, num_dum=self.num_dum
        )
        solver = SolverQuasi1D(
            mesh=mesh, k=self.k, R=self.R,
            T_ini=self.T_ini, u_ini=self.u_ini, p_ini=self.p_ini,
            space_scheme='Godunov', time_scheme='Explicit Euler',
            time_stepping='Local', ts_num=350, log_file='', log_console=True,
            log_level='info', cfl=0.6
        )
        solver.solve()
        plot_rel(solver.data, solver.mesh)

    def test_inlet_pressure_outlet(self):
        inlet = SubsonicInlet(p_stag=self.p1_stag, T_stag=self.T_stag)
        outlet = PressureOutlet(p=self.p2)
        mesh = Quasi1DBlock(
            area=self.area, x1=self.x1, x2=self.x2, bc1=inlet, bc2=outlet,
            num_real=self.num_real, num_dum=self.num_dum
        )
        solver = SolverQuasi1D(
            mesh=mesh, k=self.k, R=self.R,
            T_ini=self.T_ini, u_ini=self.u_ini, p_ini=self.p_ini,
            space_scheme='Godunov', time_scheme='Explicit Euler',
            time_stepping='Local', ts_num=300, log_file='', log_console=True,
            log_level='info', cfl=0.6
        )
        solver.solve()
        plot_rel(solver.data, solver.mesh)

    def test_inlet_riemann_pressure_outlet(self):
        inlet = SubsonicInletRiemann(p_stag=self.p1_stag, T_stag=self.T_stag)
        outlet = PressureOutlet(p=self.p2)
        mesh = Quasi1DBlock(
            area=self.area, x1=self.x1, x2=self.x2, bc1=inlet, bc2=outlet,
            num_real=self.num_real, num_dum=self.num_dum
        )
        solver = SolverQuasi1D(
            mesh=mesh, k=self.k, R=self.R,
            T_ini=self.T_ini, u_ini=self.u_ini, p_ini=self.p_ini,
            space_scheme='Godunov', time_scheme='Explicit Euler',
            time_stepping='Local', ts_num=300, log_file='', log_console=True,
            log_level='info', cfl=0.6
        )
        solver.solve()
        plot_rel(solver.data, solver.mesh)


class RiemannProblemTest(unittest.TestCase):
    def setUp(self):
        self.k = 1.4
        self.R = 287.
        self.rho_l = 1
        self.u_l = 0.75
        self.p_l = 1.0
        self.T_l = self.p_l / (self.R * self.rho_l)
        self.rho_r = 0.125
        self.u_r = 0
        self.p_r = 0.1
        self.T_r = self.p_r / (self.R * self.rho_r)

        self.x1 = 0
        self.x2 = 1
        self.x0 = 0.3
        self.num_real = 100
        self.num_dum = 1
        self.dt = 1e-3
        self.ts_num = 50

        self.T_ini = lambda x: self.T_l if x <= self.x0 else self.T_r
        self.p_ini = lambda x: self.p_l if x <= self.x0 else self.p_r
        self.u_ini = lambda x: self.u_l if x <= self.x0 else self.u_r
        self.area = lambda x: 1

    def test(self):
        inlet = Transmissive()
        outlet = Transmissive()
        mesh = Quasi1DBlock(
            area=self.area, x1=self.x1, x2=self.x2, bc1=inlet, bc2=outlet,
            num_real=self.num_real, num_dum=self.num_dum
        )
        solver = SolverQuasi1D(
            mesh=mesh, k=self.k, R=self.R,
            T_ini=self.T_ini, u_ini=self.u_ini, p_ini=self.p_ini,
            space_scheme='Godunov', time_scheme='Explicit Euler',
            time_stepping='Local', ts_num=self.ts_num, log_file='', log_console=True,
            log_level='info', cfl=0.5
        )
        solver.solve()
        plot_rel(solver.data, solver.mesh)


