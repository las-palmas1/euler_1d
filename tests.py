import pytest
import os
from post_proc import plot_vars_jointly, plot_vars_separately, Variable, VarSettings
from riemann_solvers.godunov import GodunovRiemannSolver
from laval_nozzle import LavalNozzleSolver
import solver
from constant import *
import numpy as np


log_dirname = 'log'
plots_dirname = 'test_plots'


def setup_module():
    if not os.path.exists(log_dirname):
        os.makedirs(log_dirname)

    for name in os.listdir(log_dirname):
        abs_name = os.path.join(log_dirname, name)
        if os.path.isfile(abs_name):
            os.remove(abs_name)

    if not os.path.exists(plots_dirname):
        os.makedirs(plots_dirname)

    for name in os.listdir(plots_dirname):
        abs_name = os.path.join(plots_dirname, name)
        if os.path.isfile(abs_name):
            os.remove(abs_name)


@pytest.mark.parametrize(
    argnames=['rho_l', 'u_l', 'p_l', 'rho_r', 'u_r', 'p_r', 'x0_rel', 'name'],
    argvalues=[
        (1.0, 0.75, 1.0, 0.125, 0.0, 0.1, 0.3, 'Test 1'),
        (1.0, -2, 0.4, 1.0, 2.0, 0.4, 0.5, 'Test 2'),
        (1.0, 0, 1000, 1.0, 0.0, 0.01, 0.5, 'Test 3'),
        (5.99924, 19.5975, 460.894, 5.99242, -6.19633, 46.0950, 0.4, 'Test 4'),
        (1.0, -19.59745, 1000, 1.0, -19.59745, 0.01, 0.8, 'Test 5'),
    ]
)
class TestRiemannProblem:
    def setup(self):
        self.k = 1.4
        self.R = 287
        self.x1 = 0
        self.x2 = 1
        self.cfl = 0.5
        self.ts_num = 150
        self.num_real = 100
        self.num_dum = 1
        self.num_pnt_exact = 500
        self.time_stepping = TimeStepping.Local
        self.time_scheme = TimeScheme.ExplicitEuler
        self.log_level = 'info'
        self.fname_pref = 'riem_pr-'

        self.area = lambda x: 1
        self.inlet = solver.Transmissive()
        self.outlet = solver.Transmissive()
        self.mesh = solver.Quasi1DBlock(
            area=self.area, x1=self.x1, x2=self.x2, bc1=self.inlet, bc2=self.outlet,
            num_real=self.num_real, num_dum=self.num_dum
        )

        self.rho_set = VarSettings(var=Variable.rho, color='red', axis_label='Density')
        self.u_set = VarSettings(var=Variable.u, color='red', axis_label='Velocity')
        self.p_set = VarSettings(var=Variable.p, color='red', axis_label='Pressure')
        self.e_set = VarSettings(var=Variable.e, color='red', axis_label='Internal Energy')
        self.figsize = (8, 6)

    def solve(self, rho_l, u_l, p_l, rho_r, u_r, p_r, x0_rel, name, space_scheme, plot_title, fname_sub):
        x0 = self.x1 + (self.x2 - self.x1) * x0_rel

        T_l = p_l / (self.R * rho_l)
        T_r = p_r / (self.R * rho_r)

        T_ini = lambda x: T_l if x <= x0 else T_r
        p_ini = lambda x: p_l if x <= x0 else p_r
        u_ini = lambda x: u_l if x <= x0 else u_r

        num_solver = solver.SolverQuasi1D(
            mesh=self.mesh, k=self.k, R=self.R, T_ini=T_ini, u_ini=u_ini, p_ini=p_ini,
            space_scheme=space_scheme, time_scheme=self.time_scheme, time_stepping=self.time_stepping,
            ts_num=self.ts_num,
            log_file=os.path.join(log_dirname, self.fname_pref + '%s-%s.log' % (fname_sub, name)), log_console=True,
            log_level=self.log_level, cfl=self.cfl
        )
        num_solver.solve()

        sol_time = num_solver.data.time

        x_exact = np.linspace(self.x1, self.x2, self.num_pnt_exact)
        exact_solver = GodunovRiemannSolver(rho_l, u_l, p_l, rho_r, u_r, p_r, p_star_init_type='PV', k=self.k)
        exact_solver.compute_star()

        rho_exact = np.zeros(self.num_pnt_exact)
        u_exact = np.zeros(self.num_pnt_exact)
        p_exact = np.zeros(self.num_pnt_exact)
        for n, x in enumerate(x_exact):
            solution = exact_solver.get_point(x - x0, sol_time)
            rho_exact[n] = solution[0]
            u_exact[n] = solution[1]
            p_exact[n] = solution[2]
        e_exact = p_exact / (rho_exact * (self.k - 1))

        plot_vars_separately(
            solver_data=num_solver.data, mesh=self.mesh,
            vars_settings=[self.rho_set, self.u_set, self.p_set, self.e_set],
            ref_vars_distrs={
                Variable.rho: (x_exact, rho_exact), Variable.u: (x_exact, u_exact), Variable.p: (x_exact, p_exact),
                Variable.e: (x_exact, e_exact)
            },
            figsize=self.figsize, show=False,
            fname=os.path.join(plots_dirname, self.fname_pref + '%s-%s' % (fname_sub, name)),
            title=plot_title
        )

    def test_godunov(self, rho_l, u_l, p_l, rho_r, u_r, p_r, x0_rel, name):
        self.solve(
            rho_l, u_l, p_l, rho_r, u_r, p_r, x0_rel, name=name, space_scheme=SpaceScheme.Godunov,
            plot_title='Godunov', fname_sub='godunov'
        )

    def test_steger_warming(self, rho_l, u_l, p_l, rho_r, u_r, p_r, x0_rel, name):
        self.solve(
            rho_l, u_l, p_l, rho_r, u_r, p_r, x0_rel, name=name, space_scheme=SpaceScheme.StegerWarming,
            plot_title='Steger-Warming', fname_sub='steg-warm'
        )


@pytest.mark.parametrize(
    argnames=['p2', 'name'],
    argvalues=[
        (60e3, 'nominal'),
        (350e3, 'underexpansion'),
    ]
)
class TestLavalNozzle:
    def setup(self):
        self.k = 1.4
        self.R = 287.

        self.area = lambda x: 1 + 2 * x ** 2
        self.x1 = -1 / 3
        self.x2 = 1
        self.T_stag = 1000
        self.p1_stag = 500e3

        self.cfl = 0.5
        self.ts_num = 800
        self.num_real = 60
        self.num_dum = 1
        self.time_stepping = TimeStepping.Local
        self.time_scheme = TimeScheme.ExplicitEuler
        self.log_level = 'info'
        self.fname_pref = 'lav-nozz-'

        self.inlet = solver.SubsonicInlet(p_stag=self.p1_stag, T_stag=self.T_stag)

        self.num_exact = 200
        self.lam_pre_init = 1.5

        self.lam_set = VarSettings(var=Variable.lam, color='red', axis_label=r'$ \lambda $')
        self.T_set = VarSettings(var=Variable.T, color='blue', axis_label='Temperature')
        self.p_set = VarSettings(var=Variable.p, color='green', axis_label='Pressure')
        self.figsize = (8, 6)

    def solve(self, p2, name, space_scheme, plot_title, fname_sub):
        T = self.T_stag * (p2 / self.p1_stag) ** ((self.k - 1) / self.k)
        Mach = ((self.T_stag / T - 1) * 2 / (self.k - 1)) ** 0.5
        a = (self.k * self.R * T) ** 0.5

        T_ini = lambda x: T
        u_ini = lambda x: a * Mach
        p_ini = lambda x: p2

        outlet = solver.PressureOutlet(p=p2)
        mesh = solver.Quasi1DBlock(
            area=self.area, x1=self.x1, x2=self.x2, bc1=self.inlet, bc2=outlet,
            num_real=self.num_real, num_dum=self.num_dum
        )
        num_solver = solver.SolverQuasi1D(
            mesh=mesh, k=self.k, R=self.R,
            T_ini=T_ini, u_ini=u_ini, p_ini=p_ini,
            space_scheme=space_scheme, time_scheme=self.time_scheme,
            time_stepping=self.time_stepping, ts_num=self.ts_num,
            log_file=os.path.join(log_dirname, self.fname_pref + '%s-%s.log' % (fname_sub, name)), log_console=True,
            log_level=self.log_level, cfl=self.cfl
        )
        num_solver.solve()

        exact_solver = LavalNozzleSolver(
            T_stag=self.T_stag, p1_stag=self.p1_stag, p2=p2, area=self.area, x1=self.x1, x2=self.x2, k=self.k,
            R=self.R, num=self.num_exact, lam_pre_init=self.lam_pre_init
        )
        exact_solver.compute()

        plot_vars_jointly(
            solver_data=num_solver.data, mesh=mesh,
            vars_settings=[self.lam_set, self.T_set, self.p_set],
            ref_vars_distrs={
                Variable.lam: (exact_solver.x_arr, exact_solver.lam_arr),
                Variable.T: (exact_solver.x_arr, exact_solver.T_arr),
                Variable.p: (exact_solver.x_arr, exact_solver.p_arr),
            },
            figsize=self.figsize, show=False,
            fname=os.path.join(plots_dirname, self.fname_pref + '%s-%s' % (fname_sub, name)),
            title=plot_title, sol_markevery=4
        )

    def test_godunov(self, p2, name):
        self.solve(p2, name, space_scheme=SpaceScheme.Godunov, plot_title='Godunov', fname_sub='godunov')

    def test_steger_warming(self, p2, name):
        self.solve(p2, name, space_scheme=SpaceScheme.StegerWarming, plot_title='Steger-Warming', fname_sub='steg-warm')










