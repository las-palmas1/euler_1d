import pytest
import os
from post_proc import plot_vars_jointly, plot_vars_separately, Variable, VarSettings
from godunov import GodunovRiemannSolver
from laval_nozzle import LavalNozzleSolver
import solver
import numpy as np


log_dirname = 'log'
plots_dirname = 'test_plots'

k = 1.4
R = 287


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
def test_riemann_problem_godunov(rho_l, u_l, p_l, rho_r, u_r, p_r, x0_rel, name):
    x1 = 0
    x2 = 1
    cfl = 0.5
    ts_num = 100
    num_real = 100
    num_dum = 1
    num_pnt_exact = 500

    x0 = x1 + (x2 - x1) * x0_rel

    T_l = p_l / (R * rho_l)
    T_r = p_r / (R * rho_r)

    T_ini = lambda x: T_l if x <= x0 else T_r
    p_ini = lambda x: p_l if x <= x0 else p_r
    u_ini = lambda x: u_l if x <= x0 else u_r
    area = lambda x: 1

    inlet = solver.Transmissive()
    outlet = solver.Transmissive()
    mesh = solver.Quasi1DBlock(area=area, x1=x1, x2=x2, bc1=inlet, bc2=outlet, num_real=num_real, num_dum=num_dum)
    num_solver = solver.SolverQuasi1D(
        mesh=mesh, k=k, R=R, T_ini=T_ini, u_ini=u_ini, p_ini=p_ini,
        space_scheme='Godunov', time_scheme='Explicit Euler', time_stepping='Local', ts_num=ts_num,
        log_file=os.path.join(log_dirname, 'riem_pr-godunov-%s.log' % name), log_console=True,
        log_level='info', cfl=cfl
    )
    num_solver.solve()

    sol_time = num_solver.data.time

    x_exact = np.linspace(x1, x2, num_pnt_exact)
    exact_solver = GodunovRiemannSolver(rho_l, u_l, p_l, rho_r, u_r, p_r, p_star_init_type='PV', k=k)
    exact_solver.compute_star()

    rho_exact = np.zeros(num_pnt_exact)
    u_exact = np.zeros(num_pnt_exact)
    p_exact = np.zeros(num_pnt_exact)
    for n, x in enumerate(x_exact):
        solution = exact_solver.get_point(x - x0, sol_time)
        rho_exact[n] = solution[0]
        u_exact[n] = solution[1]
        p_exact[n] = solution[2]
    e_exact = p_exact / (rho_exact * (k - 1))

    rho_set = VarSettings(var=Variable.rho, color='red', axis_label='Density')
    u_set = VarSettings(var=Variable.u, color='red', axis_label='Velocity')
    p_set = VarSettings(var=Variable.p, color='red', axis_label='Pressure')
    e_set = VarSettings(var=Variable.e, color='red', axis_label='Internal Energy')

    plot_vars_separately(
        solver_data=num_solver.data, mesh=mesh, vars_settings=[rho_set, u_set, p_set, e_set],
        ref_vars_distrs={
            Variable.rho: (x_exact, rho_exact), Variable.u: (x_exact, u_exact), Variable.p: (x_exact, p_exact),
            Variable.e: (x_exact, e_exact)
        },
        figsize=(8, 6), show=False, fname=os.path.join(plots_dirname, 'riem_pr-godunov-%s' % name),
        title='Godunov'
    )









