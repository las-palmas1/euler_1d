import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.figure
import matplotlib.axes
from solver import SolverData, Quasi1DBlock
from typing import List, Dict, Tuple
import numpy as np
from enum import Enum
import gdf
import os


# TODO: сделать сохранение данных на каждом шаге и запись по ним анимации

class Variable(Enum):
    u = 0
    p = 1
    T = 2
    lam = 3
    Mach = 4
    e = 5
    rho = 6


class VarSettings:
    def __init__(
            self, var: Variable = None, min_val=None, max_val=None, legend_label='', axis_label='', color='',
            scale_factor=1
    ):
        self.var: Variable = var
        self.min = min_val
        self.max = max_val
        self.legend_label = legend_label
        self.axis_label = axis_label
        self.color = color
        self.scale_factor = scale_factor


def make_patch_spines_invisible(ax):
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for sp in ax.spines.values():
        sp.set_visible(False)


def plot_vars_jointly(
        solver_data: SolverData, mesh: Quasi1DBlock,
        vars_settings: List[VarSettings],
        ref_vars_distrs: Dict[Variable, Tuple[np.ndarray, np.ndarray]],
        figsize=(10, 6), axis_gap=0.2, host_right_axis_pos=0.68,
        sol_lw=2, sol_ls='--', sol_marker='o', sol_ms=6, sol_markevery=4,
        ref_lw=2, ref_ls=':',
        label_fontszie=14,
        legend_fontsize=12,
        ticks_size=4,
        ticks_width=1.5,
        title='',
        title_fontsize=16,
        fname: str = '',
        show=True
):
    x_solv = mesh.x_c
    u_solv = solver_data.cv[1, :] / solver_data.cv[0, :]
    p_solv = solver_data.p
    rho_solv = solver_data.cv[0, :] / mesh.area_c
    e_solv = p_solv / (rho_solv * (solver_data.k - 1))
    a_solv = solver_data.a
    Mach_solv = u_solv / a_solv
    T_solv = p_solv / (rho_solv * solver_data.R)
    T_stag_solv = T_solv * gdf.tau_M(Mach_solv, solver_data.k)
    a_cr_solv = gdf.a_cr(T_stag_solv, solver_data.k, solver_data.R)
    lam_solv = u_solv / a_cr_solv

    subplots = plt.subplots()
    fig: mpl.figure.Figure = subplots[0]
    fig.set_size_inches(figsize[0], figsize[1])

    host: mpl.axes.Axes = subplots[1]
    fig.subplots_adjust(right=host_right_axis_pos)

    axes: List[mpl.axes.Axes] = [host]
    for n_var in range(1, len(vars_settings)):
        par: mpl.axes.Axes = host.twinx()
        par.spines["right"].set_position(("axes", 1 + axis_gap * (n_var - 1)))
        if n_var >= 2:
            make_patch_spines_invisible(par)
            par.spines["right"].set_visible(True)
        axes.append(par)

    lines = []
    y_solv = None
    tkw = dict(size=ticks_size, width=ticks_width)

    if title:
        title = title + ', time = %.3f' % solver_data.time
    else:
        title = 'time = %.3f' % solver_data.time

    for axis, var_settings in zip(axes, vars_settings):
        x_ref, y_ref = ref_vars_distrs[var_settings.var]

        if var_settings.var == Variable.u:
            y_solv = u_solv
        elif var_settings.var == Variable.p:
            y_solv = p_solv
        elif var_settings.var == Variable.T:
            y_solv = T_solv
        elif var_settings.var == Variable.Mach:
            y_solv = Mach_solv
        elif var_settings.var == Variable.e:
            y_solv = e_solv
        elif var_settings.var == Variable.lam:
            y_solv = lam_solv
        elif var_settings.var == Variable.rho:
            y_solv = rho_solv

        y_solv = y_solv * var_settings.scale_factor

        l1, = axis.plot(
            x_solv, y_solv, ls=sol_ls, lw=sol_lw, marker=sol_marker, markevery=sol_markevery, ms=sol_ms,
            label=var_settings.legend_label, color=var_settings.color
        )
        axis.plot(x_ref, y_ref, lw=ref_lw, ls=ref_ls, color=var_settings.color)

        axis.set_ylabel(var_settings.axis_label, color=l1.get_color(), fontsize=label_fontszie)
        axis.yaxis.label.set_color(var_settings.color)
        axis.tick_params(axis='y', colors=var_settings.color, **tkw)

        if var_settings.min and var_settings.max:
            axis.set_ylim(bottom=var_settings.min, top=var_settings.max)

        elif not var_settings.min and var_settings.max:
            axis.set_ylim(top=var_settings.max)

        elif var_settings.min and not var_settings.max:
            axis.set_ylim(bottom=var_settings.min)

        lines.append(l1)

    host.set_title(title, fontsize=title_fontsize)
    host.set_xlim(x_solv.min(), x_solv.max())
    host.set_xlabel(r"$x,\ м$", fontsize=label_fontszie)
    host.tick_params(axis='x', **tkw)
    host.grid(True)

    host.legend(lines, [l.get_label() for l in lines], fontsize=legend_fontsize, loc='lower left')

    if fname:
        plt.savefig(fname)
    if show:
        plt.show()


def plot_vars_separately(
        solver_data: SolverData, mesh: Quasi1DBlock,
        vars_settings: List[VarSettings],
        ref_vars_distrs: Dict[Variable, Tuple[np.ndarray, np.ndarray]],
        figsize=(10, 6), axes_rect=(0.12, 0.12, 0.78, 0.74),
        sol_lw=2, sol_ls='--',
        sol_marker='o', sol_ms=6, sol_markevery=4,
        ref_lw=2, ref_ls=':',
        label_fontszie=14,
        ticks_size=4,
        ticks_width=1.5,
        title='',
        title_fontsize=16,
        fname: str = '',
        show=True
):
    x_solv = mesh.x_c
    u_solv = solver_data.cv[1, :] / solver_data.cv[0, :]
    p_solv = solver_data.p
    rho_solv = solver_data.cv[0, :] / mesh.area_c
    e_solv = p_solv / (rho_solv * (solver_data.k - 1))
    a_solv = solver_data.a
    Mach_solv = u_solv / a_solv
    T_solv = p_solv / (rho_solv * solver_data.R)
    T_stag_solv = T_solv * gdf.tau_M(Mach_solv, solver_data.k)
    a_cr_solv = gdf.a_cr(T_stag_solv, solver_data.k, solver_data.R)
    lam_solv = u_solv / a_cr_solv

    y_solv = None
    tkw = dict(size=ticks_size, width=ticks_width)

    if title:
        title = title + ', time = %.3f' % solver_data.time
    else:
        title = 'time = %.3f' % solver_data.time

    for var_settings in vars_settings:
        x_ref, y_ref = ref_vars_distrs[var_settings.var]

        if var_settings.var == Variable.u:
            y_solv = u_solv
        elif var_settings.var == Variable.p:
            y_solv = p_solv
        elif var_settings.var == Variable.T:
            y_solv = T_solv
        elif var_settings.var == Variable.Mach:
            y_solv = Mach_solv
        elif var_settings.var == Variable.e:
            y_solv = e_solv
        elif var_settings.var == Variable.lam:
            y_solv = lam_solv
        elif var_settings.var == Variable.rho:
            y_solv = rho_solv

        y_solv = y_solv * var_settings.scale_factor

        fig: plt.Figure = plt.figure(figsize=figsize)
        ax: plt.Axes = fig.add_axes(axes_rect)

        ax.plot(
            x_solv, y_solv, ls=sol_ls, lw=sol_lw, marker=sol_marker, markevery=sol_markevery,
            ms=sol_ms, color=var_settings.color
        )
        ax.plot(x_ref, y_ref, lw=ref_lw, ls=ref_ls, color=var_settings.color)

        ax.set_title(title, fontsize=title_fontsize)
        ax.set_xlim(x_solv.min(), x_solv.max())
        ax.set_xlabel(r"$x,\ м$", fontsize=label_fontszie)
        ax.set_ylabel(var_settings.axis_label, fontsize=label_fontszie)
        ax.tick_params(axis='x', **tkw)
        ax.tick_params(axis='y', **tkw)
        ax.grid(True)

        if var_settings.min and var_settings.max:
            ax.set_ylim(bottom=var_settings.min, top=var_settings.max)

        elif not var_settings.min and var_settings.max:
            ax.set_ylim(top=var_settings.max)

        elif var_settings.min and not var_settings.max:
            ax.set_ylim(bottom=var_settings.min)

        if fname:
            plt.savefig(os.path.splitext(fname)[0] + '-%s.png' % var_settings.var.name)
        if show:
            plt.show()

