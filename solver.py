import typing
import numpy as np
import godunov as gd

# Будет два типа задач:
# 1 - c г.у. inlet-outlet
# 2 - c г.у., когда наложены либо reflective, либо transmissive, либо и то, и то.
#
# В 1 случае инициализация будет проводится по данным г.у.
# Во 2 случае должон задаваться распределение консервативных переменных по X.

# NOTE:
# 1. Правая грань ячейки - в направлении оси i, левая грань - против.
# 2. Размер всех массивов равен количеству ячеек всех типов.
# 3. Для массивов, содержащих параметры для граней - i-й элемент содержит параметры для левой грани i-й ячейки.

#
# |-------- l-state for i-th face --------------|           |-------- r-state for i-th face --------------|
# |--------------- (i-1)-th cell ---------------| i-th face |----------------- i-th cell -----------------|


class SolverData:
    def __init__(self, num_real: int, num_dum, k=1.4, R=287):
        self.num_real = num_real
        self.num_dum = num_dum
        self.num_sum = num_real + 2 * num_dum
        self.k = k
        self.R = R
        # индекс первой действительной ячейки
        self.i_start = self.num_dum
        # площадь в центрах всех ячеек
        self.area_c = np.zeros(self.num_sum)
        # Значения консерватиных переменных в центрах ячеек на текущем временном уровне
        self.cv = np.zeros(3, self.num_sum)
        # Значения консерватиных переменных в центрах ячеек на предудыщем временном уровне
        self.cv_old = np.zeros(3, self.num_sum)
        # Значения невязок
        self.res = np.zeros(3, self.num_sum)
        # Статическое давление
        self.p = np.zeros(self.num_sum)
        # Шаг по времени
        self.dt = np.zeros(self.num_sum)
        # Текущее время
        self.time = np.zeros(self.num_sum)
        # Значения слева и справа от граней действительных ячеек, i-й элемент - состояния для левой грани i-й ячейки
        self.rs = np.zeros(3, self.num_sum)
        self.ls = np.zeros(3, self.num_sum)
        # Потоки через грани действительных ячеек, i-й элемент - поток для левой грани i-ой ячейки
        self.flux = np.zeros(3, self.num_sum)


class BoundCond:
    def __init__(self):
        # Куда накладывать г.у.
        self.i_bc = None
        # Направление вовнутрь блока, num_bc + dom_vec - индекс крайней точки, лежащей внутри блока. Равен 1 или -1.
        self.ins_vec = None
        # Число мнимых ячеек
        self.num_dum = None

    def impose(self, data: SolverData):
        pass


class SubsonicInlet(BoundCond):
    def __init__(self, p_stag, T_stag):
        BoundCond.__init__(self)
        self.p_stag = p_stag
        self.T_stag = T_stag

    def impose(self, data: SolverData):
        i_ins = self.i_bc + self.ins_vec
        u_d = data.cv[1, i_ins] / data.cv[0, i_ins]
        # квадрат скорости звука на крайней реальной ячейке
        c_d2 = data.k * data.p[i_ins] * data.area_c[i_ins]
        # квадарат скорости звука по параметрам торможения
        c_02 = c_d2 + 0.5 * (data.k - 1) * u_d**2
        # инвариант Римана
        rinv = u_d - 2 * c_d2**0.5 / (data.k - 1)
        # подкоренное выражение
        dis = c_02 * (data.k + 1) / (rinv**2 * (data.k - 1)) - (data.k - 1) / 2
        if dis < 0:
            dis = 0
        c_b = -rinv * (data.k - 1) / (data.k + 1) * (1 + dis**0.5)
        T_b = c_b**2 / (data.k * data.R)
        cc = min(c_b**2 / c_02, 1)
        p_b = self.p_stag * cc ** (data.k / (data.k - 1))
        rho_b = p_b / (data.R * T_b)
        c_p = data.k * data.R / (data.k - 1)
        u_b = (2 * c_p * (self.T_stag - T_b))**0.5

        data.cv[0, self.i_bc] = rho_b * data.area_c[self.i_bc]
        data.cv[1, self.i_bc] = rho_b * data.area_c[self.i_bc] * u_b
        data.cv[2, self.i_bc] = (p_b / (data.k - 1) + 0.5 * u_b**2 * rho_b) * data.area_c[self.i_bc]
        data.p[self.i_bc] = p_b

        # линейная экстраполяция на остальные слои мнимых ячеек
        for i in range(1, self.num_dum):
            i_dum = self.i_bc + i * self.ins_vec * (-1)
            data.cv[0, i_dum] = 2 * data.cv[0, i_dum + self.ins_vec] - data.cv[0, i_dum + 2 * self.ins_vec]
            data.cv[1, i_dum] = 2 * data.cv[1, i_dum + self.ins_vec] - data.cv[1, i_dum + 2 * self.ins_vec]
            data.cv[2, i_dum] = 2 * data.cv[2, i_dum + self.ins_vec] - data.cv[2, i_dum + 2 * self.ins_vec]


class Outlet(BoundCond):
    def __init__(self, p):
        BoundCond.__init__(self)
        self.p = p

    def impose(self, data: SolverData):
        i_ins = self.i_bc + self.ins_vec
        u_d = data.cv[1, i_ins] / data.cv[0, i_ins]
        rho_d = data.cv[0, i_ins] / data.area_c[i_ins]
        c_d = (data.k * data.p[i_ins] / rho_d)**0.5
        # supersonic outlet
        if u_d >= c_d:
            p_b = data.p[i_ins]
            rho_b = rho_d
            u_b = u_d
        # subsonic outlet
        else:
            p_b = self.p
            rho_b = rho_d + (p_b - data.p[i_ins]) / c_d**2
            u_b = u_d + (data.p[i_ins] - p_b) / (rho_d * c_d)

        data.cv[0, self.i_bc] = rho_b * data.area_c[self.i_bc]
        data.cv[1, self.i_bc] = rho_b * data.area_c[self.i_bc] * u_b
        data.cv[2, self.i_bc] = (p_b / (data.k - 1) + 0.5 * u_b ** 2 * rho_b) * data.area_c[self.i_bc]
        data.p[self.i_bc] = p_b

        # линейная экстраполяция на остальные слои мнимых ячеек
        for i in range(1, self.num_dum):
            i_dum = self.i_bc + i * self.ins_vec * (-1)
            data.cv[0, i_dum] = 2 * data.cv[0, i_dum + self.ins_vec] - data.cv[0, i_dum + 2 * self.ins_vec]
            data.cv[1, i_dum] = 2 * data.cv[1, i_dum + self.ins_vec] - data.cv[1, i_dum + 2 * self.ins_vec]
            data.cv[2, i_dum] = 2 * data.cv[2, i_dum + self.ins_vec] - data.cv[2, i_dum + 2 * self.ins_vec]


class Reflective(BoundCond):
    def __init__(self):
        BoundCond.__init__(self)


class Transmissive(BoundCond):
    def __init__(self):
        BoundCond.__init__(self)


class Quasi1DBlock:
    def __init__(
            self,
            area: typing.Callable[[float], float],
            x1, x2,
            bc1: BoundCond, bc2: BoundCond,
            num_real: int, num_dum: int = 1
    ):
        self.area = area
        self.x1 = x1
        self.x2 = x2
        self.num_real = num_real
        self.num_dum = num_dum
        self.num_sum = num_real + 2 * num_dum
        # индекс первой действительной ячейки
        self.i_start = self.num_dum
        self.bc1 = bc1
        self.bc2 = bc2
        self.x_face = None
        self.area_face = None
        self.x_c = None
        self.area_c = None

    def init_mesh(self):
        dx = (self.x2 - self.x1) / self.num_real
        # грани действительных ячеек, i-я грань - левая грань i-ой ячейки
        self.x_face = np.zeros(self.num_sum)
        # центры всех ячеек
        self.x_c = np.zeros(self.num_sum)
        # площадь в центрах всех ячеек
        self.area_c = np.zeros(self.num_sum)
        # площадь на гранях действительных ячеек, i-й элемент - площадь левой грани i-ой ячейки
        self.area_face = np.zeros(self.num_sum)

        self.x_c[self.i_start: self.num_real + self.i_start] = \
            self.x_face[self.i_start: self.num_real + self.i_start] + dx

        self.x_face[self.i_start: self.i_start + self.num_real + 1] = \
            np.linspace(self.x1, self.x2, self.num_real + 1)

        self.area_c[self.i_start: self.num_real + self.i_start] = np.array([
            self.area(x) for x in self.x_c[self.i_start: self.num_real + self.i_start]
        ])
        self.area_face[self.i_start: self.num_real + self.i_start + 1] = np.array([
            self.area(x) for x in self.x_face[self.i_start: self.num_real + self.i_start + 1]
        ])

        for i in range(self.num_dum):
            self.x_c[i] = self.x_c[self.num_dum] - dx * (self.num_dum - i)
            self.area_c[i] = self.area_c[self.num_dum]
            self.x_c[self.num_real + self.num_dum + i] = self.x_c[self.num_real + self.num_dum - 1] + (i + 1) * dx
            self.area_c[self.num_real + self.num_dum + i] = self.area_c[self.num_real + self.num_dum - 1]

    def init_bc(self):
        self.bc1.num_dum = self.num_dum
        self.bc1.ins_vec = 1
        self.bc1.i_bc = self.num_dum - 1
        self.bc2.num_dum = self.num_dum
        self.bc2.ins_vec = -1
        self.bc2.i_bc = self.num_sum - self.num_dum


class SolverQuasi1D:
    def __init__(
            self,
            mesh: Quasi1DBlock,
            k, R,
            rho_ini: typing.Callable[[float], float],
            u_ini: typing.Callable[[float], float],
            p_ini: typing.Callable[[float], float],
            space_scheme='Godunov',
            time_scheme='Explicit Euler',
            time_stepping='Global',
            conv=1e-5,
            max_iter=500,
            **kwargs):
        self.mesh = mesh
        self.rho_ini = rho_ini
        self.u_ini = u_ini
        self.p_ini = p_ini
        self.data = SolverData(mesh.num_real, mesh.num_dum, k, R)
        self.space_scheme = space_scheme
        self.time_scheme = time_scheme
        self.conv = conv
        self.max_iter = max_iter
        self._kwargs = kwargs
        if time_stepping == 'Global'and 'ts' not in kwargs:
            raise Exception('"ts" value must be set in the case of global time stepping.')

    @classmethod
    def _check_bc(cls, bc1: BoundCond, bc2: BoundCond):
        if (type(bc1) == SubsonicInlet and type(bc2) == Outlet) or (type(bc1) == Outlet or type(bc2) == SubsonicInlet):
            pass
        elif (type(bc1) == Transmissive and type(bc2) == Transmissive) or\
                (type(bc1) == Transmissive and type(bc2) == Reflective) or\
                (type(bc1) == Reflective and type(bc2) == Transmissive) or\
                (type(bc1) == Reflective and type(bc2) == Reflective):
            pass
        else:
            raise Exception('Invalid bc setting.')

    def init_flow(self):
        self._check_bc(self.mesh.bc1, self.mesh.bc2)
        rho_ini_arr = np.array([self.rho_ini(x) for x in self.mesh.x_c])
        u_ini_arr = np.array([self.u_ini(x) for x in self.mesh.x_c])
        p_ini_arr = np.array([self.p_ini(x) for x in self.mesh.x_c])
        self.data.cv[0, :] = rho_ini_arr * self.mesh.area_c
        self.data.cv[1, :] = u_ini_arr * rho_ini_arr * self.mesh.area_c
        self.data.p[:] = p_ini_arr
        e_ini_arr = p_ini_arr / (rho_ini_arr * (self.data.k - 1))
        self.data.cv[2, :] = self.mesh.area_c * self.rho_ini * (e_ini_arr + 0.5 * u_ini_arr**2)

    def compute_rl_state(self):
        if self.space_scheme == 'Godunov':
            for i in range(self.mesh.i_start, self.mesh.i_start + self.mesh.num_real + 1):
                self.data.rs[0, i] = self.data.cv[0, i] / self.data.area_c[i]
                self.data.rs[1, i] = self.data.cv[1, i] / self.data.cv[0, i]
                self.data.rs[2, i] = self.data.p[i]
                self.data.ls[0, i] = self.data.cv[0, i - 1] / self.data.area_c[i - 1]
                self.data.ls[1, i] = self.data.cv[1, i - 1] / self.data.cv[0, i - 1]
                self.data.ls[2, i] = self.data.p[i]

    def compute_flux(self):
        if self.space_scheme == 'Godunov':
            pass







