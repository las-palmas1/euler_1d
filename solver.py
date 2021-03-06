import typing
import numpy as np
import log
from constant import *
import gdf
from dataclasses import dataclass, field
from conv_flux import get_conv_flux

# Будет два типа задач:
# 1 - c г.у. inlet-outlet
# 2 - c г.у., когда наложены либо reflective, либо transmissive, либо и то, и то.
#
# Инициализация проводится путем задания распределения по X температуры, скорости и давления.

# NOTE:
# 1. Правая грань ячейки - в направлении оси i, левая грань - против.
# 2. Размер всех массивов равен количеству ячеек всех типов.
# 3. Для массивов, содержащих параметры для граней - i-й элемент содержит параметры для левой грани i-й ячейки.

#
# |-------- l-state for i-th face --------------|           |-------- r-state for i-th face --------------|
# |--------------- (i-1)-th cell ---------------| i-th face |----------------- i-th cell -----------------|

# Vector of conservative variables:
#       rho * A
# cv =  u * rho * A
#       rho * (e + 0.5 * u^2) * A

# Flux vector:
#       rho * u * A
# F  =  (u * u * rho + p) * A
#       (rho * (e + 0.5 * u^2) + p) * u * A

# where specific internal energy e = p / (rho * (k - 1))

# Vector of left and right states: (rho, u, p)


@dataclass
class SolverData:
    num_real: int = field(init=True)
    num_dum: int = field(init=True, default=1)
    k: float = field(init=True, default=1.4)
    R: float = field(init=True, default=287.)

    def __post_init__(self):
        self.num_sum: int = self.num_real + 2 * self.num_dum
        # индекс первой действительной ячейки
        self.i_start: int = self.num_dum
        # площадь в центрах всех ячеек
        self.area_c = np.zeros(self.num_sum)
        # Значения консерватиных переменных в центрах ячеек на текущем временном уровне
        self.cv = np.zeros((3, self.num_sum))
        # Значения консерватиных переменных в центрах ячеек на предудыщем временном уровне
        self.cv_old = np.zeros((3, self.num_sum))
        # Значения невязок
        self.res = np.zeros((3, self.num_sum))
        # Статическое давление
        self.p = np.zeros(self.num_sum)
        # Скорость звука
        self.a = np.zeros(self.num_sum)
        # Шаг по времени
        self.dt: float = 0.
        # Текущее время
        self.time: float = 0.
        # Значения слева и справа от граней действительных ячеек, i-й элемент - состояния для левой грани i-й ячейки
        self.rs = np.zeros((3, self.num_sum))
        self.ls = np.zeros((3, self.num_sum))
        # Потоки через грани действительных ячеек, i-й элемент - поток для левой грани i-ой ячейки
        self.flux = np.zeros((3, self.num_sum))
        # Источниковые члены
        self.source_term = np.zeros((3, self.num_sum))


class BoundCond:
    def __init__(self):
        # Куда накладывать г.у.
        self.i_bc: int = 0
        # Направление вовнутрь блока, i_bc + ins_vec - индекс крайней точки, лежащей внутри блока. Равен 1 или -1.
        self.ins_vec: int = 1
        # Число мнимых ячеек
        self.num_dum: int = 1

    def impose(self, data: SolverData):
        pass


class SubsonicInletRiemann(BoundCond):
    def __init__(self, p_stag, T_stag):
        BoundCond.__init__(self)
        self.p_stag = p_stag
        self.T_stag = T_stag

    def impose(self, data: SolverData):
        i_ins = self.i_bc + self.ins_vec
        u_d = data.cv[1, i_ins] / data.cv[0, i_ins]
        # квадрат скорости звука на крайней реальной ячейке
        c_d2 = data.k * data.p[i_ins] * data.area_c[i_ins] / data.cv[0, i_ins]
        # квадарат скорости звука по параметрам торможения
        c_02 = c_d2 + 0.5 * (data.k - 1) * u_d**2
        # инвариант Римана
        rinv = u_d - 2 * c_d2**0.5 / (data.k - 1)
        # NOTE возможно -u_d, так как скалярное произведение нормали на вектор скорости, а нормаль направлена наружу области
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
        u_b = (2 * c_p * abs(self.T_stag - T_b))**0.5

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


class SubsonicInlet(BoundCond):
    def __init__(self, p_stag, T_stag):
        BoundCond.__init__(self)
        self.p_stag = p_stag
        self.T_stag = T_stag

    def impose(self, data: SolverData):
        i_ins = self.i_bc + self.ins_vec
        u_d = data.cv[1, i_ins] / data.cv[0, i_ins]
        u_b = u_d
        a_cr = gdf.a_cr(self.T_stag, data.k, data.R)
        lam_b = u_b / a_cr
        p_b = self.p_stag * gdf.pi_lam(lam_b, data.k)
        T_b = self.T_stag * gdf.tau_lam(lam_b, data.k)
        rho_b = p_b / (data.R * T_b)

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


class PressureOutletCharacteristic(BoundCond):
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


class PressureOutlet(BoundCond):
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
            u_b = u_d
            rho_b = rho_d * p_b / data.p[i_ins]

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

    def impose(self, data: SolverData):
        data.cv[0, self.i_bc] = data.cv[0, self.i_bc + self.ins_vec]
        data.cv[1, self.i_bc] = data.cv[1, self.i_bc + self.ins_vec]
        data.cv[2, self.i_bc] = data.cv[2, self.i_bc + self.ins_vec]
        data.p[self.i_bc] = data.p[self.i_bc + self.ins_vec]

        # линейная экстраполяция на остальные слои мнимых ячеек
        for i in range(1, self.num_dum):
            i_dum = self.i_bc + i * self.ins_vec * (-1)
            data.cv[0, i_dum] = 2 * data.cv[0, i_dum + self.ins_vec] - data.cv[0, i_dum + 2 * self.ins_vec]
            data.cv[1, i_dum] = 2 * data.cv[1, i_dum + self.ins_vec] - data.cv[1, i_dum + 2 * self.ins_vec]
            data.cv[2, i_dum] = 2 * data.cv[2, i_dum + self.ins_vec] - data.cv[2, i_dum + 2 * self.ins_vec]


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
        self.dx = None

    def init_mesh(self, data: SolverData):
        dx = (self.x2 - self.x1) / self.num_real
        self.dx = dx
        # грани действительных ячеек, i-я грань - левая грань i-ой ячейки
        self.x_face = np.zeros(self.num_sum)
        # центры всех ячеек
        self.x_c = np.zeros(self.num_sum)
        # площадь в центрах всех ячеек
        self.area_c = np.zeros(self.num_sum)
        # площадь на гранях действительных ячеек, i-й элемент - площадь левой грани i-ой ячейки
        self.area_face = np.zeros(self.num_sum)

        self.x_face[self.i_start: self.i_start + self.num_real + 1] = \
            np.linspace(self.x1, self.x2, self.num_real + 1)

        self.x_c[self.i_start: self.num_real + self.i_start] = \
            self.x_face[self.i_start: self.num_real + self.i_start] + dx / 2

        for i in range(self.num_dum):
            self.x_c[i] = self.x_c[self.num_dum] - dx * (self.num_dum - i)
            # self.area_c[i] = self.area_c[self.num_dum]
            self.x_c[self.num_real + self.num_dum + i] = self.x_c[self.num_real + self.num_dum - 1] + (i + 1) * dx
            # self.area_c[self.num_real + self.num_dum + i] = self.area_c[self.num_real + self.num_dum - 1]

        self.area_c[:] = np.array([
            self.area(x) for x in self.x_c[:]
        ])
        self.area_face[:] = np.array([
            self.area(x) for x in self.x_face[:]
        ])

        data.area_c = self.area_c

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
            T_ini: typing.Callable[[float], float],
            u_ini: typing.Callable[[float], float],
            p_ini: typing.Callable[[float], float],
            space_scheme: SpaceScheme = SpaceScheme.Godunov,
            time_scheme: TimeScheme = TimeScheme.ExplicitEuler,
            time_stepping: TimeStepping = TimeStepping.Global,
            ts_num=500,
            log_file='log.txt',
            log_console=True,
            log_level='info',
            **kwargs
    ):
        self.mesh = mesh
        self.T_ini = T_ini
        self.u_ini = u_ini
        self.p_ini = p_ini
        self.data = SolverData(mesh.num_real, mesh.num_dum, k, R)
        self.space_scheme = space_scheme
        self.time_scheme = time_scheme
        self.ts_num = ts_num
        self._kwargs = kwargs
        self.time_stepping = time_stepping
        # Невязка
        self.res = np.zeros(ts_num)
        # Нормированная невязка
        self.res_norm = np.zeros(ts_num)
        self.logger = log.Logger(log_level, log_file, log_console)

        if time_stepping == TimeStepping.Global and 'dt' not in kwargs:
            raise Exception('"dt" value must be set in case of global time stepping.')
        if time_stepping == TimeStepping.Local and 'cfl' not in kwargs:
            raise Exception('"cfl" must be set in case of local time stepping')

        if time_stepping == TimeStepping.Global:
            self.data.time = 0
            self.data.dt = self._kwargs['dt']
        if time_stepping == TimeStepping.Local:
            self.clf = kwargs['cfl']
            self.data.time = 0
            self.data.dt = None

    @classmethod
    def _check_bc(cls, bc1: BoundCond, bc2: BoundCond):
        if (type(bc1) == SubsonicInletRiemann and type(bc2) == PressureOutletCharacteristic) or \
                (type(bc1) == PressureOutletCharacteristic and type(bc2) == SubsonicInletRiemann) or \
                (type(bc1) == PressureOutlet and type(bc2) == SubsonicInletRiemann) or \
                (type(bc1) == SubsonicInletRiemann and type(bc2) == PressureOutlet) or \
                (type(bc1) == SubsonicInlet and type(bc2) == PressureOutletCharacteristic) or \
                (type(bc1) == PressureOutletCharacteristic and type(bc2) == SubsonicInlet) or \
                (type(bc1) == SubsonicInlet and type(bc2) == PressureOutlet) or \
                (type(bc1) == PressureOutlet and type(bc2) == SubsonicInlet):
            pass
        elif (type(bc1) == Transmissive and type(bc2) == Transmissive) or \
                (type(bc1) == Transmissive and type(bc2) == Reflective) or \
                (type(bc1) == Reflective and type(bc2) == Transmissive) or \
                (type(bc1) == Reflective and type(bc2) == Reflective):
            pass
        else:
            raise Exception('Invalid bc setting.')

    def init_flow(self):
        self._check_bc(self.mesh.bc1, self.mesh.bc2)
        T_ini_arr = np.array([self.T_ini(x) for x in self.mesh.x_c])
        u_ini_arr = np.array([self.u_ini(x) for x in self.mesh.x_c])
        p_ini_arr = np.array([self.p_ini(x) for x in self.mesh.x_c])
        rho_ini_arr = p_ini_arr / (self.data.R * T_ini_arr)
        self.data.cv[0, :] = rho_ini_arr * self.mesh.area_c
        self.data.cv[1, :] = u_ini_arr * rho_ini_arr * self.mesh.area_c
        self.data.p[:] = p_ini_arr
        self.data.a[:] = (self.data.k * self.data.p / rho_ini_arr)**0.5
        e_ini_arr = p_ini_arr / (rho_ini_arr * (self.data.k - 1))
        self.data.cv[2, :] = self.mesh.area_c * rho_ini_arr * (e_ini_arr + 0.5 * u_ini_arr ** 2)

    def _compute_rl_state(self):
        for i in range(self.mesh.i_start, self.mesh.i_start + self.mesh.num_real + 1):
            # Variables for left and right states: (rho u p)
            self.data.rs[0, i] = self.data.cv[0, i] / self.data.area_c[i]
            self.data.rs[1, i] = self.data.cv[1, i] / self.data.cv[0, i]
            self.data.rs[2, i] = self.data.p[i]
            self.data.ls[0, i] = self.data.cv[0, i - 1] / self.data.area_c[i - 1]
            self.data.ls[1, i] = self.data.cv[1, i - 1] / self.data.cv[0, i - 1]
            self.data.ls[2, i] = self.data.p[i - 1]

    def _compute_flux(self):
        self.logger.debug('Computing fluxes')
        # for flux vector splitting schemes - F_{i+1/2} = F^{+}_{U_l} + F^{-}_{U_r}

        conv_flux = get_conv_flux(self.space_scheme)

        for i in range(self.mesh.i_start, self.mesh.i_start + self.mesh.num_real + 1):
            self.logger.debug('Cell i = %s' % i)
            f0, f1, f2 = conv_flux.compute(self.data.ls[:, i], self.data.rs[:, i], self.data.k, self.data.dt)

            self.data.flux[0, i] = f0 * self.mesh.area_face[i]
            self.data.flux[1, i] = f1 * self.mesh.area_face[i]
            self.data.flux[2, i] = f2 * self.mesh.area_face[i]

    def _compute_source_term(self):
        for i in range(self.mesh.i_start, self.mesh.i_start + self.mesh.num_real):
            da = 0.5 * (self.mesh.area_c[i + 1] - self.mesh.area_c[i - 1])
            self.data.source_term[0, i] = 0
            self.data.source_term[1, i] = da * self.data.p[i]
            self.data.source_term[2, i] = 0

    def _compute_res(self):
        # Невязка вычисляется следующим образом:
        # R = F_{i+1/2} - F_{i-1/2} - S_i
        self.data.res[:, self.mesh.i_start: self.mesh.i_start + self.mesh.num_real] = (
                self.data.flux[:, self.mesh.i_start + 1: self.mesh.i_start + self.mesh.num_real + 1] -
                self.data.flux[:, self.mesh.i_start: self.mesh.i_start + self.mesh.num_real]
        ) - self.data.source_term[:, self.mesh.i_start: self.mesh.i_start + self.mesh.num_real]

    def solve(self):
        # Сходимость проверяется по норме изменения вектора плотности.
        # Величина невязок на каждом шаге по времени относится к величине невязки на первом шаге.
        self.logger.info('Initializing mesh')
        self.mesh.init_mesh(self.data)
        self.mesh.init_bc()
        self.logger.info('Initializing flow')
        self.init_flow()
        self.mesh.bc1.impose(self.data)
        self.mesh.bc2.impose(self.data)
        if self.time_scheme == TimeScheme.ExplicitEuler:
            # Для явной схемы Эйлера 1-го порядка значения на следующем временном уровне:
            # U^{i+1} = U^{i} - dt / dx * R
            self.logger.info('Start time iterating')
            for i in range(self.ts_num):
                try:
                    self.logger.info('Time step %s' % i)
                    self.data.cv_old = self.data.cv

                    if self.time_stepping == TimeStepping.Global:
                        pass
                    if self.time_stepping == TimeStepping.Local:
                        u = self.data.cv[1, :] / self.data.cv[0, :]
                        s_max = np.max(np.abs(u) + self.data.a)
                        self.data.dt = self.clf * self.mesh.dx / s_max
                    self.data.time = self.data.time + self.data.dt
                    self.logger.info('Time = %s' % self.data.time)

                    self._compute_rl_state()
                    self._compute_flux()
                    self._compute_source_term()
                    self._compute_res()

                    self.data.cv = self.data.cv_old - self.data.dt / self.mesh.dx * self.data.res
                    u = self.data.cv[1, :] / self.data.cv[0, :]
                    rho = self.data.cv[0, :] / self.mesh.area_c
                    e = self.data.cv[2, :] / (self.mesh.area_c * rho) - 0.5 * u**2
                    self.data.p = e * (self.data.k - 1) * rho
                    self.data.a = (self.data.k * self.data.p / rho)**0.5

                    self.mesh.bc1.impose(self.data)
                    self.mesh.bc2.impose(self.data)

                    rho_old = self.data.cv_old[0, :] / self.mesh.area_c
                    drho2 = (rho - rho_old)**2
                    self.res[i] = np.sqrt(np.sum(drho2))
                    self.res_norm[i] = self.res[i] / self.res[0]
                    self.logger.info('Res_rho = %.4f' % self.res[i])
                    self.logger.info('Res_rho_norm = %.4f\n' % self.res_norm[i])
                except Exception as ex:
                    self.logger.info('ERROR: ' + str(ex))
                    break









