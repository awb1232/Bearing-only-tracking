"""
   修改了创建模型时初始化传入的参数
   与以前的代码结合
"""

"""
    2025 05 06
    之前的代码是计算pos误差和vel误差，现在分别计算x y vx vy

    2025 05 12
    1.完善了CRLB下界的计算与可视化
    2.修改了8形状轨迹的参数，增加了其他几种轨迹（o,z,s）可供选择

"""

"""
    mk0以及之前的代码方位角都是按照x0逆时针增大计算的，现在修改为y0顺时针增大
    涉及的修改：

    1. Class Model中 def generate_bearings:
       true_bearing = np.arctan2(dy, dx) 修改为 np.arctan2(dx, dy)

    2. class BearingOnlyEKF中
       def predict_bearing
          bearing = np.arctan2(dy, dx) 修改
       def measurement_jacobian 修改
          H[0, 0]和H[0, 1] 修改

    3. class BearingOnlyPLKF中
       def predict_bearing

       def construct_pseudo_linear_measurement

    4. class BearingOnlyUKF和CKF中
       def predict_bearing需要修改

    整合了CRLB绘图（尚有部分问题）

    将Model类中generate_bearings拆分为generate_bearings和generate_measurements，后者生成含噪方位

"""
"""
   由文件basic_kf_mc_simulator改编而来，仅验证基础的卡尔曼滤波算法
   1.修改了Runner类：原来的Runner在运行蒙特卡洛仿真前会先刷新一次含噪方位角，这组方位角不是所有算法通用的，导致对比性能的时候产生差异
     现在Runner类中增加了一个生成指定数量个含噪方位的函数，生成全局数据让后续所有被选择的算法进行仿真
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.animation as animation
from scipy.stats import chi2
import time
from tqdm import tqdm

plt.rcParams['font.sans-serif'] = ['SimSun']  # 使用宋体字体
plt.rcParams['axes.unicode_minus'] = False  # 显示负号

def deg1deg2add(deg1, deg2):

    return (deg1 + deg2) % 360

def deg1deg2sub(deg1, deg2):
    """
    求角度差，返回0-360度的差角

    :param deg1:
    :param deg2:
    :return:
    """
    return (deg1 - deg2) % 360

def deg1deg2sub1(deg1, deg2):

    return (deg1 - deg2 + 180) % 360 - 180


class Point2D:
    def __init__(self, state, mode, acceleration, omega, maneuver=None):
        """
        按照mode初始化ship的状态，并为其运动参数赋值

        :param state: 状态，可以是
        :param mode: 状态对应的物理量

        :param omega:
        :param maneuver:
        """

        # 按照指定的mode初始化ship类的状态
        if mode == 'xyvv':
            self.state = state.astype(float)
        elif mode == 'bdcv':
            self.state = np.array([
                state[1] * np.sin(np.deg2rad(state[0])),
                state[1] * np.cos(np.deg2rad(state[0])),
                state[3] * np.sin(np.deg2rad(state[2])),
                state[3] * np.cos(np.deg2rad(state[2]))
            ])
        else:
            raise ValueError('unsupported initialization method')

        self.position = np.array([self.state[0], self.state[1]], dtype=float)
        self.course = np.rad2deg( np.arctan2(self.state[2], self.state[3]) )
        self.speed = np.sqrt(self.state[2]**2 + self.state[3]**2)

        self.acceleration = acceleration
        self.omega = omega
        self.maneuver = maneuver

    def uniform_linear_motion(self, dt):
        """
        执行一次匀速直线运动，

        :param dt: 模拟最小的时间步长
        :return: 无
        """
        delta_position = self.speed * np.array(
            [np.sin(np.deg2rad(self.course)), np.cos(np.deg2rad(self.course))]) * dt
        self.position += delta_position

    def accelerated_linear_motion(self, dt, acc_direction):
        """匀加速直线运动"""
        if acc_direction == 0:
            self.speed -= self.acceleration * dt
        else:
            self.speed += self.acceleration * dt
        self.uniform_linear_motion(dt)

    def uniform_circular_motion(self, dt, turn_direction):
        """
        步进一次圆周运动

        :param dt:
        :param turn_direction: 0左转1右转
        :return:
        """
        delta_angular = self.omega * dt
        if turn_direction == 0:
            self.course -= delta_angular
        else:
            self.course += delta_angular
        self.course %= 360
        self.uniform_linear_motion(dt)

    def hold_course_speed(self, dt, crs, spd):

        delta_crs = deg1deg2sub1(crs, self.course)
        delta_spd = spd - self.speed

        if abs(delta_crs) > 1e-3:
            turn_direction = 0 if delta_crs < 0 else 1
            self.uniform_circular_motion(dt, turn_direction)

        elif abs(delta_spd) > 1e-3:
            acc_direction = 0 if delta_spd < 0 else 1
            self.accelerated_linear_motion(dt, acc_direction)


class Model:

    def __init__(self,
                 Sensor,
                 Target,
                 dt,
                 maxt,
                 brg_noise_mean,
                 brg_noise_std,
                 ):
        self.Sensor = Sensor
        self.Target = Target
        self.target_init_state = Target.state
        self.sample_time = dt
        self.max_simulation_time = maxt
        self.bearing_noise_mean = brg_noise_mean
        self.bearing_noise_std = brg_noise_std
        self.R = np.deg2rad((self.bearing_noise_std) ** 2)  # 测量噪声方差 (弧度)

        self.steps = int(self.max_simulation_time / self.sample_time)
        self.times = np.arange(0, self.steps * self.sample_time + self.sample_time, self.sample_time)
        self.sensor_trajectory = np.zeros((self.steps + 1, 2))  # 传感器轨迹

        # 目标真实状态
        self.target_states = np.zeros((self.steps + 1, 4))
        self.target_states[0] = self.target_init_state

        # 方位角
        self.bearings = np.zeros((self.steps + 1, 1))
        self.measurements = np.zeros((self.steps + 1, 1))

        self.crlb = np.zeros((self.steps + 1, 4))

    def generate_target_trajectory(self):

        F = np.array([
            [1, 0, self.sample_time, 0],
            [0, 1, 0, self.sample_time],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        for k in range(1, self.steps + 1):
            self.target_states[k] = F @ self.target_states[k - 1]

    def generate_sensor_trajectory_8(self):
        """
        生成横”8“子型的曲线运动轨迹

        :return:
        """

        for k in range(self.steps + 1):
            t = k * self.sample_time
            self.sensor_trajectory[k, 0] = 100 * np.sin(t / 20)
            self.sensor_trajectory[k, 1] = 50 * np.sin(t / 10)

    def generate_sensor_trajectory_circle(self,
                                          clockwise=False):
        """
        生成圆周运动轨迹

        参数:
        start_angle_deg: 起始角度，单位为度，默认为90度（沿x轴正方向）
        linear_speed: 线速度，默认为5 m/s
        angular_speed_deg: 角速度，单位为度/秒，默认为1度/秒
        clockwise: 是否顺时针运动，默认为False（逆时针）

        :return: None
        """
        # 生成圆周轨迹，左转逆时针右转顺时针
        turn_direction = 0 if not clockwise else 1
        for k in range(self.steps + 1):
            self.Sensor.uniform_circular_motion(self.sample_time, turn_direction)

            # 位置坐标
            self.sensor_trajectory[k, 0] = self.Sensor.position[0]
            self.sensor_trajectory[k, 1] = self.Sensor.position[1]

    def generate_sensor_trajectory_s(self, main_time = 120, delta_crs=45, part_time = 60):

        main_crs = self.Sensor.course.copy()
        main_steps = int(main_time / self.sample_time)
        part_steps = int(part_time / self.sample_time)

        for k in range(self.steps + 1):

            if k <= main_steps:
                self.Sensor.uniform_linear_motion(self.sample_time)

            else:
                turn_stage = 0
                turn_flag = True
                line_step = 0

                next_crs = deg1deg2add(main_crs, ((-1)**turn_stage) * delta_crs)

                delta_to_next = deg1deg2sub1(self.Sensor.course, next_crs)
                if abs(deg1deg2sub1(self.Sensor.course, next_crs)) <= 1e-3:
                    turn_flag = False

                if turn_flag:
                    self.Sensor.hold_course_speed(self.sample_time, next_crs, self.Sensor.speed)
                else:
                    self.Sensor.uniform_linear_motion(self.sample_time)
                    line_step += 1

                line_steps = part_steps if turn_stage < 1 else 2 * part_steps

                if line_step >= line_steps:
                    turn_flag = True
                    turn_stage += 1
                    line_step = 0

            # 位置坐标
            self.sensor_trajectory[k, 0] = self.Sensor.position[0]
            self.sensor_trajectory[k, 1] = self.Sensor.position[1]

    def generate_sensor_trajectory_z(self, step_interval=50):
        """
        生成Z字型leg-leg机动轨迹

        参数:
        step_interval: 每段直线运动的步数
        velocity: 运动速度，默认为5 m/s

        运动模式:
        - 0到step_interval: 航向90度（向上），速度v
        - step_interval到3*step_interval: 航向0度（向右），速度v
        - 3*step_interval到5*step_interval: 航向90度（向上），速度v
        - 以此类推...
        """
        # 初始位置
        start_x = 0
        start_y = 0

        # 当前位置
        current_x = start_x
        current_y = start_y

        # 航向角度（初始为90度，即向上）
        current_heading_deg = 90

        # 每个时间步的距离
        distance_per_step = self.Sensor.speed * self.sample_time

        # 生成Z型轨迹
        for k in range(self.steps + 1):
            # 计算当前所处的leg段
            leg_number = k // step_interval

            # 每两段改变一次航向
            if leg_number % 2 == 0:
                current_heading_deg = 90  # 向上
            else:
                current_heading_deg = 0  # 向右

            # 航向角转换为弧度
            heading_rad = np.deg2rad(current_heading_deg)

            # 计算当前位置
            if k == 0:
                self.sensor_trajectory[k, 0] = current_x
                self.sensor_trajectory[k, 1] = current_y
            else:
                # 根据当前航向更新位置
                current_x += distance_per_step * np.cos(heading_rad)
                current_y += distance_per_step * np.sin(heading_rad)

                self.sensor_trajectory[k, 0] = current_x
                self.sensor_trajectory[k, 1] = current_y

    def generate_bearings(self):

        for k in range(self.steps + 1):
            observer_pos = self.sensor_trajectory[k]
            dx = self.target_states[k, 0] - observer_pos[0]
            dy = self.target_states[k, 1] - observer_pos[1]
            true_bearing = np.arctan2(dx, dy)
            self.bearings[k, 0] = true_bearing

    def generate_measurements(self):

        for k in range(self.steps + 1):
            observer_pos = self.sensor_trajectory[k]
            dx = self.target_states[k, 0] - observer_pos[0]
            dy = self.target_states[k, 1] - observer_pos[1]
            true_bearing = np.arctan2(dx, dy)
            self.measurements[k, 0] = true_bearing + np.sqrt(self.R) * np.random.randn()

    def generate_crlb(self):

        #j = 0

        for step in range(1, self.steps + 1):
            jacobian = np.zeros((step, 4))

            j = step - 1

            for i in range(step):

                xt_i, yt_i, _, _ = self.target_states[i]
                xo_i, yo_i = self.sensor_trajectory[i]

                d_squared = (xt_i - xo_i) ** 2 + (yt_i - yo_i) ** 2
                d_Bi_to_Xtxj = (yt_i - yo_i) / d_squared
                d_Bi_to_Xtyj = -(xt_i - xo_i) / d_squared
                d_Bi_to_Xtvx = (self.times[i] - self.times[j]) * d_Bi_to_Xtxj
                d_Bi_to_Xtvy = (self.times[i] - self.times[j]) * d_Bi_to_Xtyj

                jacobian[i] = [d_Bi_to_Xtxj, d_Bi_to_Xtyj, d_Bi_to_Xtvx, d_Bi_to_Xtvy]

            fim = jacobian.T @ jacobian / self.R

            try:
                fim_inv = np.linalg.inv(fim)
                self.crlb[step] = np.diag(fim_inv)
            except np.linalg.LinAlgError:
                # In case of singular matrix
                self.crlb[step] = np.array([float('inf'), float('inf'), float('inf'), float('inf')])


class BearingOnlyEKF:

    def __init__(self, x0, P0, Q, R, dt, observer_trajectory, measurements, backward=False):
        """
        说明：假设你要进行n次卡尔曼滤波递推，那么observer_trajectory和measurements应该包含从初始状态和往后n个状态下的量测和坐标

        :param x0: 初始状态向量 [x, y, vx, vy]
        :param P0: 初始化协方差矩阵
        :param Q: 过程噪声协方差矩阵
        :param R: 测量噪声协方差 (标量，弧度)
        :param dt: 时间步长（采样周期）
        :param observer_trajectory: 传感器轨迹，每行为一个时间步的位置 [x, y]，长度为n+1
        :param measurements: 量测方位序列（弧度）
        :param backward: 是否逆向滤波
        """

        self.n = len(x0)  # 状态维度
        self.x = x0.copy()  # 状态向量
        self.P = P0.copy()  # 协方差矩阵
        self.Q = Q.copy()  # 过程噪声协方差矩阵
        self.R = R  # 测量噪声协方差 (标量)
        self.dt = dt  # 时间步长
        self.backward = backward

        if self.backward:
            self.observer_trajectory = observer_trajectory[::-1]
            self.measurements = measurements[::-1]
        else:
            self.observer_trajectory = observer_trajectory
            self.measurements = measurements

        if len(self.observer_trajectory) != len(self.measurements):
            raise ValueError("传入的坐标序列和方位序列不等长")

        self.current_step = 1  # 当前步

    def state_transition(self, x, add_noise=False):
        """
        状态转移函数 - 匀速直线运动模型
        x = [位置x, 位置y, 速度vx, 速度vy]
        """
        if self.backward:
            # 逆向状态转移矩阵
            F = np.array([
                [1, 0, -self.dt, 0],
                [0, 1, 0, -self.dt],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])
        else:
            # 正向状态转移矩阵
            F = np.array([
                [1, 0, self.dt, 0],
                [0, 1, 0, self.dt],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])

        new_x = F @ x

        # 添加过程噪声
        if add_noise:
            noise = np.random.multivariate_normal(np.zeros(self.n), self.Q)
            new_x += noise

        return new_x

    def cov_transition(self, P, add_noise=False):
        if self.backward:
            # 逆向状态转移矩阵
            F = np.array([
                [1, 0, -self.dt, 0],
                [0, 1, 0, -self.dt],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])
        else:
            # 正向状态转移矩阵
            F = np.array([
                [1, 0, self.dt, 0],
                [0, 1, 0, self.dt],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])

        Ppre = F @ P @ F.T

        Ppre += self.Q

        return Ppre

    def measurement_jacobian(self, x):
        """
        计算测量函数关于状态的雅可比矩阵H
        H = [∂h/∂x, ∂h/∂y, ∂h/∂vx, ∂h/∂vy]
        """
        observer_pos = self.observer_trajectory[self.current_step]
        dx = x[0] - observer_pos[0]
        dy = x[1] - observer_pos[1]

        # 计算方位角测量对位置的偏导数
        denominator = dx ** 2 + dy ** 2

        # 防止分母为零
        if denominator < 1e-10:
            denominator = 1e-10

        H = np.zeros((1, self.n))
        H[0, 0] = dy / denominator  # ∂(arctan2(dy,dx))/∂x
        H[0, 1] = -dx / denominator  # ∂(arctan2(dy,dx))/∂y
        H[0, 2] = 0  # 速度不直接影响方位角测量
        H[0, 3] = 0

        return H

    def predict_bearing(self, x, step=None):
        """
        测量函数 - 计算从观测者到目标的方位角
        返回以弧度表示的方位角
        """
        if step is None:
            step = self.current_step

        observer_pos = self.observer_trajectory[step]
        dx = x[0] - observer_pos[0]
        dy = x[1] - observer_pos[1]
        bearing = np.arctan2(dx, dy)

        return np.array([bearing])

    def normalize_angle(self, angle):
        """将角度归一化到[-pi, pi]范围内"""
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def predict(self):

        Xpre = self.state_transition(self.x)

        Ppre = self.cov_transition(self.P)

        self.x = Xpre

        self.P = Ppre

    def update(self):
        """EKF更新步骤"""
        # 计算预测测量值
        z_pred = self.predict_bearing(self.x)

        # 计算测量雅可比矩阵
        H = self.measurement_jacobian(self.x)

        # 计算新息协方差
        S = H @ self.P @ H.T + self.R

        # 计算卡尔曼增益
        K = self.P @ H.T / S

        # 获取实际测量值
        z = self.measurements[self.current_step]

        # 计算创新序列（测量残差）
        z_residual = z - z_pred
        z_residual[0] = self.normalize_angle(z_residual[0])

        # 更新状态
        self.x += K.flatten() * z_residual[0]

        # 约瑟夫形式更新协方差，提高数值稳定性
        I = np.eye(self.n)
        self.P = (I - np.outer(K, H)) @ self.P #@ (I - np.outer(K, H)).T + self.R * np.outer(K, K)

    def step(self):
        """执行完整的EKF步骤：预测和更新"""
        # 预测步骤
        self.predict()

        # 更新步骤
        self.update()

        # 更新当前步
        self.current_step += 1

        return self.x, self.P


class BearingOnlyPLKF:
    """使用伪线性卡尔曼滤波器(PLKF)进行纯方位目标运动分析"""

    def __init__(self, x0, P0, Q, R, dt, observer_trajectory, measurements, debais=False, backward=False):
        """
        初始化伪线性卡尔曼滤波器

        :param x0: 初始状态向量 [x, y, vx, vy]
        :param P0: 初始化协方差矩阵
        :param Q: 过程噪声协方差矩阵
        :param R: 测量噪声协方差 (标量，弧度)
        :param dt: 时间步长（采样周期）
        :param observer_trajectory: 传感器轨迹，每行为一个时间步的位置 [x, y]，长度为n+1
        :param measurements: 量测方位序列（弧度）
        :param backward: 是否逆向滤波
        """

        self.n = len(x0)  # 状态维度
        self.x = x0.copy()  # 状态向量
        self.x = np.array([0,0,0,0])
        self.P = P0.copy()  # 协方差矩阵
        self.P = np.eye(4)
        self.Q = Q.copy()  # 过程噪声协方差矩阵
        self.R = R  # 测量噪声协方差 (标量)
        self.dt = dt  # 时间步长
        self.backward = backward

        if self.backward:
            self.observer_trajectory = observer_trajectory[::-1]
            self.measurements = measurements[::-1]
        else:
            self.observer_trajectory = observer_trajectory
            self.measurements = measurements

        if len(self.observer_trajectory) != len(self.measurements):
            raise ValueError("传入的坐标序列和方位序列不等长")

        self.current_step = 1  # 当前步（用于索引observer_trajectory和measurements）

        # 存储诊断信息
        self.innovation = []  # 保存创新序列
        self.innovation_covariance = []  # 保存创新协方差

        # PLKF特定参数
        self.debiasing_enabled = debais  # 启用去偏处理
        self.max_iterations = 3  # 迭代去偏的最大迭代次数

    def state_transition(self, x, add_noise=False):
        """
        状态转移函数 - 匀速直线运动模型
        x = [位置x, 位置y, 速度vx, 速度vy]
        """
        if self.backward:
            # 逆向状态转移矩阵
            F = np.array([
                [1, 0, -self.dt, 0],
                [0, 1, 0, -self.dt],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])
        else:
            # 正向状态转移矩阵
            F = np.array([
                [1, 0, self.dt, 0],
                [0, 1, 0, self.dt],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])

        new_x = F @ x

        # 添加过程噪声
        if add_noise:
            noise = np.random.multivariate_normal(np.zeros(self.n), self.Q)
            new_x += noise

        return new_x

    def cov_transition(self, P, add_noise=False):
        if self.backward:
            # 逆向状态转移矩阵
            F = np.array([
                [1, 0, -self.dt, 0],
                [0, 1, 0, -self.dt],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])
        else:
            # 正向状态转移矩阵
            F = np.array([
                [1, 0, self.dt, 0],
                [0, 1, 0, self.dt],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])

        Ppre = F @ P @ F.T

        Ppre += self.Q

        return Ppre

    def predict_bearing(self, x, step=None):
        """
        测量函数 - 计算从观测者到目标的方位角
        返回以弧度表示的方位角
        """
        if step is None:
            step = self.current_step

        observer_pos = self.observer_trajectory[step]
        dx = x[0] - observer_pos[0]
        dy = x[1] - observer_pos[1]
        bearing = np.arctan2(dx, dy)

        return np.array([bearing])

    def construct_pseudo_linear_measurement(self, z):
        """
        构造伪线性测量方程
        z = arctan2(dy, dx) => sin(z)*dx - cos(z)*dy = 0

        返回:
        - 伪线性测量矩阵H
        - 伪线性测量值z_pl (通常为0)
        """
        observer_pos = self.observer_trajectory[self.current_step]
        sin_z = np.sin(z)
        cos_z = np.cos(z)

        # 构造伪线性测量矩阵 H = [sin(z), -cos(z), 0, 0]
        H = np.zeros((1, self.n))
        H[0, 0] = -cos_z
        H[0, 1] = sin_z

        # 伪线性测量中的偏置项
        bias_term = sin_z * observer_pos[0] - cos_z * observer_pos[1]

        # 伪线性测量值(通常为0)
        z_pl = np.array([bias_term])

        return H, z_pl

    def pseudo_linear_measurement(self, z):

        sin_z = np.sin(z)
        cos_z = np.cos(z)

        # 构造伪线性测量矩阵 H = [sin(z), -cos(z), 0, 0]
        H = np.zeros((1, self.n))
        H[0, 0] = -cos_z
        H[0, 1] = sin_z

        return H

    def calculate_debiasing_gain(self, H, P):
        """
        计算去偏增益矩阵
        """
        # 计算测量预测协方差
        S = H @ P @ H.T + self.R

        # 计算去偏增益参数
        gamma = S / (S + self.R)

        return gamma

    def normalize_angle(self, angle):
        """将角度归一化到[-pi, pi]范围内"""
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def predict(self):
        """PLKF预测步骤"""
        # 计算状态转移矩阵

        # 预测状态
        self.x = self.state_transition(self.x)

        # 预测协方差
        self.P = self.cov_transition(self.P)

    def update(self):
        """PLKF更新步骤"""
        # 获取当前测量值
        z = self.measurements[self.current_step]

        # 预测测量值
        z_pred = self.predict_bearing(self.x)

        # 计算测量残差
        z_residual = z - z_pred
        z_residual = self.normalize_angle(z_residual[0])

        # 构造伪线性测量系统
        H, z_pl = self.construct_pseudo_linear_measurement(z)

        # 标准卡尔曼滤波更新过程
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T / S[0, 0]

        # 计算伪线性测量残差
        # 在伪线性测量中，残差是与当前状态的线性关系
        y = z_pl - H @ self.x

        # 保存诊断信息
        self.innovation.append(z_residual)
        self.innovation_covariance.append(S[0, 0])

        # 标准更新（无去偏）
        self.x = self.x + K.flatten() * y[0]
        I_KH = np.eye(self.n) - np.outer(K, H)
        self.P = I_KH @ self.P @ I_KH.T + np.outer(K, K) * self.R

    def step(self):
        """执行完整的PLKF步骤：预测和更新"""

        # 预测
        # 预测状态
        Xpre = self.state_transition(self.x)

        # 预测协方差
        Ppre = self.cov_transition(self.P)

        z = self.measurements[self.current_step]
        H = self.pseudo_linear_measurement(z)

        Zpre = H @ Xpre

        S = H @ Ppre @ H.T + self.R

        K = Ppre @ H.T / S[0,0]

        self.x = Xpre + np.squeeze(K) * (z - Zpre)

        self.P = Ppre - K @ H @ Ppre

        # 更新当前步
        self.current_step += 1

        return self.x, self.P


class BearingOnlyUKF:
    """增强版无迹卡尔曼滤波器用于纯方位目标运动分析"""

    def __init__(self, x0, P0, Q, R, dt, observer_trajectory, measurements, backward=False):
        """
        说明：假设你要进行n次卡尔曼滤波递推，那么observer_trajectory和measurements应该包含从初始状态和往后n个状态下的量测和坐标

        :param x0: 初始状态向量 [x, y, vx, vy]
        :param P0: 初始化协方差矩阵
        :param Q: 过程噪声协方差矩阵
        :param R: 测量噪声协方差 (标量，弧度)
        :param dt: 时间步长（采样周期）
        :param observer_trajectory: 传感器轨迹，每行为一个时间步的位置 [x, y]，长度为n+1
        :param measurements: 量测方位序列（弧度）
        :param backward: 是否逆向滤波
        """

        self.n = len(x0)  # 状态维度
        self.x = x0.copy()  # 状态向量
        self.P = P0.copy()  # 协方差矩阵
        self.Q = Q.copy()  # 过程噪声协方差矩阵
        self.R = R  # 测量噪声协方差 (标量)
        self.dt = dt  # 时间步长
        self.backward = backward

        if self.backward:
            self.observer_trajectory = observer_trajectory[::-1]
            self.measurements = measurements[::-1]
        else:
            self.observer_trajectory = observer_trajectory
            self.measurements = measurements

        if len(self.observer_trajectory) != len(self.measurements):
            raise ValueError("传入的坐标序列和方位序列不等长")

        self.current_step = 1  # 当前步

        # UKF参数
        self.alpha = 0.3  # 控制sigma点的散布程度
        self.beta = 2.0  # 先验分布的最优值 (2表示高斯分布)
        self.kappa = 0  # 次要参数，通常设为0

        # 计算缩放参数
        self.lambda_ = self.alpha ** 2 * (self.n + self.kappa) - self.n

        # 权重参数计算
        self.compute_weights()

        # 存储诊断信息
        self.innovation = []  # 保存创新序列
        self.innovation_covariance = []  # 保存创新协方差
        self.nees = []  # 归一化估计误差平方 (NEES)
        self.nis = []  # 归一化创新平方 (NIS)

    def compute_weights(self):
        """计算sigma点的权重"""
        # 计算权重系数
        self.weights_m = np.zeros(2 * self.n + 1)  # 均值权重
        self.weights_c = np.zeros(2 * self.n + 1)  # 协方差权重

        # 中心点权重
        self.weights_m[0] = self.lambda_ / (self.n + self.lambda_)
        self.weights_c[0] = self.weights_m[0] + (1 - self.alpha ** 2 + self.beta)

        # 其余点权重
        for i in range(1, 2 * self.n + 1):
            self.weights_m[i] = 1 / (2 * (self.n + self.lambda_))
            self.weights_c[i] = self.weights_m[i]

    def generate_sigma_points(self):
        """生成sigma点"""
        try:
            # 计算矩阵平方根
            L = self.n + self.lambda_

            # 确保协方差矩阵是对称的
            #self.P = (self.P + self.P.T) / 2

            # 检查正定性
            if not np.all(np.linalg.eigvals(self.P) > 0):
                # 如果不是正定的，添加一个小的对角矩阵
                min_eig = np.min(np.linalg.eigvals(self.P))
                if min_eig < 0:
                    self.P -= 1.1 * min_eig * np.eye(self.n)

            sqrt_P = np.linalg.cholesky((L * self.P).astype(float))

            # 创建sigma点矩阵
            sigma_points = np.zeros((2 * self.n + 1, self.n))
            sigma_points[0] = self.x

            for i in range(self.n):
                #sigma_points[i + 1] = self.x + sqrt_P[i]
                #sigma_points[i + 1 + self.n] = self.x - sqrt_P[i]
                sigma_points[i + 1] = self.x + sqrt_P[:, i]
                sigma_points[i + 1 + self.n] = self.x - sqrt_P[:, i]

            return sigma_points

        except np.linalg.LinAlgError:
            # 如果Cholesky分解失败，使用特征值分解作为备选方案
            print("警告：Cholesky分解失败，使用特征值分解代替")
            eigvals, eigvecs = np.linalg.eigh(self.P)
            eigvals = np.maximum(eigvals, 1e-6)  # 确保所有特征值为正
            sqrt_P = eigvecs @ np.diag(np.sqrt(L * eigvals)) @ eigvecs.T

            sigma_points = np.zeros((2 * self.n + 1, self.n))
            sigma_points[0] = self.x

            for i in range(self.n):
                sigma_points[i + 1] = self.x + sqrt_P[:, i]
                sigma_points[i + 1 + self.n] = self.x - sqrt_P[:, i]

            return sigma_points

    def state_transition(self, x, add_noise=False):
        """
        状态转移函数 - 匀速直线运动模型
        x = [位置x, 位置y, 速度vx, 速度vy]
        """
        if self.backward:
            # 逆向状态转移矩阵
            F = np.array([
                [1, 0, -self.dt, 0],
                [0, 1, 0, -self.dt],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])
        else:
            # 正向状态转移矩阵
            F = np.array([
                [1, 0, self.dt, 0],
                [0, 1, 0, self.dt],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])

        new_x = F @ x

        # 添加过程噪声
        if add_noise:
            noise = np.random.multivariate_normal(np.zeros(self.n), self.Q)
            new_x += noise

        return new_x

    def measurement_function(self, x, step=None):
        """
        测量函数 - 计算从观测者到目标的方位角
        返回以弧度表示的方位角
        """
        if step is None:
            step = self.current_step

        observer_pos = self.observer_trajectory[step]
        dx = x[0] - observer_pos[0]
        dy = x[1] - observer_pos[1]
        bearing = np.arctan2(dx, dy)

        return np.array([bearing])

    def normalize_angle(self, angle):
        """将角度归一化到[-pi, pi]范围内"""
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def predict(self):
        """UKF预测步骤"""
        # 生成sigma点
        sigma_points = self.generate_sigma_points()

        # 传播sigma点
        sigma_points_pred = np.array([self.state_transition(sigma, add_noise=False) for sigma in sigma_points])

        # 计算预测状态
        x_pred = np.sum(self.weights_m.reshape(-1, 1) * sigma_points_pred, axis=0)

        # 计算预测协方差
        P_pred = np.zeros((self.n, self.n))
        for i in range(len(sigma_points_pred)):
            diff = sigma_points_pred[i] - x_pred
            P_pred += self.weights_c[i] * np.outer(diff, diff)

        # 添加过程噪声协方差
        P_pred += self.Q

        # 更新状态和协方差
        self.x = x_pred
        self.P = P_pred

        return sigma_points_pred

    def update(self, sigma_points_pred):
        """UKF更新步骤"""
        # 预测测量值
        z_pred = np.array([self.measurement_function(x) for x in sigma_points_pred])

        # 计算预测测量值的均值
        z_mean = np.sum(self.weights_m.reshape(-1, 1) * z_pred, axis=0)

        # 计算测量预测协方差
        P_zz = 0
        for i in range(len(z_pred)):
            diff = z_pred[i] - z_mean
            diff[0] = self.normalize_angle(diff[0])
            P_zz += self.weights_c[i] * diff[0] ** 2

        # 添加测量噪声
        P_zz += self.R

        # 计算状态与测量的互相关矩阵
        P_xz = np.zeros(self.n)
        for i in range(len(sigma_points_pred)):
            diff_x = sigma_points_pred[i] - self.x
            diff_z = z_pred[i] - z_mean
            diff_z[0] = self.normalize_angle(diff_z[0])
            P_xz += self.weights_c[i] * diff_x * diff_z[0]

        # 计算卡尔曼增益
        K = P_xz / P_zz

        # 计算测量残差
        z = self.measurements[self.current_step]  # 获取当前回合的量测方位角
        z_residual = z - z_mean
        z_residual[0] = self.normalize_angle(z_residual[0])

        # 保存创新和创新协方差用于诊断
        self.innovation.append(z_residual[0])
        self.innovation_covariance.append(P_zz)

        # 计算归一化创新平方 (NIS)
        nis = z_residual[0] ** 2 / P_zz
        self.nis.append(nis)

        # 更新状态和协方差
        self.x += K * z_residual[0]
        self.P -= np.outer(K, K) * P_zz

        # 确保协方差矩阵保持对称
        self.P = (self.P + self.P.T) / 2

    def step(self, true_state=None):
        """执行完整的UKF步骤：预测和更新"""
        # 预测步骤
        sigma_points_pred = self.predict()

        # 更新步骤
        self.update(sigma_points_pred)

        # 更新当前步
        self.current_step += 1

        return self.x, self.P


class BearingOnlyCKF:
    """
    使用立方卡尔曼滤波器(CKF)进行纯方位目标运动分析
    """

    def __init__(self, x0, P0, Q, R, dt, observer_trajectory, measurements, backward=False):
        """
        说明：假设你要进行n次卡尔曼滤波递推，那么observer_trajectory和measurements应该包含从初始状态和往后n个状态下的量测和坐标

        :param x0: 初始状态向量 [x, y, vx, vy]
        :param P0: 初始化协方差矩阵
        :param Q: 过程噪声协方差矩阵
        :param R: 测量噪声协方差 (标量，弧度)
        :param dt: 时间步长（采样周期）
        :param observer_trajectory: 传感器轨迹，每行为一个时间步的位置 [x, y]，长度为n+1
        :param measurements: 量测方位序列（弧度）
        :param backward: 是否逆向滤波
        """

        self.n = len(x0)  # 状态维度
        self.x = x0.copy()  # 状态向量
        self.P = P0.copy()  # 协方差矩阵
        self.Q = Q.copy()  # 过程噪声协方差矩阵
        self.R = R  # 测量噪声协方差 (标量)
        self.dt = dt  # 时间步长

        self.backward = backward

        if self.backward:
            self.observer_trajectory = observer_trajectory[::-1]
            self.measurements = measurements[::-1]
        else:
            self.observer_trajectory = observer_trajectory
            self.measurements = measurements

        if len(self.observer_trajectory) != len(self.measurements):
            raise ValueError("传入的坐标序列和方位序列不等长")

        self.current_step = 1  # 当前步（用于索引observer_trajectory和measurements中对应元素）
        # 从 1 开始是因为初始量测值并不参与卡尔曼滤波迭代更新

        # CKF参数
        self.num_points = 2 * self.n  # CKF使用2n个立方点
        self.weight = 1.0 / (2 * self.n)  # 所有点的权重相等

        # 保存诊断信息
        self.bearing_pred_error = []  # 保存方位角预测误差
        self.innovation_covariance = []  # 保存P阵


    def generate_cubature_points(self):
        """生成立方点"""
        # 计算矩阵平方根
        try:
            # 确保协方差矩阵是对称的，这步没必要因为P一般都是对称的
            #self.P = (self.P + self.P.T) / 2

            # 检查正定性
            if not np.all(np.linalg.eigvals(self.P) > 0):
                # 如果不是正定的，添加一个小的对角矩阵
                min_eig = np.min(np.linalg.eigvals(self.P))
                if min_eig < 0:
                    self.P -= 1.1 * min_eig * np.eye(self.n)

            sqrt_P = np.linalg.cholesky(self.P)

            # 生成单位方向向量（立方点方向）
            # CKF使用2n个立方点，位于单位超球面上
            directions = np.zeros((2 * self.n, self.n))
            for i in range(self.n):
                directions[i, i] = 1.0  # [1,0,...,0], [0,1,...,0], ...
                directions[i + self.n, i] = -1.0  # [-1,0,...,0], [0,-1,...,0], ...

            # 缩放单位方向向量为立方点
            cubature_points = np.zeros((2 * self.n, self.n))
            scaled_directions = np.sqrt(self.n) * directions

            # 应用换元公式生成完整立方点
            for i in range(2 * self.n):
                cubature_points[i] = self.x + sqrt_P @ scaled_directions[i]

            return cubature_points

        except np.linalg.LinAlgError:
            # 如果Cholesky分解失败，使用特征值分解作为备选方案
            print("警告：Cholesky分解失败，使用特征值分解代替")
            eigvals, eigvecs = np.linalg.eigh(self.P)
            eigvals = np.maximum(eigvals, 1e-6)  # 确保所有特征值为正
            sqrt_P = eigvecs @ np.diag(np.sqrt(eigvals))

            # 生成单位方向向量
            directions = np.zeros((2 * self.n, self.n))
            for i in range(self.n):
                directions[i, i] = 1.0
                directions[i + self.n, i] = -1.0

            # 缩放单位方向向量为立方点
            cubature_points = np.zeros((2 * self.n, self.n))
            scaled_directions = np.sqrt(self.n) * directions

            # 应用换元公式生成完整立方点
            for i in range(2 * self.n):
                cubature_points[i] = self.x + sqrt_P @ scaled_directions[i]

            return cubature_points

    def state_transition(self, x, add_noise=False):
        """
        状态转移函数 - 匀速直线运动模型
        x = [位置x, 位置y, 速度vx, 速度vy]
        """
        if self.backward:
            # 逆向状态转移矩阵
            F = np.array([
                [1, 0, -self.dt, 0],
                [0, 1, 0, -self.dt],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])
        else:
            # 正向状态转移矩阵
            F = np.array([
                [1, 0, self.dt, 0],
                [0, 1, 0, self.dt],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])

        new_x = F @ x

        # 添加过程噪声
        if add_noise:
            noise = np.random.multivariate_normal(np.zeros(self.n), self.Q)
            new_x += noise

        return new_x

    def measurement_function(self, x, step=None):
        """
        测量函数 - 计算从观测者到目标的方位角
        返回以弧度表示的方位角
        """
        if step is None:
            step = self.current_step

        observer_pos = self.observer_trajectory[step]
        dx = x[0] - observer_pos[0]
        dy = x[1] - observer_pos[1]
        bearing = np.arctan2(dx, dy)

        return np.array([bearing])

    def normalize_angle(self, angle):
        """将角度归一化到[-pi, pi]范围内"""
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def predict(self):
        """CKF预测步骤"""
        # 生成立方点
        cubature_points = self.generate_cubature_points()

        # 传播立方点
        propagated_points = np.array([self.state_transition(x, add_noise=False) for x in cubature_points])

        # 计算预测状态均值
        x_pred = np.sum(propagated_points * self.weight, axis=0)

        # 计算预测状态协方差
        P_pred = np.zeros((self.n, self.n))
        for i in range(len(propagated_points)):
            diff = propagated_points[i] - x_pred
            P_pred += self.weight * np.outer(diff, diff)

        # 添加过程噪声协方差
        P_pred += self.Q

        # 更新状态和协方差
        self.x = x_pred
        self.P = P_pred

        return propagated_points

    def update(self, propagated_points):
        """CKF更新步骤"""
        # 通过测量函数变换传播点
        z_points = np.array([self.measurement_function(x) for x in propagated_points])

        # 计算预测测量均值
        z_pred = np.sum(z_points * self.weight, axis=0)

        # 计算预测测量协方差
        P_zz = 0
        for i in range(len(z_points)):
            diff = z_points[i] - z_pred
            diff[0] = self.normalize_angle(diff[0])
            P_zz += self.weight * diff[0] ** 2

        # 添加测量噪声协方差
        P_zz += self.R

        # 计算状态与测量的互相关矩阵
        P_xz = np.zeros(self.n)
        for i in range(len(propagated_points)):
            diff_x = propagated_points[i] - self.x
            diff_z = z_points[i] - z_pred
            diff_z[0] = self.normalize_angle(diff_z[0])
            P_xz += self.weight * diff_x * diff_z[0]

        # 计算卡尔曼增益
        K = P_xz / P_zz

        # 计算测量残差（创新）
        z = self.measurements[self.current_step]    # 获取当前回合的量测方位角
        z_residual = z - z_pred
        z_residual[0] = self.normalize_angle(z_residual[0])

        # 保存创新和创新协方差用于诊断
        self.bearing_pred_error.append(z_residual[0])

        self.innovation_covariance.append(P_zz)



        # 更新状态和协方差
        self.x += K * z_residual[0]
        self.P -= np.outer(K, K) * P_zz

        # 确保协方差矩阵保持对称
        self.P = (self.P + self.P.T) / 2

    def step(self):
        """执行完整的CKF步骤：预测和更新"""
        # 预测步骤
        propagated_points = self.predict()

        # 更新步骤
        self.update(propagated_points)

        # 更新当前步
        self.current_step += 1

        return self.x, self.P


class Algorithms:

    def __init__(self, model):
        self.model = model

    def run_ekf(self, x0, P0, Q):

        positions = model.sensor_trajectory[: model.steps + 1]
        measurements = model.measurements[: model.steps + 1]
        fliter = BearingOnlyEKF(x0, P0, Q, model.R, model.sample_time, positions, measurements)

        # 运行滤波
        estimated_states = np.zeros((model.steps + 1, 4))
        estimated_covs = np.zeros((model.steps + 1, 4, 4))

        estimated_states[0] = x0
        estimated_covs[0] = P0

        for k in range(1, model.steps + 1):
            fliter.step()
            estimated_states[k] = fliter.x
            estimated_covs[k] = fliter.P

        state_err = (self.model.target_states - estimated_states) ** 2

        return {'color': 'grey',
                'states': estimated_states,
                'covs': estimated_covs,
                'square_error': state_err,
                }

    def run_plkf(self, x0, P0, Q):

        positions = model.sensor_trajectory[: model.steps + 1]
        measurements = model.measurements[: model.steps + 1]
        fliter = BearingOnlyPLKF(x0, P0, Q, model.R, model.sample_time, positions, measurements)

        # 运行滤波
        estimated_states = np.zeros((model.steps + 1, 4))
        estimated_covs = np.zeros((model.steps + 1, 4, 4))

        estimated_states[0] = x0
        estimated_covs[0] = P0

        for k in range(1, model.steps + 1):
            fliter.step()
            estimated_states[k] = fliter.x
            estimated_covs[k] = fliter.P

        state_err = (self.model.target_states - estimated_states) ** 2

        return {'color': 'grey',
                'states': estimated_states,
                'covs': estimated_covs,
                'square_error': state_err,
                }

    def run_ukf(self, x0, P0, Q):

        positions = model.sensor_trajectory[: model.steps + 1]
        measurements = model.measurements[: model.steps + 1]
        fliter = BearingOnlyUKF(x0, P0, Q, model.R, model.sample_time, positions, measurements)

        # 运行滤波
        estimated_states = np.zeros((model.steps + 1, 4))
        estimated_covs = np.zeros((model.steps + 1, 4, 4))

        estimated_states[0] = x0
        estimated_covs[0] = P0

        for k in range(1, model.steps + 1):
            fliter.step()
            estimated_states[k] = fliter.x
            estimated_covs[k] = fliter.P

        state_err = (self.model.target_states - estimated_states) ** 2

        return {'color': 'red',
                'states': estimated_states,
                'covs': estimated_covs,
                'square_error': state_err,
                }

    def run_ckf(self, x0, P0, Q):

        positions = model.sensor_trajectory[: model.steps + 1]
        measurements = model.measurements[: model.steps + 1]
        fliter = BearingOnlyCKF(x0, P0, Q, model.R, model.sample_time, positions, measurements)

        # 运行滤波
        estimated_states = np.zeros((model.steps + 1, 4))
        estimated_covs = np.zeros((model.steps + 1, 4, 4))

        estimated_states[0] = x0
        estimated_covs[0] = P0

        for k in range(1, model.steps + 1):
            fliter.step()
            estimated_states[k] = fliter.x
            estimated_covs[k] = fliter.P

        state_err = (self.model.target_states - estimated_states) ** 2

        return {'color': 'blue',
                'states': estimated_states,
                'covs': estimated_covs,
                'square_error': state_err,
                }

    def run_frckf(self, x0, P0, Q, rev_start_step):
        """
        单次正逆向滤波

        :param x0:
        :param P0:
        :param Q:
        :param rev_start_step:
        :return:
        """

        # 创建数组用于存储解算结果（估算状态Xest和协方差矩阵P）
        estimated_states = np.zeros((model.steps + 1, 4))
        estimated_covs = np.zeros((model.steps + 1, 4, 4))

        # 正向滤波到rev_start_step
        part_frckf_positions = model.sensor_trajectory[:(rev_start_step + 1)]
        part_frckf_measurements = model.measurements[:(rev_start_step + 1)]

        part_forward_ckf = BearingOnlyCKF(x0, P0, Q, model.R, model.sample_time,
                                          part_frckf_positions, part_frckf_measurements)

        # 进行正向滤波迭代
        for i in range(1, rev_start_step + 1):
            part_forward_ckf.step()

        # 获取逆向滤波初值
        reverse_ckf_init_state = part_forward_ckf.x
        reverse_ckf_init_cov = part_forward_ckf.P

        # 初始化逆向滤波
        part_reverse_ckf = BearingOnlyCKF(reverse_ckf_init_state, reverse_ckf_init_cov, Q, model.R, model.sample_time,
                                          part_frckf_positions, part_frckf_measurements, backward=True)

        # 进行逆向滤波迭代
        for i in range(1, rev_start_step + 1):
            part_reverse_ckf.step()

        # 获取逆向滤波终值作为优化初值
        optimized_initial_state = part_reverse_ckf.x
        optimized_initial_cov = part_reverse_ckf.P

        estimated_states[0] = optimized_initial_state
        estimated_covs[0] = optimized_initial_cov

        # 基于优化初值进行全局正向滤波
        forward_ckf = BearingOnlyCKF(optimized_initial_state, optimized_initial_cov, Q, model.R, model.sample_time,
                                          model.sensor_trajectory, model.measurements)

        # 存储结果，这里最好不要用 i
        for k in range(1, model.steps + 1):
            forward_ckf.step()
            estimated_states[k] = forward_ckf.x
            estimated_covs[k] = forward_ckf.P

        state_err = (self.model.target_states - estimated_states) ** 2

        return {'color': 'green',
                'states': estimated_states,
                'covs': estimated_covs,
                'square_error': state_err,
                }

    def run_frfrckf(self, x0, P0, Q, rev_start_step, short_rev_step_length):
        """
        长短正逆向容积卡尔曼滤波

        :param x0:
        :param P0:
        :param Q:
        :param rev_start_step:
        :param short_rev_step_length:
        :return:
        """

        estimated_states = np.zeros((model.steps + 1, 4))
        estimated_covs = np.zeros((model.steps + 1, 4, 4))

        """
           进行一次到k = rev_start_step轮次的长正逆向滤波
           (1)>正向滤波到 k 环节
           (2)>以(1)的x[k]和P[k]开始，逆向滤波short_rev_step_length次
           (3)>基于(2)中逆向滤波的结果，再正向滤波回到k+1环节，(3)的结果即为后面回合的最终估算结果
           """

        # 正向滤波到rev_start_step环节

        long_frckf_positions = model.sensor_trajectory[:(rev_start_step + 1)]
        long_frckf_measurements = model.measurements[:(rev_start_step + 1)]

        long_forward_ckf = BearingOnlyCKF(x0, P0, Q, model.R, model.sample_time,
                                       long_frckf_positions, long_frckf_measurements)


        for i in range(1, rev_start_step + 1):
            long_forward_ckf.step()

        # 从该环节开始逆向滤波优化初值
        long_reverse_ckf_init_state = long_forward_ckf.x
        long_forward_ckf_init_covs = long_forward_ckf.P

        long_reverse_ckf = BearingOnlyCKF(long_reverse_ckf_init_state, long_forward_ckf_init_covs, Q, model.R, model.sample_time,
                                       long_frckf_positions, long_frckf_measurements, backward=True)

        for i in range(1, rev_start_step + 1):
            long_reverse_ckf.step()

        optimized_initial_state = long_reverse_ckf.x
        optimized_initial_covariance = long_reverse_ckf.P

        estimated_states[0] = optimized_initial_state
        estimated_covs[0] = optimized_initial_covariance

        # 再正向滤波回到rev_start_step环节
        long_forward_ckf_again = BearingOnlyCKF(optimized_initial_state, optimized_initial_covariance, Q, model.R,
                                        model.sample_time, long_frckf_positions, long_frckf_measurements)

        for ii in range(1, rev_start_step + 1):
            long_forward_ckf_again.step()
            estimated_states[ii] = long_forward_ckf_again.x
            estimated_covs[ii] = long_forward_ckf_again.P

        """接下来的回合进行短正逆向滤波，重复以下操作：
           (1).从最后一个估计k开始，正向滤波到下一回合k+1
           (2).以(1)的x[k+1]和P[k+1]开始，向前逆向滤波short_rev_step_length次
           (3).基于(2)中逆向滤波的结果，再正向滤波回到k+1环节，(3)的结果即为后面回合的最终估算结果
           """

        for j in range(rev_start_step+1, model.steps+1):

            # 正向滤波到下一回合
            one_step_ckf_init_state = estimated_states[j-1]
            one_step_ckf_init_cov = estimated_covs[j-1]

            one_step_position = model.sensor_trajectory[j-1: j+1]   # 获取第j个元素
            one_step_measurement = model.measurements[j-1: j+1]

            one_step_ckf = BearingOnlyCKF(one_step_ckf_init_state, one_step_ckf_init_cov, Q, model.R,
                                        model.sample_time, one_step_position, one_step_measurement)

            one_step_ckf.step()

            # 向前逆向滤波short_rev_step_length个回合
            short_reverse_init_state = one_step_ckf.x
            short_reverse_init_cov = one_step_ckf.P

            short_frckf_position = model.sensor_trajectory[j - short_rev_step_length:j+1]
            short_frckf_measurement = model.measurements[j - short_rev_step_length:j+1]

            short_reverse_ckf = BearingOnlyCKF(short_reverse_init_state, short_reverse_init_cov, Q, model.R,
                                        model.sample_time, short_frckf_position, short_frckf_measurement,
                                               backward=True)

            for k in range(1, short_rev_step_length + 1):
                short_reverse_ckf.step()

            # 再正向滤波回到k+1时刻
            short_forward_init_state = short_reverse_ckf.x
            short_forward_init_cov = short_reverse_ckf.P

            short_forward_ckf = BearingOnlyCKF(short_forward_init_state, short_forward_init_cov, Q, model.R,
                                        model.sample_time, short_frckf_position, short_frckf_measurement)

            for k in range(1, short_rev_step_length + 1):
                short_forward_ckf.step()

            estimated_states[j] = short_forward_ckf.x
            estimated_covs[j] = short_forward_ckf.P

        state_err = (self.model.target_states - estimated_states) ** 2

        return {'color': 'purple',
                'states': estimated_states,
                'covs': estimated_covs,
                'square_error': state_err,
                }


class Runner:
    def __init__(self, algorithms):
        self.model = algorithms.model

        self.sensor_maneuver_types = {
            '8': self.model.generate_sensor_trajectory_8,
            'o': self.model.generate_sensor_trajectory_circle,
            's': self.model.generate_sensor_trajectory_s,

            'z': self.model.generate_sensor_trajectory_z,
        }

        self.method_name = None
        self.method_map = {
            "ekf": algorithms.run_ekf,
            "plkf": algorithms.run_plkf,
            "ukf": algorithms.run_ukf,
            "ckf": algorithms.run_ckf,
            "frckf": algorithms.run_frckf,
            'frfrckf': algorithms.run_frfrckf
        }

        self.result = []

        self.measurements_generated = False  # 标记是否已生成测量数据
        self.mc_iterations = 0  # 蒙特卡洛迭代次数
        self.all_measurements = []  # 存储所有迭代的测量值

    def select_maneuver_type(self, type):
        self.maneuver_type = type
        # 检查方法是否有效
        if type not in self.sensor_maneuver_types:
            # 提取所有支持的算法名称，用逗号分隔
            supported_methods = ", ".join(sorted(self.sensor_maneuver_types.keys()))
            raise ValueError(
                f"Unknown maneuver: '{type}'. Supported maneuver types are: {supported_methods}"
            )

    def generate_monte_carlo_data(self, num):
        """
        为蒙特卡洛仿真生成固定的测量数据集

        :param num: 蒙特卡洛仿真次数
        """
        if self.measurements_generated:
            return  # 如果已经生成数据，直接返回

        # 生成轨迹（仅需一次）
        selected_maneuver = self.sensor_maneuver_types[self.model.Sensor.maneuver]

        selected_maneuver()
        self.model.generate_target_trajectory()
        self.model.generate_bearings()
        self.model.generate_crlb()

        print(f'正在生成{num}次蒙特卡洛仿真的测量数据...')

        # 保存所有迭代的测量值
        self.all_measurements = []
        for i in tqdm(range(num),
                      desc="数据生成进度",
                      unit="次",
                      bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{remaining}]"):
            # 生成新的含噪方位测量
            self.model.generate_measurements()
            # 保存本次迭代的测量值
            self.all_measurements.append(self.model.measurements.copy())

        self.measurements_generated = True
        self.mc_iterations = num
        print(f'测量数据生成完毕')

    def select_method(self, method):
        self.method_name = method
        # 检查方法是否有效
        if method not in self.method_map:
            # 提取所有支持的算法名称，用逗号分隔
            supported_methods = ", ".join(sorted(self.method_map.keys()))
            raise ValueError(
                f"Unknown method: '{method}'. Supported methods are: {supported_methods}"
            )

    def run_monte_carlo(self, num=None):
        """
        使用已经生成的数据集运行蒙特卡洛仿真

        :param num: 可选，如果提供则重新生成数据集
        """
        if num is not None and (not self.measurements_generated or num != self.mc_iterations):
            self.measurements_generated = False
            self.generate_monte_carlo_data(num)
        elif not self.measurements_generated:
            raise ValueError("请先调用 generate_monte_carlo_data 生成数据集")

        # 获取对应的函数
        target_method = self.method_map[self.method_name]

        # 初始状态设置
        x0 = np.array([4000.0, 4000.0, 0, -0])
        P0 = np.diag([100.0 ** 2, 100.0 ** 2, 1.0 ** 2, 50.0 ** 2])
        Q = np.diag([0.1, 0.1, 0.01, 0.01])

        estimation_all = []
        square_error_all = []

        print(f'对{self.method_name}方法进行{self.mc_iterations}次蒙特卡洛仿真')
        start_time = time.time()

        for i in tqdm(range(self.mc_iterations),
                      desc="仿真进度",
                      unit="次",
                      bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{remaining}]"):

            # 使用保存的测量值
            self.model.measurements = self.all_measurements[i].copy()

            reverse_step = 600
            partical_rev_step = 1

            if self.method_name == 'frckf':
                result = target_method(x0, P0, Q, reverse_step)
            elif self.method_name == 'frffckf':
                result = target_method(x0, P0, Q, reverse_step, partical_rev_step)
            elif self.method_name == 'frfrckf':
                result = target_method(x0, P0, Q, reverse_step, partical_rev_step)
            else:
                result = target_method(x0, P0, Q)

            estimation_all.append(result['states'])
            square_error_all.append(result['square_error'])

        time1 = time.time() - start_time
        print(f'仿真结束，用时{time1:.2f}')

        avg_estimation = np.mean(estimation_all, axis=0)

        avg_rse = np.sqrt((self.model.target_states - avg_estimation) ** 2)

        avg_pos_rmse = np.sqrt(np.sum((self.model.target_states[:, 0:2] - avg_estimation[:, 0:2]) ** 2, axis=1))
        avg_vel_rmse = np.sqrt(np.sum((self.model.target_states[:, 2:4] - avg_estimation[:, 2:4]) ** 2, axis=1))

        # 计算真实航向角
        true_crs = np.arctan2(self.model.target_states[:, 2], self.model.target_states[:, 3])  # arctan2(vx, vy)

        # 计算估计航向角
        avg_estimated_crs = np.arctan2(avg_estimation[:, 2], avg_estimation[:, 3])  # arctan2(vx, vy)

        # 计算航向误差（考虑角度的周期性）
        # 这种方法可以正确处理如359°和1°之间的差异（应该是2°而非358°）
        avg_crs_rmse = np.arctan2(np.sin(true_crs - avg_estimated_crs),
                                  np.cos(true_crs - avg_estimated_crs))
        avg_crs_rmse = np.rad2deg(avg_crs_rmse)

        rmse = np.sqrt(np.mean(square_error_all, axis=0))

        pos_rmse = np.sqrt(np.sum(rmse[:, :2]**2, axis=1))

        crs_error = []
        for arr in estimation_all:
            vx = arr[:, 2]
            vy = arr[:, 3]
            estimated_crs = np.arctan2(vx, vy)
            error = estimated_crs - true_crs
            error = np.arctan2(np.sin(true_crs - estimated_crs),
                               np.cos(true_crs - estimated_crs))

            crs_error.append(error)

        crs_rmse = np.mean(crs_error, axis=0)
        crs_rmse = np.rad2deg(crs_rmse)

        spd_rmse = np.sqrt(np.sum(rmse[:, 2:]**2, axis=1))

        runner_result = {'name': self.method_name,
                         'num': self.mc_iterations,
                         'time': self.model.times,
                         'sensor_traj': self.model.sensor_trajectory,
                         'true_state': self.model.target_states,
                         'color': result['color'],
                         'crlb': self.model.crlb,
                         'estimation': avg_estimation,
                         'avg_x_rmse': avg_rse[:, 0],
                         'avg_y_rmse': avg_rse[:, 1],
                         'avg_vx_rmse': avg_rse[:, 2],
                         'avg_vy_rmse': avg_rse[:, 3],
                         'avg_pos_rmse': avg_pos_rmse,
                         'avg_vel_rmse': avg_crs_rmse,
                         'avg_crs_rmse': avg_vel_rmse,
                         'x_rmse': rmse[:, 0],
                         'y_rmse': rmse[:, 1],
                         'vx_rmse': rmse[:, 2],
                         'vy_rmse': rmse[:, 3],
                         'pos_rmse': pos_rmse,
                         'crs_rmse': crs_rmse,
                         'spd_rmse': spd_rmse,
                         }

        self.result.append(runner_result)

        return avg_pos_rmse, avg_vel_rmse

class Visulation:

    def __init__(self, Runner):

        self.plot_result = Runner.result

    def plot_figure(self, crlb_analysis=True):

        num_of_methods_used = len(self.plot_result)

        if num_of_methods_used < 1:
            raise ValueError('未使用任何方法进行仿真！')

        # 绘制完整的真实轨迹和观测者轨迹
        true_states = self.plot_result[0]['true_state']
        sensor_trajectory = self.plot_result[0]['sensor_traj']
        num = self.plot_result[0]['num']

        # 创建静态图
        plt.figure()

        # 绘制真实轨迹和估计轨迹

        plt.plot(true_states[:, 0], true_states[:, 1], 'y-', label='真实轨迹')
        plt.plot(sensor_trajectory[:, 0], sensor_trajectory[:, 1], 'k-', label='观测者轨迹')

        for i in range(num_of_methods_used):
            estimation = self.plot_result[i]['estimation']
            name = self.plot_result[i]['name']
            color = self.plot_result[i]['color']
            plt.plot(estimation[:, 0], estimation[:, 1], color=color, label=f"{name}算法{num}次平均估计轨迹")

        plt.axis('equal')
        plt.grid(True)
        plt.xlabel('X 位置 (m)')
        plt.ylabel('Y 位置 (m)')
        plt.title(f'{num}次Monte Carlo仿真的平均轨迹')
        plt.legend()
        plt.show()

        # 绘制XRMSE
        crlb = self.plot_result[0]['crlb']
        crlb_x = np.sqrt(crlb[:, 0])
        crlb_y = np.sqrt(crlb[:, 1])
        crlb_vx = np.sqrt(crlb[:, 2])
        crlb_vy = np.sqrt(crlb[:, 3])

        plt.figure()
        times_range = self.plot_result[0]['time']
        plt.subplot(1, 2, 1)
        for i in range(num_of_methods_used):
            name = self.plot_result[i]['name']
            color = self.plot_result[i]['color']
            rmse = self.plot_result[i]['x_rmse']
            plt.plot(times_range, rmse, color=color, label=f"{name}算法{num}次仿真 x RMSE")

        if crlb_analysis:
            plt.plot(times_range, crlb_x, color='black', label='状态估计x CRLB')
            plt.ylim(0, 1000)

        plt.xlabel('时间 (s)')
        plt.ylabel('位置x误差 (m)')
        plt.title('位置x估计RMSE')
        plt.legend()

        plt.subplot(1, 2, 2)
        for i in range(num_of_methods_used):
            name = self.plot_result[i]['name']
            color = self.plot_result[i]['color']
            avg_rse = self.plot_result[i]['avg_x_rmse']
            plt.plot(times_range, avg_rse, color=color, label=f"{name}算法{num}次仿真平均位置x误差")

        if crlb_analysis:
            plt.plot(times_range, crlb_x, color='black', label='状态估计x CRLB')
            plt.ylim(0, 1000)

        plt.xlabel('时间 (s)')
        plt.ylabel('位置x误差 (m)')
        plt.title('位置x平均估计误差')
        plt.legend()
        plt.show()

        # Y
        plt.figure()
        times_range = self.plot_result[0]['time']
        plt.subplot(1, 2, 1)
        for i in range(num_of_methods_used):
            name = self.plot_result[i]['name']
            color = self.plot_result[i]['color']
            rmse = self.plot_result[i]['y_rmse']
            plt.plot(times_range, rmse, color=color, label=f"{name}算法{num}次仿真 y RMSE")

        if crlb_analysis:
            plt.plot(times_range, crlb_y, color='black', label='状态估计y CRLB')
            plt.ylim(0, 1000)

        plt.xlabel('时间 (s)')
        plt.ylabel('位置y误差 (m)')
        plt.title('位置y估计RMSE')
        plt.legend()

        plt.subplot(1, 2, 2)
        for i in range(num_of_methods_used):
            name = self.plot_result[i]['name']
            color = self.plot_result[i]['color']
            avg_rse = self.plot_result[i]['avg_y_rmse']
            plt.plot(times_range, avg_rse, color=color, label=f"{name}算法{num}次仿真平均位置y误差")

        if crlb_analysis:
            plt.plot(times_range, crlb_y, color='black', label='状态估计y CRLB')
            plt.ylim(0, 1000)

        plt.xlabel('时间 (s)')
        plt.ylabel('位置y误差 (m)')
        plt.title('位置y平均估计误差')
        plt.legend()
        plt.show()

        # Vx
        plt.figure()
        times_range = self.plot_result[0]['time']
        plt.subplot(1,2,1)
        for i in range(num_of_methods_used):
            name = self.plot_result[i]['name']
            color = self.plot_result[i]['color']
            rmse = self.plot_result[i]['vx_rmse']
            plt.plot(times_range, rmse, color=color, label=f"{name}算法{num}次仿真 vx RMSE")

        if crlb_analysis:
            plt.plot(times_range, crlb_vx, color='black', label='状态估计vx CRLB')
            plt.ylim(0, 5)

        plt.xlabel('时间 (s)')
        plt.ylabel('速度x误差 (m/s)')
        plt.title('速度x估计RMSE')
        plt.legend()

        plt.subplot(1, 2, 2)
        for i in range(num_of_methods_used):
            name = self.plot_result[i]['name']
            color = self.plot_result[i]['color']
            avg_rse = self.plot_result[i]['avg_vx_rmse']
            plt.plot(times_range, avg_rse, color=color, label=f"{name}算法{num}次仿真平均速度x误差")

        if crlb_analysis:
            plt.plot(times_range, crlb_vx, color='black', label='状态估计vx CRLB')
            plt.ylim(0, 5)

        plt.xlabel('时间 (s)')
        plt.ylabel('速度x误差 (m/s)')
        plt.title('速度x平均估计误差')
        plt.legend()
        plt.show()

        # Vy
        plt.figure()
        times_range = self.plot_result[0]['time']
        plt.subplot(1, 2, 1)
        for i in range(num_of_methods_used):
            name = self.plot_result[i]['name']
            color = self.plot_result[i]['color']
            rmse = self.plot_result[i]['vy_rmse']
            plt.plot(times_range, rmse, color=color, label=f"{name}算法{num}次仿真 vy RMSE")

        if crlb_analysis:
            plt.plot(times_range, crlb_vy, color='black', label='状态估计vy CRLB')
            plt.ylim(0, 5)

        plt.xlabel('时间 (s)')
        plt.ylabel('速度y误差 (m/s)')
        plt.title('速度y估计RMSE')
        plt.legend()

        plt.subplot(1, 2, 2)
        for i in range(num_of_methods_used):
            name = self.plot_result[i]['name']
            color = self.plot_result[i]['color']
            avg_rse = self.plot_result[i]['avg_vy_rmse']
            plt.plot(times_range, avg_rse, color=color, label=f"{name}算法{num}次仿真平均速度y误差")

        if crlb_analysis:
            plt.plot(times_range, crlb_vy, color='black', label='状态估计vy CRLB')
            plt.ylim(0, 5)

        plt.xlabel('时间 (s)')
        plt.ylabel('速度y误差 (m/s)')
        plt.title('速度y平均估计误差')
        plt.legend()
        plt.show()

        # 绘制位置RMSE
        plt.figure()

        times_range = self.plot_result[0]['time']
        plt.subplot(1,2,1)
        for i in range(num_of_methods_used):
            name = self.plot_result[i]['name']
            color = self.plot_result[i]['color']
            rmse = self.plot_result[i]['pos_rmse']
            plt.plot(times_range, rmse, color=color, label=f"{name}算法{num}次仿真 pos RMSE")

        plt.xlabel('时间 (s)')
        plt.ylabel('位置误差 (m)')
        plt.title('位置估计RMSE')
        plt.legend()

        plt.subplot(1, 2, 2)
        for i in range(num_of_methods_used):
            name = self.plot_result[i]['name']
            color = self.plot_result[i]['color']
            avg_rse = self.plot_result[i]['avg_pos_rmse']
            plt.plot(times_range, avg_rse, color=color, label=f"{name}算法{num}次仿真平均位置误差")

        plt.xlabel('时间 (s)')
        plt.ylabel('位置误差 (m)')
        plt.title('位置平均估计误差')
        plt.legend()
        plt.show()

        # 绘制速度RMSE
        plt.figure()
        times_range = self.plot_result[0]['time']
        plt.subplot(1, 2, 1)
        for i in range(num_of_methods_used):
            name = self.plot_result[i]['name']
            color = self.plot_result[i]['color']
            rmse = self.plot_result[i]['spd_rmse']
            plt.plot(times_range, rmse, color=color, label=f"{name}算法{num}次仿真 spd RMSE")
            plt.xlabel('时间 (s)')
            plt.ylabel('速度误差 (m/s)')
            plt.title('速度估计RMSE')
            plt.legend()
        plt.subplot(1, 2, 2)
        for i in range(num_of_methods_used):
            name = self.plot_result[i]['name']
            color = self.plot_result[i]['color']
            avg_rse = self.plot_result[i]['avg_vel_rmse']
            plt.plot(times_range, avg_rse, color=color, label=f"{name}算法{num}次仿真平均速度误差")
        plt.xlabel('时间 (s)')
        plt.ylabel('速度误差 (m/s)')
        plt.title('速度平均估计误差')
        plt.legend()
        plt.show()

        # 绘制航向RMSE
        plt.figure()
        times_range = self.plot_result[0]['time']
        plt.subplot(1,2,1)
        for i in range(num_of_methods_used):
            name = self.plot_result[i]['name']
            color = self.plot_result[i]['color']
            rmse = self.plot_result[i]['crs_rmse']
            plt.plot(times_range, rmse, color=color, label=f"{name}算法{num}次仿真 crs RMSE")
        plt.xlabel('时间 (s)')
        plt.ylabel('航向误差 (deg)')
        plt.title('航向估计RMSE')
        plt.legend()
        plt.subplot(1, 2, 2)
        for i in range(num_of_methods_used):
            name = self.plot_result[i]['name']
            color = self.plot_result[i]['color']
            avg_rse = self.plot_result[i]['avg_crs_rmse']
            plt.plot(times_range, avg_rse, color=color, label=f"{name}算法{num}次仿真平均速度误差")
        plt.xlabel('时间 (s)')
        plt.ylabel('航向误差 (deg)')
        plt.title('航向平均估计误差')
        plt.legend()
        plt.show()



if __name__ == '__main__':

    dt = 1
    maxt = 2000
    noise_mean = 0
    noise_std = 0.2
    Sensor = Point2D(np.array([0, 0, 180, 2]), mode='bdcv', acceleration=1, omega=1, maneuver='s')
    Target = Point2D(np.array([5000, 5000, 0, -5]), mode='xyvv', acceleration=1, omega=1)
    model = Model(Sensor, Target, dt=dt, maxt=maxt, brg_noise_mean=noise_mean, brg_noise_std=noise_std)

    algorithms = Algorithms(model)

    number = 1
    Runner = Runner(algorithms)

    # 生成固定的蒙特卡洛仿真数据
    Runner.generate_monte_carlo_data(number)

    Runner.select_method('ekf')
    Runner.run_monte_carlo()

    #Runner.select_method('plkf')
    #Runner.run_monte_carlo()

    #Runner.select_method('ukf')
    #Runner.run_monte_carlo()

    #Runner.select_method('ckf')
    #Runner.run_monte_carlo()

    #Runner.select_method('frckf')
    #Runner.run_monte_carlo()

    #Runner.select_method('frfrckf')
    #Runner.run_monte_carlo()

    Visulation = Visulation(Runner)
    Visulation.plot_figure()