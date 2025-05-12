import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.animation as animation
from scipy.stats import chi2
import time
from tqdm import tqdm

plt.rcParams['font.sans-serif'] = ['SimSun']  # 使用宋体字体
plt.rcParams['axes.unicode_minus'] = False  # 显示负号


class Model:

    def __init__(self,
                 tgt,
                 dt,
                 maxt,
                 brg_noise_mean,
                 brg_noise_std,
                 ):
        self.target_init_state = tgt
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

    def generate_sensor_trajectory_8(self):
        """
        生成横”8“子型的曲线运动轨迹

        :return:
        """

        for k in range(self.steps + 1):
            t = k * self.sample_time
            self.sensor_trajectory[k, 0] = 100 * np.sin(t / 20)
            self.sensor_trajectory[k, 1] = 50 * np.sin(t / 10)

    def generate_target_trajectory(self):

        F = np.array([
            [1, 0, self.sample_time, 0],
            [0, 1, 0, self.sample_time],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        for k in range(1, self.steps + 1):
            self.target_states[k] = F @ self.target_states[k - 1]

    def generate_sensor_trajectory_circle(self, start_angle_deg=90, linear_speed=5, angular_speed_deg=1,
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
        # 将度转换为弧度
        start_angle_rad = np.deg2rad(start_angle_deg)
        angular_speed_rad = np.deg2rad(angular_speed_deg)  # 角速度转为弧度/秒

        # 如果是顺时针方向，将角速度取反
        if clockwise:
            angular_speed_rad = -angular_speed_rad

        # 计算圆的半径: r = v / |ω|
        radius = linear_speed / abs(angular_speed_rad)

        # 确定圆心坐标
        # 如果是逆时针，起始角度为90度时圆心在(-radius, 0)
        # 如果是顺时针，起始角度为90度时圆心在(radius, 0)
        if clockwise:
            center_x = radius * np.sin(start_angle_rad)
            center_y = -radius * np.cos(start_angle_rad)
        else:
            center_x = -radius * np.sin(start_angle_rad)
            center_y = radius * np.cos(start_angle_rad)

        # 生成圆周轨迹
        for k in range(self.steps + 1):
            t = k * self.sample_time
            # 当前角度 = 起始角度 + 角速度 * 时间
            angle = start_angle_rad + angular_speed_rad * t

            # 位置坐标
            self.sensor_trajectory[k, 0] = center_x + radius * np.cos(angle)
            self.sensor_trajectory[k, 1] = center_y + radius * np.sin(angle)

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


class Runner:
    def __init__(self, Model):
        self.model = Model
        self.maneuver_type = None
        self.sensor_maneuver_types = {
            '8': Model.generate_sensor_trajectory_8,
            'circle': Model.generate_sensor_trajectory_circle,
        }
        self.result = []

    def select_maneuver_type(self, type):
        self.maneuver_type = type
        # 检查方法是否有效
        if type not in self.sensor_maneuver_types:
            # 提取所有支持的算法名称，用逗号分隔
            supported_methods = ", ".join(sorted(self.sensor_maneuver_types.keys()))
            raise ValueError(
                f"Unknown maneuver: '{type}'. Supported maneuver types are: {supported_methods}"
            )

    def compute_crlb(self):

        selected_maneuver = self.sensor_maneuver_types[self.maneuver_type]

        selected_maneuver()

        self.model.generate_target_trajectory()

        self.model.generate_bearings()

        self.model.generate_crlb()

        runner_result = {
            'name': self.maneuver_type,
            'time': self.model.times,
            'true_state': self.model.target_states,
            'sensor_traj': self.model.sensor_trajectory,
            'crlb': self.model.crlb,
        }
        self.result.append(runner_result)


class Visulation:
    def __init__(self, Runner):
        self.result = Runner.result

    def plot_results(self):
        num_of_methods_used = len(self.result)

        if num_of_methods_used < 1:
            raise ValueError('未使用任何方法进行仿真！')

        # 绘制完整的真实轨迹和观测者轨迹
        true_states = self.result[0]['true_state']
        sensor_trajectory = self.result[0]['sensor_traj']

        # 创建静态图
        plt.figure()

        # 绘制真实轨迹和估计轨迹
        plt.plot(true_states[:, 0], true_states[:, 1], 'y-', label='跟踪目标轨迹')

        for i in range(num_of_methods_used):
            name = self.result[i]['name']
            sensor_trajectory = self.result[i]['sensor_traj']
            plt.plot(sensor_trajectory[:, 0], sensor_trajectory[:, 1], label=f'{name}机动传感器轨迹')
        plt.axis('equal')
        plt.grid(True)
        plt.xlabel('X 位置 (m)')
        plt.ylabel('Y 位置 (m)')
        plt.title(f'二维轨迹')
        plt.legend()
        plt.show()

        plt.figure()
        times_range = self.result[0]['time']
        for i in range(num_of_methods_used):
            name = self.result[i]['name']
            crlb = self.result[i]['crlb']
            rcrlbx = np.sqrt(crlb[:, 0])
            plt.plot(times_range, rcrlbx, label=f"{name}轨迹x CRLB")

        plt.xlabel('时间 (s)')
        plt.ylabel('位置xCRLB (m)')
        plt.ylim(0, 1000)
        plt.title('位置xCRLB比对')
        plt.legend()
        plt.show()

if __name__ == '__main__':
    target_init_state = np.array([5000, 5000, 0, -5])
    dt = 1
    maxt = 2000
    noise_mean = 0
    noise_std = 0.1
    model = Model(target_init_state, dt=dt, maxt=maxt, brg_noise_mean=noise_mean, brg_noise_std=noise_std)

    Runner = Runner(model)
    Runner.select_maneuver_type('8')
    Runner.compute_crlb()

    Runner.select_maneuver_type('circle')
    Runner.compute_crlb()

    Visulation = Visulation(Runner)
    Visulation.plot_results()
