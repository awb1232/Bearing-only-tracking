from angel_process import *


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

    def update_state(self):
        self.state[0] = self.position[0]
        self.state[1] = self.position[1]
        self.state[2] = self.speed * np.sin(np.deg2rad(self.course))
        self.state[3] = self.speed * np.cos(np.deg2rad(self.course))

    def uniform_linear_motion(self, dt):
        """
        执行一次匀速直线运动，

        :param dt: 模拟最小的时间步长
        :return: 无
        """
        delta_position = self.speed * np.array(
            [np.sin(np.deg2rad(self.course)), np.cos(np.deg2rad(self.course))]) * dt
        self.position += delta_position
        self.update_state()

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

        # 传感器状态
        self.sensor_trajectory = np.zeros((self.steps + 1, 2))  # 传感器轨迹
        self.sensor_states = np.zeros((self.steps + 1, 4))
        self.sensor_states[0] = self.Sensor.state

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

            self.sensor_states[k] = self.Sensor.state.copy()

    def generate_sensor_trajectory_s(self, main_time = 120, delta_crs=45, part_time = 60):

        main_crs = self.Sensor.course.copy()
        main_steps = int(main_time / self.sample_time)
        part_steps = int(part_time / self.sample_time)

        turn_stage = 0
        turn_flag = True
        line_step = 0

        for k in range(self.steps + 1):

            if k <= main_steps:
                self.Sensor.uniform_linear_motion(self.sample_time)

            else:

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

            self.sensor_states[k] = self.Sensor.state.copy()

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

    def generate_sensor_trajectory_z1(self):


        return 0

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