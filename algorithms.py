from model import *
from kalman_fliters import *
from mle import *

class Algorithms:

    def __init__(self, model):
        self.model = model

    def ekf(self, x0, P0, Q):

        #positions = self.model.sensor_trajectory[: self.model.steps + 1]
        positions = self.model.sensor_states[: self.model.steps + 1]
        measurements = self.model.measurements[: self.model.steps + 1]
        fliter = BearingOnlyEKF(x0, P0, Q, self.model.R, self.model.sample_time, positions, measurements)

        # 运行滤波
        estimated_states = np.zeros((self.model.steps + 1, 4))
        estimated_covs = np.zeros((self.model.steps + 1, 4, 4))

        estimated_states[0] = x0
        estimated_covs[0] = P0

        for k in range(1, self.model.steps + 1):
            fliter.step()
            estimated_states[k] = fliter.x
            estimated_covs[k] = fliter.P

        state_err = (self.model.target_states - estimated_states) ** 2

        return {'color': 'grey',
                'states': estimated_states,
                'covs': estimated_covs,
                'square_error': state_err,
                }

    def plkf(self, x0, P0, Q):

        # positions = self.model.sensor_trajectory[: self.model.steps + 1]
        positions = self.model.sensor_states[: self.model.steps + 1]
        measurements = self.model.measurements[: self.model.steps + 1]
        fliter = BearingOnlyPLKF(x0, P0, Q, self.model.R, self.model.sample_time, positions, measurements)

        # 运行滤波
        estimated_states = np.zeros((self.model.steps + 1, 4))
        estimated_covs = np.zeros((self.model.steps + 1, 4, 4))

        estimated_states[0] = fliter.x
        estimated_covs[0] = fliter.P

        for k in range(1, self.model.steps + 1):
            fliter.step()
            estimated_states[k] = fliter.x
            estimated_covs[k] = fliter.P

        state_err = (self.model.target_states - estimated_states) ** 2

        return {'color': 'black',
                'states': estimated_states,
                'covs': estimated_covs,
                'square_error': state_err,
                }

    def ukf(self, x0, P0, Q):

        # positions = self.model.sensor_trajectory[: self.model.steps + 1]
        positions = self.model.sensor_states[: self.model.steps + 1]
        measurements = self.model.measurements[: self.model.steps + 1]
        fliter = BearingOnlyUKF(x0, P0, Q, self.model.R, self.model.sample_time, positions, measurements)

        # 运行滤波
        estimated_states = np.zeros((self.model.steps + 1, 4))
        estimated_covs = np.zeros((self.model.steps + 1, 4, 4))

        estimated_states[0] = x0
        estimated_covs[0] = P0

        for k in range(1, self.model.steps + 1):
            fliter.step()
            estimated_states[k] = fliter.x
            estimated_covs[k] = fliter.P

        state_err = (self.model.target_states - estimated_states) ** 2

        return {'color': 'cyan',
                'states': estimated_states,
                'covs': estimated_covs,
                'square_error': state_err,
                }

    def ckf(self, x0, P0, Q):

        # positions = self.model.sensor_trajectory[: self.model.steps + 1]
        positions = self.model.sensor_states[: self.model.steps + 1]
        measurements = self.model.measurements[: self.model.steps + 1]
        fliter = BearingOnlyCKF(x0, P0, Q, self.model.R, self.model.sample_time, positions, measurements)

        # 运行滤波
        estimated_states = np.zeros((self.model.steps + 1, 4))
        estimated_covs = np.zeros((self.model.steps + 1, 4, 4))

        estimated_states[0] = x0
        estimated_covs[0] = P0

        for k in range(1, self.model.steps + 1):
            fliter.step()
            estimated_states[k] = fliter.x
            estimated_covs[k] = fliter.P

        state_err = (self.model.target_states - estimated_states) ** 2

        return {'color': 'violet',
                'states': estimated_states,
                'covs': estimated_covs,
                'square_error': state_err,
                }

    def frkf(self, x0, P0, Q, *args):
        """

        :param x0:
        :param P0:
        :param Q:
        :param args:
        :return:
        """

        if len(args) == 2:
            fliter = args[0]
            rev_start_step = args[1]
        else:
            #print("未按照指定格式初始化正逆向滤波参数，默认设置滤波器为EKF，逆向滤波起始步长为600")
            fliter = 'ekf'
            rev_start_step = 600

        # 创建数组用于存储解算结果（估算状态Xest和协方差矩阵P）
        estimated_states = np.zeros((self.model.steps + 1, 4))
        estimated_covs = np.zeros((self.model.steps + 1, 4, 4))

        # 正向滤波到rev_start_step
        part_frkf_positions = self.model.sensor_states[:(rev_start_step + 1)]
        part_frkf_measurements = self.model.measurements[:(rev_start_step + 1)]

        if fliter == 'ekf':
            part_forward_fliter = BearingOnlyEKF(x0, P0, Q, self.model.R, self.model.sample_time,
                                              part_frkf_positions, part_frkf_measurements)

        elif fliter == 'plkf':
            part_forward_fliter = BearingOnlyPLKF(x0, P0, Q, self.model.R, self.model.sample_time,
                                              part_frkf_positions, part_frkf_measurements)

        elif fliter == 'ckf':
            part_forward_fliter = BearingOnlyCKF(x0, P0, Q, self.model.R, self.model.sample_time,
                                              part_frkf_positions, part_frkf_measurements)
        elif fliter == 'ukf':
            part_forward_fliter = BearingOnlyUKF(x0, P0, Q, self.model.R, self.model.sample_time,
                                              part_frkf_positions, part_frkf_measurements)
        else:
            raise ValueError(f'不支持的滤波器类型，当前选择滤波器为{fliter}')

        # 进行正向滤波迭代
        for i in range(1, rev_start_step + 1):
            part_forward_fliter.step()

        # 获取逆向滤波初值
        reverse_kf_init_state = part_forward_fliter.x
        reverse_kf_init_cov = part_forward_fliter.P

        # 初始化逆向滤波
        if fliter == 'ekf':
            part_reverse_kf = BearingOnlyEKF(reverse_kf_init_state, reverse_kf_init_cov, Q, self.model.R,
                                             self.model.sample_time,
                                             part_frkf_positions, part_frkf_measurements, backward=True)
        elif fliter == 'plkf':
            part_reverse_kf = BearingOnlyPLKF(reverse_kf_init_state, reverse_kf_init_cov, Q, self.model.R,
                                             self.model.sample_time,
                                             part_frkf_positions, part_frkf_measurements, backward=True)
        elif fliter == 'ukf':
            part_reverse_kf = BearingOnlyUKF(reverse_kf_init_state, reverse_kf_init_cov, Q, self.model.R,
                                             self.model.sample_time,
                                             part_frkf_positions, part_frkf_measurements, backward=True)
        elif fliter == 'ckf':
            part_reverse_kf = BearingOnlyCKF(reverse_kf_init_state, reverse_kf_init_cov, Q, self.model.R,
                                             self.model.sample_time,
                                             part_frkf_positions, part_frkf_measurements, backward=True)
        else:
            raise ValueError(f'不支持的滤波器类型，当前选择滤波器为{fliter}')

        # 进行逆向滤波迭代
        for i in range(1, rev_start_step + 1):
            part_reverse_kf.step()

        # 获取逆向滤波终值作为优化初值
        optimized_initial_state = part_reverse_kf.x
        optimized_initial_cov = part_reverse_kf.P

        estimated_states[0] = optimized_initial_state
        estimated_covs[0] = optimized_initial_cov

        # 基于优化初值进行全局正向滤波
        if fliter == 'ekf':
            forward_kf = BearingOnlyEKF(optimized_initial_state, optimized_initial_cov, Q, self.model.R,
                                         self.model.sample_time,
                                         self.model.sensor_states, self.model.measurements)

        elif fliter == 'plkf':
            forward_kf = BearingOnlyPLKF(optimized_initial_state, optimized_initial_cov, Q, self.model.R,
                                         self.model.sample_time,
                                         self.model.sensor_states, self.model.measurements)

        elif fliter == 'ukf':
            forward_kf = BearingOnlyUKF(optimized_initial_state, optimized_initial_cov, Q, self.model.R,
                                         self.model.sample_time,
                                         self.model.sensor_states, self.model.measurements)
        elif fliter == 'ckf':
            forward_kf = BearingOnlyCKF(optimized_initial_state, optimized_initial_cov, Q, self.model.R,
                                         self.model.sample_time,
                                         self.model.sensor_states, self.model.measurements)
        else:
            raise ValueError(f'不支持的滤波器类型，当前选择滤波器为{fliter}')

        # 存储结果，这里最好不要用 i
        for k in range(1, self.model.steps + 1):
            forward_kf.step()
            estimated_states[k] = forward_kf.x
            estimated_covs[k] = forward_kf.P

        state_err = (self.model.target_states - estimated_states) ** 2

        return {'color': 'green',
                'states': estimated_states,
                'covs': estimated_covs,
                'square_error': state_err,
                }

    def lsfrkf(self, x0, P0, Q, *args):
        """
        长短正逆向容积卡尔曼滤波

        :param x0:
        :param P0:
        :param Q:

        :return:
        """

        if len(args) == 3:
            fliter = args[0]
            rev_start_step = args[1]
            short_rev_step_length = args[2]
        else:
            #print("未按照指定格式初始化正逆向滤波参数，默认设置滤波器为EKF，长正逆向滤波起始步长为600，短正逆向滤波步长为20")
            fliter = 'ekf'
            rev_start_step = 600
            short_rev_step_length = 20

        estimated_states = np.zeros((self.model.steps + 1, 4))
        estimated_covs = np.zeros((self.model.steps + 1, 4, 4))

        """
           进行一次到k = rev_start_step轮次的长正逆向滤波
           (1)>正向滤波到 k 环节
           (2)>以(1)的x[k]和P[k]开始，逆向滤波short_rev_step_length次
           (3)>基于(2)中逆向滤波的结果，再正向滤波回到k+1环节，(3)的结果即为后面回合的最终估算结果
           """

        # 正向滤波到rev_start_step环节

        long_frkf_positions = self.model.sensor_states[:(rev_start_step + 1)]
        long_frkf_measurements = self.model.measurements[:(rev_start_step + 1)]

        if fliter == 'ekf':
            long_forward_kf = BearingOnlyEKF(x0, P0, Q, self.model.R, self.model.sample_time,
                                             long_frkf_positions, long_frkf_measurements)

        elif fliter == 'plkf':
            long_forward_kf = BearingOnlyPLKF(x0, P0, Q, self.model.R, self.model.sample_time,
                                             long_frkf_positions, long_frkf_measurements)

        elif fliter == 'ukf':
            long_forward_kf = BearingOnlyUKF(x0, P0, Q, self.model.R, self.model.sample_time,
                                             long_frkf_positions, long_frkf_measurements)
        elif fliter == 'ckf':
            long_forward_kf = BearingOnlyCKF(x0, P0, Q, self.model.R, self.model.sample_time,
                                             long_frkf_positions, long_frkf_measurements)
        else:
            raise ValueError(f'不支持的滤波器类型，当前选择滤波器为{fliter}')

        for i in range(1, rev_start_step + 1):
            long_forward_kf.step()

        # 从该环节开始逆向滤波优化初值
        long_reverse_ckf_init_state = long_forward_kf.x
        long_forward_ckf_init_covs = long_forward_kf.P

        if fliter == 'ekf':
            long_reverse_kf = BearingOnlyEKF(long_reverse_ckf_init_state, long_forward_ckf_init_covs, Q, self.model.R,
                                             self.model.sample_time,
                                             long_frkf_positions, long_frkf_measurements, backward=True)
        elif fliter == 'plkf':
            long_reverse_kf = BearingOnlyPLKF(long_reverse_ckf_init_state, long_forward_ckf_init_covs, Q, self.model.R,
                                             self.model.sample_time,
                                             long_frkf_positions, long_frkf_measurements, backward=True)

        elif fliter == 'ukf':
            long_reverse_kf = BearingOnlyUKF(long_reverse_ckf_init_state, long_forward_ckf_init_covs, Q, self.model.R,
                                             self.model.sample_time,
                                             long_frkf_positions, long_frkf_measurements, backward=True)
        elif fliter == 'ckf':
            long_reverse_kf = BearingOnlyCKF(long_reverse_ckf_init_state, long_forward_ckf_init_covs, Q, self.model.R,
                                             self.model.sample_time,
                                             long_frkf_positions, long_frkf_measurements, backward=True)
        else:
            raise ValueError(f'不支持的滤波器类型，当前选择滤波器为{fliter}')

        for i in range(1, rev_start_step + 1):
            long_reverse_kf.step()

        optimized_initial_state = long_reverse_kf.x
        optimized_initial_covariance = long_reverse_kf.P

        estimated_states[0] = optimized_initial_state
        estimated_covs[0] = optimized_initial_covariance

        # 再正向滤波回到rev_start_step环节
        if fliter == 'ekf':
            long_forward_kf_again = BearingOnlyEKF(optimized_initial_state, optimized_initial_covariance, Q,
                                                   self.model.R,
                                                   self.model.sample_time, long_frkf_positions, long_frkf_measurements)
        elif fliter == 'plkf':
            long_forward_kf_again = BearingOnlyPLKF(optimized_initial_state, optimized_initial_covariance, Q,
                                                   self.model.R,
                                                   self.model.sample_time, long_frkf_positions, long_frkf_measurements)

        elif fliter == 'ukf':
            long_forward_kf_again = BearingOnlyUKF(optimized_initial_state, optimized_initial_covariance, Q,
                                                   self.model.R,
                                                   self.model.sample_time, long_frkf_positions, long_frkf_measurements)
        elif fliter == 'ckf':
            long_forward_kf_again = BearingOnlyCKF(optimized_initial_state, optimized_initial_covariance, Q,
                                                   self.model.R,
                                                   self.model.sample_time, long_frkf_positions, long_frkf_measurements)
        else:
            raise ValueError(f'不支持的滤波器类型，当前选择滤波器为{fliter}')

        for ii in range(1, rev_start_step + 1):
            long_forward_kf_again.step()
            estimated_states[ii] = long_forward_kf_again.x
            estimated_covs[ii] = long_forward_kf_again.P

        """接下来的回合进行短正逆向滤波，重复以下操作：
           (1).从最后一个估计k开始，正向滤波到下一回合k+1
           (2).以(1)的x[k+1]和P[k+1]开始，向前逆向滤波short_rev_step_length次
           (3).基于(2)中逆向滤波的结果，再正向滤波回到k+1环节，(3)的结果即为后面回合的最终估算结果
           """

        for j in range(rev_start_step+1, self.model.steps+1):

            # 正向滤波到下一回合
            one_step_ckf_init_state = estimated_states[j-1]
            one_step_ckf_init_cov = estimated_covs[j-1]

            one_step_position = self.model.sensor_states[j-1: j+1]   # 获取第j个元素
            one_step_measurement = self.model.measurements[j-1: j+1]

            if fliter == 'ekf':
                one_step_kf = BearingOnlyEKF(one_step_ckf_init_state, one_step_ckf_init_cov, Q, self.model.R,
                                             self.model.sample_time, one_step_position, one_step_measurement)

            elif fliter == 'plkf':
                one_step_kf = BearingOnlyPLKF(one_step_ckf_init_state, one_step_ckf_init_cov, Q, self.model.R,
                                              self.model.sample_time, one_step_position, one_step_measurement)
            elif fliter == 'ukf':
                one_step_kf = BearingOnlyUKF(one_step_ckf_init_state, one_step_ckf_init_cov, Q, self.model.R,
                                             self.model.sample_time, one_step_position, one_step_measurement)
            elif fliter == 'ckf':
                one_step_kf = BearingOnlyCKF(one_step_ckf_init_state, one_step_ckf_init_cov, Q, self.model.R,
                                             self.model.sample_time, one_step_position, one_step_measurement)
            else:
                raise ValueError(f'不支持的滤波器类型，当前选择滤波器为{fliter}')

            one_step_kf.step()

            # 向前逆向滤波short_rev_step_length个回合
            short_reverse_init_state = one_step_kf.x
            short_reverse_init_cov = one_step_kf.P

            short_frckf_position = self.model.sensor_states[j - short_rev_step_length:j+1]
            short_frckf_measurement = self.model.measurements[j - short_rev_step_length:j+1]

            if fliter == 'ekf':
                short_reverse_kf = BearingOnlyEKF(short_reverse_init_state, short_reverse_init_cov, Q, self.model.R,
                                                  self.model.sample_time, short_frckf_position, short_frckf_measurement,
                                                  backward=True)
            elif fliter == 'plkf':
                short_reverse_kf = BearingOnlyPLKF(short_reverse_init_state, short_reverse_init_cov, Q, self.model.R,
                                                  self.model.sample_time, short_frckf_position, short_frckf_measurement,
                                                  backward=True)
            elif fliter == 'ukf':
                short_reverse_kf = BearingOnlyUKF(short_reverse_init_state, short_reverse_init_cov, Q, self.model.R,
                                                  self.model.sample_time, short_frckf_position, short_frckf_measurement,
                                                  backward=True)
            elif fliter == 'ckf':
                short_reverse_kf = BearingOnlyCKF(short_reverse_init_state, short_reverse_init_cov, Q, self.model.R,
                                                  self.model.sample_time, short_frckf_position, short_frckf_measurement,
                                                  backward=True)
            else:
                raise ValueError(f'不支持的滤波器类型，当前选择滤波器为{fliter}')

            for k in range(1, short_rev_step_length + 1):
                short_reverse_kf.step()

            # 再正向滤波回到k+1时刻
            short_forward_init_state = short_reverse_kf.x
            short_forward_init_cov = short_reverse_kf.P

            if fliter == 'ekf':
                short_forward_kf = BearingOnlyEKF(short_forward_init_state, short_forward_init_cov, Q, self.model.R,
                                                  self.model.sample_time, short_frckf_position, short_frckf_measurement)
            elif fliter == 'plkf':
                short_forward_kf = BearingOnlyPLKF(short_forward_init_state, short_forward_init_cov, Q, self.model.R,
                                                  self.model.sample_time, short_frckf_position, short_frckf_measurement)
            elif fliter == 'ukf':
                short_forward_kf = BearingOnlyUKF(short_forward_init_state, short_forward_init_cov, Q, self.model.R,
                                                  self.model.sample_time, short_frckf_position, short_frckf_measurement)
            elif fliter == 'ckf':
                short_forward_kf = BearingOnlyCKF(short_forward_init_state, short_forward_init_cov, Q, self.model.R,
                                                  self.model.sample_time, short_frckf_position, short_frckf_measurement)
            else:
                raise ValueError(f'不支持的滤波器类型，当前选择滤波器为{fliter}')

            for k in range(1, short_rev_step_length + 1):
                short_forward_kf.step()

            estimated_states[j] = short_forward_kf.x
            estimated_covs[j] = short_forward_kf.P

        state_err = (self.model.target_states - estimated_states) ** 2

        return {'color': 'purple',
                'states': estimated_states,
                'covs': estimated_covs,
                'square_error': state_err,
                }

    def mle(self, x0, *args):

        estimated_states = np.zeros((self.model.steps + 1, 4))
        estimated_states[0] = x0

        start_step = 100

        from copy import deepcopy
        for i in range(start_step, self.model.steps + 1):
            self.model_t = deepcopy(self.model)

            self.model_t.times = self.model.times[:i]
            self.model_t.measurements = self.model.measurements[:i, 0]
            self.model_t.sensor_states = self.model.sensor_states[:i]

            estimated_states[i] = lev_mar(self.model_t.times,
                                          self.model_t.sensor_states,
                                          self.model_t.measurements,
                                          x0)

        state_err = (self.model.target_states - estimated_states) ** 2

        return {'color': 'grey',
                'states': estimated_states,
                'covs': None,
                'square_error': state_err,
                }

    def lstsq1(self):

        estimated_states = np.zeros((self.model.steps + 1, 4))

        start_step = 100

        from copy import deepcopy
        for i in range(start_step, self.model.steps + 1):
            self.model_t = deepcopy(self.model)

            self.model_t.times = self.model.times[:i]
            self.model_t.measurements = self.model.measurements[:i, 0]
            self.model_t.sensor_states = self.model.sensor_states[:i]

            estimated_states[i] = lstsq1(self.model_t.times,
                                          self.model_t.sensor_states,
                                          self.model_t.measurements,
                                          )

        state_err = (self.model.target_states - estimated_states) ** 2

        return {'color': 'grey',
                'states': estimated_states,
                'covs': None,
                'square_error': state_err,
                }

    def lstsq(self):

        estimated_states = np.zeros((self.model.steps + 1, 4))

        start_step = 100

        from copy import deepcopy
        for i in range(start_step, self.model.steps + 1):
            self.model_t = deepcopy(self.model)

            self.model_t.times = self.model.times[:i]
            self.model_t.measurements = self.model.measurements[:i, 0]
            self.model_t.sensor_states = self.model.sensor_states[:i]

            estimated_states[i] = lstsq(self.model_t.times,
                                          self.model_t.sensor_states,
                                          self.model_t.measurements,
                                          )

        state_err = (self.model.target_states - estimated_states) ** 2

        return {'color': 'grey',
                'states': estimated_states,
                'covs': None,
                'square_error': state_err,
                }