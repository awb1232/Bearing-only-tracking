from model import *
from kalman_fliters import *

class Algorithms:

    def __init__(self, model):
        self.model = model

    def run_ekf(self, x0, P0, Q):

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

    def run_plkf(self, x0, P0, Q):

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

    def run_ukf(self, x0, P0, Q):

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

    def run_ckf(self, x0, P0, Q):

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
        estimated_states = np.zeros((self.model.steps + 1, 4))
        estimated_covs = np.zeros((self.model.steps + 1, 4, 4))

        # 正向滤波到rev_start_step
        part_frckf_positions = self.model.sensor_states[:(rev_start_step + 1)]
        part_frckf_measurements = self.model.measurements[:(rev_start_step + 1)]

        part_forward_ckf = BearingOnlyUKF(x0, P0, Q, self.model.R, self.model.sample_time,
                                          part_frckf_positions, part_frckf_measurements)

        # 进行正向滤波迭代
        for i in range(1, rev_start_step + 1):
            part_forward_ckf.step()

        # 获取逆向滤波初值
        reverse_ckf_init_state = part_forward_ckf.x
        reverse_ckf_init_cov = part_forward_ckf.P

        # 初始化逆向滤波
        part_reverse_ckf = BearingOnlyUKF(reverse_ckf_init_state, reverse_ckf_init_cov, Q, self.model.R, self.model.sample_time,
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
        forward_ckf = BearingOnlyUKF(optimized_initial_state, optimized_initial_cov, Q, self.model.R, self.model.sample_time,
                                          self.model.sensor_states, self.model.measurements)

        # 存储结果，这里最好不要用 i
        for k in range(1, self.model.steps + 1):
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

        estimated_states = np.zeros((self.model.steps + 1, 4))
        estimated_covs = np.zeros((self.model.steps + 1, 4, 4))

        """
           进行一次到k = rev_start_step轮次的长正逆向滤波
           (1)>正向滤波到 k 环节
           (2)>以(1)的x[k]和P[k]开始，逆向滤波short_rev_step_length次
           (3)>基于(2)中逆向滤波的结果，再正向滤波回到k+1环节，(3)的结果即为后面回合的最终估算结果
           """

        # 正向滤波到rev_start_step环节

        long_frckf_positions = self.model.sensor_states[:(rev_start_step + 1)]
        long_frckf_measurements = self.model.measurements[:(rev_start_step + 1)]

        long_forward_ckf = BearingOnlyUKF(x0, P0, Q, self.model.R, self.model.sample_time,
                                       long_frckf_positions, long_frckf_measurements)


        for i in range(1, rev_start_step + 1):
            long_forward_ckf.step()

        # 从该环节开始逆向滤波优化初值
        long_reverse_ckf_init_state = long_forward_ckf.x
        long_forward_ckf_init_covs = long_forward_ckf.P

        long_reverse_ckf = BearingOnlyUKF(long_reverse_ckf_init_state, long_forward_ckf_init_covs, Q, self.model.R, self.model.sample_time,
                                       long_frckf_positions, long_frckf_measurements, backward=True)

        for i in range(1, rev_start_step + 1):
            long_reverse_ckf.step()

        optimized_initial_state = long_reverse_ckf.x
        optimized_initial_covariance = long_reverse_ckf.P

        estimated_states[0] = optimized_initial_state
        estimated_covs[0] = optimized_initial_covariance

        # 再正向滤波回到rev_start_step环节
        long_forward_ckf_again = BearingOnlyUKF(optimized_initial_state, optimized_initial_covariance, Q, self.model.R,
                                        self.model.sample_time, long_frckf_positions, long_frckf_measurements)

        for ii in range(1, rev_start_step + 1):
            long_forward_ckf_again.step()
            estimated_states[ii] = long_forward_ckf_again.x
            estimated_covs[ii] = long_forward_ckf_again.P

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

            one_step_ckf = BearingOnlyUKF(one_step_ckf_init_state, one_step_ckf_init_cov, Q, self.model.R,
                                        self.model.sample_time, one_step_position, one_step_measurement)

            one_step_ckf.step()

            # 向前逆向滤波short_rev_step_length个回合
            short_reverse_init_state = one_step_ckf.x
            short_reverse_init_cov = one_step_ckf.P

            short_frckf_position = self.model.sensor_states[j - short_rev_step_length:j+1]
            short_frckf_measurement = self.model.measurements[j - short_rev_step_length:j+1]

            short_reverse_ckf = BearingOnlyUKF(short_reverse_init_state, short_reverse_init_cov, Q, self.model.R,
                                               self.model.sample_time, short_frckf_position, short_frckf_measurement,
                                               backward=True)

            for k in range(1, short_rev_step_length + 1):
                short_reverse_ckf.step()

            # 再正向滤波回到k+1时刻
            short_forward_init_state = short_reverse_ckf.x
            short_forward_init_cov = short_reverse_ckf.P

            short_forward_ckf = BearingOnlyUKF(short_forward_init_state, short_forward_init_cov, Q, self.model.R,
                                        self.model.sample_time, short_frckf_position, short_frckf_measurement)

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