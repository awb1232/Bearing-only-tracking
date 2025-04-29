"""增加了动画"""
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
        self.sensor_trajectory = np.zeros((self.steps + 1, 2))  # 传感器轨迹

        # 目标真实状态
        self.target_states = np.zeros((self.steps + 1, 4))
        self.target_states[0] = self.target_init_state

        # 方位角
        self.bearings = np.zeros((self.steps + 1, 1))
        self.measurements = np.zeros((self.steps + 1, 1))

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

    def generate_bearings(self):

        for k in range(self.steps + 1):
            observer_pos = self.sensor_trajectory[k]
            dx = self.target_states[k, 0] - observer_pos[0]
            dy = self.target_states[k, 1] - observer_pos[1]
            true_bearing = np.arctan2(dy, dx)
            self.bearings[k, 0] = true_bearing
            self.measurements[k, 0] = true_bearing + np.sqrt(self.R) * np.random.randn()


class BearingOnlyCKF:
    """
    使用立方卡尔曼滤波器(CKF)进行纯方位目标运动分析
    """

    def __init__(self, x0, P0, Q, R, dt, observer_trajectory, backward=False):
        """
        初始化CKF参数

        参数:
        x0: 初始状态向量 [x, y, vx, vy]
        P0: 初始协方差矩阵
        Q: 过程噪声协方差矩阵
        R: 测量噪声协方差 (标量)
        dt: 时间步长
        observer_trajectory: 观测者轨迹，每行为一个时间步的位置 [x, y]
        """
        self.n = len(x0)  # 状态维度
        self.x = x0.copy()  # 状态向量
        self.P = P0.copy()  # 协方差矩阵
        self.Q = Q.copy()  # 过程噪声协方差矩阵
        self.R = R  # 测量噪声协方差 (标量)
        self.dt = dt  # 时间步长
        self.observer_trajectory = observer_trajectory  # 观测者轨迹
        self.current_step = 0  # 当前步

        # CKF参数
        self.num_points = 2 * self.n  # CKF使用2n个立方点
        self.weight = 1.0 / (2 * self.n)  # 所有点的权重相等

        # 保存诊断信息
        self.bearing_pred_error = []  # 保存方位角预测误差
        self.innovation_covariance = []  # 保存P阵

        self.nees = []  # 归一化估计误差平方
        self.nis = []  # 归一化创新平方

        self.backward = backward

    def generate_cubature_points(self):
        """生成立方点"""
        # 计算矩阵平方根
        try:
            # 确保协方差矩阵是对称的
            self.P = (self.P + self.P.T) / 2

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

    def state_transition(self, x, add_noise=True):
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
        bearing = np.arctan2(dy, dx)

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

    def update(self, z, propagated_points):
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
        z_residual = z - z_pred
        z_residual[0] = self.normalize_angle(z_residual[0])

        # 保存创新和创新协方差用于诊断
        self.bearing_pred_error.append(z_residual[0])

        self.innovation_covariance.append(P_zz)

        # 计算归一化创新平方 (NIS)
        nis = z_residual[0] ** 2 / P_zz
        self.nis.append(nis)

        # 更新状态和协方差
        self.x += K * z_residual[0]
        self.P -= np.outer(K, K) * P_zz

        # 确保协方差矩阵保持对称
        self.P = (self.P + self.P.T) / 2

    def step(self, z, true_state=None):
        """执行完整的CKF步骤：预测和更新"""
        # 预测步骤
        propagated_points = self.predict()

        # 更新步骤
        self.update(z, propagated_points)

        # 如果提供了真实状态，计算NEES
        if true_state is not None:
            error = self.x - true_state
            nees = error @ np.linalg.inv(self.P) @ error.T

            self.nees.append(nees)

        # 更新当前步
        if self.backward:
            self.current_step -= 1
        else:
            self.current_step += 1

        return self.x, self.P


class Algorithms:

    def __init__(self, model):
        self.model = model

    def run_ckf(self, x0, P0, Q):

        ckf = BearingOnlyCKF(x0, P0, Q, model.R, model.sample_time, model.sensor_trajectory)

        # 运行滤波
        estimated_states = np.zeros((model.steps + 1, 4))
        estimated_covs = np.zeros((model.steps + 1, 4, 4))

        estimated_states[0] = x0
        estimated_covs[0] = P0

        for k in range(1, model.steps + 1):
            ckf.step(model.measurements[k], model.target_states[k])
            estimated_states[k] = ckf.x
            estimated_covs[k] = ckf.P

        # 计算性能指标
        pos_rmse = np.sqrt(np.sum((model.target_states[:, 0:2] - estimated_states[:, 0:2]) ** 2, axis=1))
        vel_rmse = np.sqrt(np.sum((model.target_states[:, 2:4] - estimated_states[:, 2:4]) ** 2, axis=1))

        return {'color':'blue',
                'states': estimated_states,
                'covs': estimated_covs,
                'pos_rmse': pos_rmse,
                'vel_rmse': vel_rmse, }

    def run_frckf(self, x0, P0, Q, backward_start_step):

        ckf = BearingOnlyCKF(x0, P0, Q, model.R, model.sample_time, model.sensor_trajectory)

        # 运行滤波
        estimated_states = np.zeros((model.steps + 1, 4))
        estimated_covs = np.zeros((model.steps + 1, 4, 4))

        estimated_states[0] = x0
        estimated_covs[0] = P0

        for k in range(1, model.steps + 1):
            ckf.step(model.measurements[k], model.target_states[k])
            estimated_states[k] = ckf.x
            estimated_covs[k] = ckf.P

        ###——————————————————————————————————————————运行逆向滤波获取优化初值——————————————————————————————————————————###

        optimized_initial_state = None
        optimized_initial_covariance = None

        if backward_start_step > 0 and backward_start_step <= model.steps:
            # print("\n=== 开始逆向滤波优化初值 ===")
            # print(f"从第 {backward_start_step} 步开始逆向滤波")

            # 使用前向滤波在指定步骤的估计作为逆向滤波的起点
            backward_x0 = estimated_states[backward_start_step].copy()
            backward_P0 = estimated_covs[backward_start_step].copy()

            # 创建逆向轨迹 (从backward_start_step到0)
            backward_trajectory = model.sensor_trajectory[:(backward_start_step + 1)][::-1]

            # 创建逆向测量
            backward_measurements = model.measurements[:(backward_start_step + 1)][::-1]

            # 实例化逆向CKF
            backward_ckf = BearingOnlyCKF(
                backward_x0, backward_P0, Q, model.R, model.sample_time, backward_trajectory, backward=True
            )

            # 运行逆向滤波
            backward_states = np.zeros((backward_start_step + 1, 4))
            backward_covs = np.zeros((backward_start_step + 1, 4, 4))

            backward_states[0] = backward_x0
            backward_covs[0] = backward_P0

            # 逆向滤波从索引1开始（对应原始序列的backward_start_step-1）
            for k in range(1, backward_start_step + 1):
                backward_ckf.step(backward_measurements[k])
                backward_states[k] = backward_ckf.x
                backward_covs[k] = backward_ckf.P

            # 逆向滤波优化后的初值是最后一步的状态
            optimized_initial_state = backward_states[-1]
            optimized_initial_covariance = backward_covs[-1]

        ###——————————————————————————————————————————基于优化的初值再进行正向滤波——————————————————————————————————————————###

        if optimized_initial_state is not None:
            # print("\n=== 使用优化初值重新运行前向滤波 ===")

            # 实例化新的前向CKF，使用优化的初值
            optimized_forward_ckf = BearingOnlyCKF(
                optimized_initial_state,
                optimized_initial_covariance,
                Q, model.R, model.sample_time, model.sensor_trajectory,
            )

            # 运行优化初值的前向滤波
            optimized_forward_states = np.zeros((model.steps + 1, 4))
            optimized_forward_covs = np.zeros((model.steps + 1, 4, 4))

            optimized_forward_states[0] = optimized_initial_state
            optimized_forward_covs[0] = optimized_initial_covariance

            for k in range(1, model.steps + 1):
                optimized_forward_ckf.step(model.measurements[k], model.target_states[k])
                optimized_forward_states[k] = optimized_forward_ckf.x
                optimized_forward_covs[k] = optimized_forward_ckf.P
        else:
            # 如果没有运行逆向滤波，则优化后的状态就是原始状态
            optimized_forward_states = estimated_states
            optimized_forward_covs = estimated_covs

        pos_rmse = np.sqrt(np.sum((model.target_states[:, 0:2] - optimized_forward_states[:, 0:2]) ** 2, axis=1))
        vel_rmse = np.sqrt(np.sum((model.target_states[:, 2:4] - optimized_forward_states[:, 2:4]) ** 2, axis=1))

        return {'color':'green',
                'states': optimized_forward_states,

                'covs': optimized_forward_covs,
                'pos_rmse': pos_rmse,
                'vel_rmse': vel_rmse, }

    def run_frffckf(self, x0, P0, Q, k, n):

        ###----------------------------------正向滤波到backward_start_step回合-----------------------------------------###

        forward_ckf = BearingOnlyCKF(x0, P0, Q, model.R, model.sample_time, model.sensor_trajectory)

        forward_states = np.zeros((model.steps + 1, 4))
        forward_covs = np.zeros((model.steps + 1, 4, 4))

        forward_states[0] = x0
        forward_covs[0] = P0

        for i in range(1, k + 1):
            forward_ckf.step(model.measurements[i], model.target_states[i])
            forward_states[i] = forward_ckf.x
            forward_covs[i] = forward_ckf.P

        ###—————————————————————————————————————-—————运行逆向滤波获取优化初值—————————-—————————————————————————————————###
        backward_start_state = forward_states[k]
        backward_start_cov = forward_covs[k]
        backward_sensor_trajectory = model.sensor_trajectory[:k + 1][::-1]
        backward_measurements = model.measurements[:(k + 1)][::-1]
        backward_ckf = BearingOnlyCKF(backward_start_state, backward_start_cov, Q, model.R, model.sample_time, backward_sensor_trajectory, backward=True)

        backward_states = np.zeros((k + 1, 4))
        backward_covs = np.zeros((k + 1, 4, 4))
        backward_states[0] = backward_start_state
        backward_covs[0] = backward_start_cov

        for i in range(1, k + 1):
            backward_ckf.step(backward_measurements[i], model.target_states[i])
            backward_states[i] = backward_ckf.x
            backward_covs[i] = backward_ckf.P

        optimized_initial_state = backward_states[-1]
        optimized_initial_covariance = backward_covs[-1]

        ###---------------------------------------------正向滤波回到k回合-------------------------------------------------###
        forward_ckf = BearingOnlyCKF(optimized_initial_state, optimized_initial_covariance, Q, model.R, model.sample_time, model.sensor_trajectory)

        forward_states = np.zeros((model.steps + 1, 4))
        forward_covs = np.zeros((model.steps + 1, 4, 4))
        forward_states[0] = optimized_initial_state
        forward_covs[0] = optimized_initial_covariance

        for i in range(1, k + 1):
            forward_ckf.step(model.measurements[i], model.target_states[i])
            forward_states[i] = forward_ckf.x
            forward_covs[i] = forward_ckf.P

        # 4. 局部正逆向滤波
        for i in range(k, model.steps, 1):
            # 正向滤波到k+1
            forward_ckf.step(model.measurements[i + 1], model.target_states[i + 1])

            # 逆向滤波到k+1-n
            local_backward_sensor_trajectory = model.sensor_trajectory[i + 1 - n:i + 2][::-1]
            local_backward_measurements = model.measurements[i + 1 - n:i + 2][::-1]
            local_backward_ckf = BearingOnlyCKF(
                forward_ckf.x, forward_ckf.P, Q, model.R, dt,
                local_backward_sensor_trajectory, backward=True
            )
            for step in range(n):
                #back_bearing = model.measurements[k + 1 - n + step]
                #fwd_bearing = model.measurements[(k + 1 - n + step)]
                #back_bearing1 = local_backward_measurements[step]
                #local_backward_ckf.step(model.measurements[k + 1 - step_size + step])
                local_backward_ckf.step(local_backward_measurements[step])

            # 再正向滤波到k+1
            local_forward_sensor_trajectory = model.sensor_trajectory[i + 1 - n:i + 2]
            local_forward_measurements = model.measurements[i + 1 - n:i + 2]
            local_forward_ckf = BearingOnlyCKF(
                local_backward_ckf.x, local_backward_ckf.P, Q, model.R, dt,
                local_forward_sensor_trajectory
            )

            for step in range(n):
                local_forward_ckf.step(local_forward_sensor_trajectory[step])

            forward_states[i+1] = local_forward_ckf.x
            forward_covs[i+1] = local_forward_ckf.P

        pos_rmse = np.sqrt(np.sum((model.target_states[:, 0:2] - forward_states[:, 0:2]) ** 2, axis=1))
        vel_rmse = np.sqrt(np.sum((model.target_states[:, 2:4] - forward_states[:, 2:4]) ** 2, axis=1))

        return {'color': 'purple',
                'states': forward_states,
                'covs': forward_covs,
                'pos_rmse': pos_rmse,
                'vel_rmse': vel_rmse, }

    def run_frffckf1(self, x0, P0, Q, k, n):

        # 存储最终估计的结果
        estimation_states = np.zeros((model.steps + 1, 4))
        estimation_covs = np.zeros((model.steps + 1, 4, 4))

        estimation_states[0] = x0
        estimation_covs[0] = P0

        """运行正向CKF到第k回合"""

        forward_ckf_k = BearingOnlyCKF(x0, P0, Q, model.R, model.sample_time, model.sensor_trajectory)

        for i in range(1, k + 1):
            forward_ckf_k.step(model.measurements[i], model.target_states[i])
            estimation_states[i] = forward_ckf_k.x
            estimation_covs[i] = forward_ckf_k.P

        """从k回合运行逆向ckf直到初始回合，优化初值"""

        # 使用前向滤波在指定步骤的估计作为逆向滤波的起点
        backward_x0 = estimation_states[k].copy()
        backward_P0 = estimation_covs[k].copy()

        # 创建逆向轨迹 (从backward_start_step到0)
        backward_trajectory = model.sensor_trajectory[:(k + 1)][::-1]

        # 创建逆向测量
        backward_measurements = model.measurements[:(k + 1)][::-1]

        # 实例化逆向CKF
        backward_ckf_k = BearingOnlyCKF(
            backward_x0, backward_P0, Q, model.R, model.sample_time, backward_trajectory, backward=True
        )

        # 运行逆向滤波
        backward_states = np.zeros((k + 1, 4))
        backward_covs = np.zeros((k + 1, 4, 4))

        backward_states[0] = backward_x0
        backward_covs[0] = backward_P0

        # 逆向滤波从索引1开始（对应原始序列的backward_start_step-1）
        for i in range(1, k + 1):
            backward_ckf_k.step(backward_measurements[i])
            backward_states[i] = backward_ckf_k.x
            backward_covs[i] = backward_ckf_k.P

        # 逆向滤波优化后的初值是最后一步的状态
        optimized_initial_state = backward_states[-1]
        optimized_initial_covariance = backward_covs[-1]

        """再正向优化回到k回合"""
        forward_ckf_k2 = BearingOnlyCKF(optimized_initial_state,
                                        optimized_initial_covariance
                                        , Q, model.R, model.sample_time,
                                        model.sensor_trajectory)

        for i in range(1, k + 1):
            forward_ckf_k2.step(model.measurements[i])
            estimation_states[i] = forward_ckf_k2.x
            estimation_covs[i] = forward_ckf_k2.P


        """接下来循环如下的操作：
           1.运行正向CKF到k+1回合
           2.以1中的k+1回合的x和P为基础，反向运行n次逆向卡尔曼滤波，进行局部的初值优化
           3.获得2中局部的优化初值后，再进行n次局部正向滤波，获得k+1回合的优化解，作为最终k+1回合的状态估计"""

        for i in range(k, model.steps, 1):
            # 运行正向CKF到k+1回合
            ckf_start = BearingOnlyCKF(estimation_states[i],
                                       estimation_covs[i],
                                       Q, model.R, model.sample_time,
                                       model.sensor_trajectory)

            ckf_start.step(model.measurements[i])

            backward_ckf_n_x0 = ckf_start.x
            backward_ckf_n_p0 = ckf_start.P

            # 反向运行n次逆向卡尔曼滤波，进行局部的初值优化

            # 获取k-n到k的所有轨迹和量测序列
            backward_sensor_trajectory_n = model.sensor_trajectory[i + 1 - n:i + 2][::-1]
            backward_measurements_n = model.measurements[i + 1 - n:i + 2][::-1]

            backward_ckf_n = BearingOnlyCKF(backward_ckf_n_x0,
                                            backward_ckf_n_p0,
                                            Q, model.R, model.sample_time,
                                            backward_sensor_trajectory_n)

            # 进行n次逆向卡尔曼滤波
            for step in range(n):
                backward_ckf_n.step(backward_measurements[step])

            ckf_n_x0 = backward_ckf_n.x
            ckf_n_p0 = backward_ckf_n.P

            # 再进行n次正向卡尔曼滤波
            forward_sensor_trajectory_n = model.sensor_trajectory[i + 1 - n:i + 2]
            forward_measurements_n = model.measurements[i + 1 - n:i + 2]

            forward_ckf_n = BearingOnlyCKF(ckf_n_x0,
                                           ckf_n_p0,
                                           Q, model.R, model.sample_time,
                                           forward_sensor_trajectory_n)

            # 进行n次正向卡尔曼滤波
            for step in range(n):
                forward_ckf_n.step(forward_measurements_n[step])

            # 存储结果
            estimation_states[i] = forward_ckf_n.x
            estimation_covs[i] = forward_ckf_n.P

        pos_rmse = np.sqrt(np.sum((model.target_states[:, 0:2] - estimation_states[:, 0:2]) ** 2, axis=1))
        vel_rmse = np.sqrt(np.sum((model.target_states[:, 2:4] - estimation_states[:, 2:4]) ** 2, axis=1))

        return {'color': 'purple',
            'states': estimation_states,
            'covs': estimation_covs,
            'pos_rmse': pos_rmse,
            'vel_rmse': vel_rmse, }

class Runner:

    def __init__(self, algorithms):
        self.model = algorithms.model
        self.method_name = None
        self.method_map = {
            "ckf": algorithms.run_ckf,
            "frckf": algorithms.run_frckf,
            "frffckf": algorithms.run_frffckf,
            "frffckf1": algorithms.run_frffckf1,
        }

        self.result = []

    def select_method(self, method):
        self.method_name = method
        # 检查方法是否有效
        if method not in self.method_map:
            # 提取所有支持的算法名称，用逗号分隔
            supported_methods = ", ".join(sorted(self.method_map.keys()))
            raise ValueError(
                f"Unknown method: '{method}'. Supported methods are: {supported_methods}"
            )

    def run_monte_carlo(self, num):

        # 获取对应的函数
        target_method = self.method_map[self.method_name]

        # 生成轨迹
        model.generate_sensor_trajectory_8()
        model.generate_target_trajectory()

        x0 = np.array([4000.0, 4000.0, 0, -0])
        # 初始状态不确定性
        P0 = np.diag([100.0 ** 2, 100.0 ** 2, 1.0 ** 2, 50.0 ** 2])
        # 过程噪声协方差 (滤波器中使用的估计值，可能与真实值不完全相同)
        Q = np.diag([0.1, 0.1, 0.01, 0.01])

        estimation_all = []
        #cov_all = []
        pos_rmse_all = []
        vel_rmse_all = []
        print(f'对{self.method_name}方法进行{num}次蒙特卡洛仿真')
        start_time = time.time()
        for i in tqdm(range(num),
                      desc="仿真进度",
                      unit="次",
                      bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{remaining}]"
                      ):
            # 生成方位
            model.generate_bearings()

            if self.method_name == 'frckf':
                reverse_step = 400
                result = target_method(x0, P0, Q, reverse_step)
            elif self.method_name == 'frffckf':
                reverse_step = 400
                partical_rev_step = 10
                result = target_method(x0, P0, Q, reverse_step, partical_rev_step)
            elif self.method_name == 'frffckf1':
                reverse_step = 400
                partical_rev_step = 10
                result = target_method(x0, P0, Q, reverse_step, partical_rev_step)
            else:
                result = target_method(x0, P0, Q)

            estimation_all.append(result['states'])
            #cov_all.append(result['covs'])
            pos_rmse_all.append(result['pos_rmse'])
            vel_rmse_all.append(result['vel_rmse'])

        time1 = time.time() - start_time
        print(f'仿真结束，用时{time1:.2f}')

        avg_estimation = np.mean(estimation_all, axis=0)
        avg_pos_rmse = np.mean(pos_rmse_all, axis=0)
        avg_vel_rmse = np.mean(vel_rmse_all, axis=0)

        runner_result = {'name': self.method_name,
                         'num': num,
                         'time': np.arange(0, self.model.steps * self.model.sample_time + self.model.sample_time, self.model.sample_time),
                         'sensor_traj': self.model.sensor_trajectory,
                         'true_state': self.model.target_states,
                         'color':result['color'],
                         'estimation': avg_estimation,
                         'pos_rmse': avg_pos_rmse,
                         'vel_rmse': avg_vel_rmse,}

        self.result.append(runner_result)

        return avg_pos_rmse, avg_vel_rmse,


class Visulation:

    def __init__(self, Runner):

        self.plot_result = Runner.result

    def animate(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)

        num_of_methods_used = len(self.plot_result)

        if num_of_methods_used < 1:
            raise ValueError('未使用任何方法进行仿真！')

        # 绘制完整的真实轨迹和观测者轨迹
        true_states = self.plot_result[0]['true_state']
        sensor_trajectory = self.plot_result[0]['sensor_traj']

        ax.plot_result(true_states[:, 0], true_states[:, 1], 'b-', alpha=0.3, label='目标轨迹')
        ax.plot_result(observer_trajectory[:, 0], observer_trajectory[:, 1], 'k-', alpha=0.5, label='传感器轨迹')

        # 初始化绘图元素
        target_true, = ax.plot_result([], [], 'bo', markersize=6)
        target_est, = ax.plot_result([], [], 'ro', markersize=6)
        observer, = ax.plot_result([], [], 'ko', markersize=6)
        estimation_line, = ax.plot_result([], [], 'r-', alpha=0.5)

        frckf_estimation_line, = ax.plot_result([], [], 'g-', alpha=0.5)

    def plot_figure(self):

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

        # 绘制RMSE
        # 绘制位置RMSE
        plt.figure()

        times_range = self.plot_result[0]['time']
        for i in range(num_of_methods_used):
            name = self.plot_result[i]['name']
            pos_rmse = self.plot_result[i]['pos_rmse']
            color = self.plot_result[i]['color']
            plt.plot(times_range, pos_rmse, color=color, label=f"{name}算法{num}次仿真平均位置误差")

        plt.xlabel('时间 (s)')
        plt.ylabel('位置误差 (m)')
        plt.title('位置估计误差')
        plt.legend()
        plt.show()

        # 绘制速度RMSE
        plt.figure()

        times_range = self.plot_result[0]['time']
        for i in range(num_of_methods_used):
            name = self.plot_result[i]['name']
            pos_rmse = self.plot_result[i]['vel_rmse']
            color = self.plot_result[i]['color']
            plt.plot(times_range, pos_rmse, color=color, label=f"{name}算法{num}次仿真平均速度误差")

        plt.xlabel('时间 (s)')
        plt.ylabel('速度误差 (m/s)')
        plt.title('速度估计误差')
        plt.legend()
        plt.show()



if __name__ == '__main__':
    target_init_state = np.array([5000, 5000, 0, -5])
    dt = 2
    maxt = 2000
    noise_mean = 0
    noise_std = 0.1
    model = Model(target_init_state, dt=dt, maxt=maxt, brg_noise_mean=noise_mean, brg_noise_std=noise_std)

    algorithms = Algorithms(model)

    number = 1
    Runner = Runner(algorithms)
    Runner.select_method('ckf')
    Runner.run_monte_carlo(number)

    Runner.select_method('frckf')
    Runner.run_monte_carlo(number)

    Runner.select_method('frffckf1')
    Runner.run_monte_carlo(number)

    Visulation = Visulation(Runner)
    Visulation.plot_figure()


