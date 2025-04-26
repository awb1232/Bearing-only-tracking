"""进行蒙特卡洛仿真"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.animation as animation
from scipy.stats import chi2
import time
from tqdm import tqdm

plt.rcParams['font.sans-serif'] = ['SimSun']  # 使用宋体字体
plt.rcParams['axes.unicode_minus'] = False  # 显示负号


class BearingOnlyUKF:
    """
    使用无迹卡尔曼滤波(UKF)进行纯方位目标运动分析
    """

    def __init__(self, x0, P0, Q, R, dt, observer_pos):
        """
        初始化UKF参数

        参数:
        x0: 初始状态向量 [x, y, vx, vy]
        P0: 初始协方差矩阵
        Q: 过程噪声协方差矩阵
        R: 测量噪声协方差 (标量)
        dt: 时间步长
        observer_pos: 观测者位置 [x, y]
        """
        self.n = len(x0)  # 状态维度
        self.x = x0.copy()  # 状态向量
        self.P = P0.copy()  # 协方差矩阵
        self.Q = Q.copy()  # 过程噪声协方差矩阵
        self.R = R  # 测量噪声协方差 (标量)
        self.dt = dt  # 时间步长
        self.observer_pos = observer_pos  # 观测者位置

        # UKF参数
        self.alpha = 1e-3  # 控制sigma点的散布程度
        self.beta = 2.0  # 先验分布的最优值 (2表示高斯分布)
        self.kappa = 0  # 次要参数，通常设为0

        # 计算缩放参数
        self.lambda_ = self.alpha ** 2 * (self.n + self.kappa) - self.n

        # 权重参数计算
        self.compute_weights()

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
        # 计算矩阵平方根
        L = self.n + self.lambda_
        sqrt_P = np.linalg.cholesky((L * self.P).astype(float))

        # 创建sigma点矩阵
        sigma_points = np.zeros((2 * self.n + 1, self.n))
        sigma_points[0] = self.x

        for i in range(self.n):
            sigma_points[i + 1] = self.x + sqrt_P[i]
            sigma_points[i + 1 + self.n] = self.x - sqrt_P[i]

        return sigma_points

    def state_transition(self, x):
        """
        状态转移函数 - 匀速直线运动模型
        x = [位置x, 位置y, 速度vx, 速度vy]
        """
        F = np.array([
            [1, 0, self.dt, 0],
            [0, 1, 0, self.dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        return F @ x

    def measurement_function(self, x):
        """
        测量函数 - 计算从观测者到目标的方位角
        返回以弧度表示的方位角
        """
        dx = x[0] - self.observer_pos[0]
        dy = x[1] - self.observer_pos[1]
        bearing = np.arctan2(dy, dx)

        return np.array([bearing])

    def normalize_angle(self, angle):
        """将角度归一化到[-pi, pi]范围内"""
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def predict(self):
        """UKF预测步骤"""
        # 生成sigma点
        sigma_points = self.generate_sigma_points()

        # 传播sigma点
        sigma_points_pred = np.array([self.state_transition(sigma) for sigma in sigma_points])

        # 计算预测状态
        self.x = np.sum(self.weights_m.reshape(-1, 1) * sigma_points_pred, axis=0)

        # 计算预测协方差
        self.P = np.zeros((self.n, self.n))
        for i in range(len(sigma_points_pred)):
            diff = sigma_points_pred[i] - self.x
            self.P += self.weights_c[i] * np.outer(diff, diff)

        # 添加过程噪声协方差
        self.P += self.Q

        return sigma_points_pred

    def update(self, z, sigma_points_pred):
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
        z_residual = z - z_mean
        z_residual[0] = self.normalize_angle(z_residual[0])

        # 更新状态和协方差
        self.x += K * z_residual[0]
        self.P -= np.outer(K, K) * P_zz

    def step(self, z):
        """执行完整的UKF步骤：预测和更新"""
        sigma_points_pred = self.predict()
        self.update(z, sigma_points_pred)
        return self.x, self.P
    

class BearingOnlyCKF:
    """
    使用立方卡尔曼滤波器(CKF)进行纯方位目标运动分析
    """

    def __init__(self, x0, P0, Q, R, dt, observer_trajectory):
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
        self.innovation = []  # 保存创新序列
        self.innovation_covariance = []  # 保存创新协方差
        self.nees = []  # 归一化估计误差平方
        self.nis = []  # 归一化创新平方

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
        self.current_step += 1

        return self.x, self.P


# 单次仿真函数
def single_simulation(seed=None):
    """执行单次仿真，返回结果而不绘图"""
    if seed is not None:
        np.random.seed(seed)

    # 设置仿真参数
    dt = 1.0  # 时间步长 (秒)
    simulation_time = 2000  # 仿真时长
    steps = int(simulation_time / dt)  # 仿真步数

    # 观测者轨迹 (弯曲轨迹以提高可观测性)
    observer_trajectory = np.zeros((steps + 1, 2))
    for k in range(steps + 1):
        t = k * dt
        observer_trajectory[k, 0] = 100 * np.sin(t / 20)
        observer_trajectory[k, 1] = 50 * np.sin(t / 10)

    # 真实目标初始状态和运动
    true_x0 = np.array([5000.0, 5000.0, 0, -10.0])  # [x, y, vx, vy]
    Q_true = np.diag([0.01, 0.01, 0.005, 0.005])  # 真实过程噪声

    # 初始化真实轨迹
    true_states = np.zeros((steps + 1, 4))
    true_states[0] = true_x0

    # 生成真实轨迹 (包括过程噪声)
    F = np.array([
        [1, 0, dt, 0],
        [0, 1, 0, dt],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    for k in range(1, steps + 1):
        process_noise = np.random.multivariate_normal(np.zeros(4), Q_true)
        true_states[k] = F @ true_states[k - 1]# + process_noise

    # 生成测量值 (仅方位角)
    measurements = np.zeros((steps + 1, 1))
    bearing_noise_std = 0.1 # degree
    R = np.deg2rad((bearing_noise_std)**2)  # 测量噪声方差 (弧度)

    for k in range(steps + 1):
        observer_pos = observer_trajectory[k]
        dx = true_states[k, 0] - observer_pos[0]
        dy = true_states[k, 1] - observer_pos[1]
        true_bearing = np.arctan2(dy, dx)
        measurements[k, 0] = true_bearing + np.sqrt(R) * np.random.randn()

    # 初始化CKF
    # 初始状态估计 (假设我们有一个不太准确的初始猜测)
    x0 = np.array([4000.0, 4000.0, 0, -0])

    # 初始状态不确定性
    P0 = np.diag([100.0 ** 2, 100.0 ** 2, 10.0 ** 2, 10.0 ** 2])

    # 过程噪声协方差 (滤波器中使用的估计值，可能与真实值不完全相同)
    Q = np.diag([0.1, 0.1, 0.01, 0.01])

    # 实例化CKF
    ckf = BearingOnlyCKF(x0, P0, Q, R, dt, observer_trajectory)

    # 运行滤波
    estimated_states = np.zeros((steps + 1, 4))
    estimated_covs = np.zeros((steps + 1, 4, 4))

    estimated_states[0] = x0
    estimated_covs[0] = P0

    for k in range(1, steps + 1):
        ckf.step(measurements[k], true_states[k])
        estimated_states[k] = ckf.x
        estimated_covs[k] = ckf.P

    # 计算性能指标
    pos_rmse = np.sqrt(np.mean(np.sum((true_states[:, 0:2] - estimated_states[:, 0:2]) ** 2, axis=1)))
    vel_rmse = np.sqrt(np.mean(np.sum((true_states[:, 2:4] - estimated_states[:, 2:4]) ** 2, axis=1)))

    # 返回结果
    return {
        'true_states': true_states,
        'estimated_states': estimated_states,
        'estimated_covs': estimated_covs,
        'observer_trajectory': observer_trajectory,
        'measurements': measurements,
        'innovation': np.array(ckf.innovation),
        'innovation_covariance': np.array(ckf.innovation_covariance),
        'nis': np.array(ckf.nis),
        'pos_rmse': pos_rmse,
        'vel_rmse': vel_rmse,
        'dt': dt,
        'steps': steps
    }


def run_monte_carlo_simulation(num_runs=100, animate=True):
    """
    执行Monte Carlo仿真

    参数:
    num_runs: 仿真次数
    animate: 是否显示动画
    """
    print(f"开始执行 {num_runs} 次Monte Carlo仿真...")
    start_time = time.time()

    # 用于存储所有仿真结果的变量
    all_true_states = []
    all_estimated_states = []
    all_estimated_covs = []
    all_nis = []
    all_innovation = []
    all_innovation_covariance = []
    all_pos_rmse = []
    all_vel_rmse = []

    # 执行多次仿真
    for i in tqdm(range(num_runs),
                  desc="仿真进度",
                  unit="次",
                  bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{remaining}]"
                  ):


        # 执行单次仿真，使用不同的随机种子
        result = single_simulation(seed=i)

        # 存储结果
        all_true_states.append(result['true_states'])
        all_estimated_states.append(result['estimated_states'])
        all_estimated_covs.append(result['estimated_covs'])
        all_nis.append(result['nis'])
        all_innovation.append(result['innovation'])
        all_innovation_covariance.append(result['innovation_covariance'])
        all_pos_rmse.append(result['pos_rmse'])
        all_vel_rmse.append(result['vel_rmse'])

    # 计算平均值
    # 由于每次仿真的真实状态是不同的，我们使用第一次仿真的真实状态作为参考
    avg_true_states = all_true_states[0]  # 使用第一次仿真的真实轨迹作为参考
    avg_estimated_states = np.mean(np.array(all_estimated_states), axis=0)

    # 协方差矩阵的平均值
    avg_estimated_covs = np.mean(np.array(all_estimated_covs), axis=0)

    # 诊断指标的平均值
    avg_nis = np.mean(np.array(all_nis), axis=0)
    avg_innovation = np.mean(np.array(all_innovation), axis=0)
    avg_innovation_covariance = np.mean(np.array(all_innovation_covariance), axis=0)

    # 计算平均RMSE
    avg_pos_rmse = np.mean(all_pos_rmse)
    avg_vel_rmse = np.mean(all_vel_rmse)

    elapsed_time = time.time() - start_time
    print(f"Monte Carlo仿真完成，耗时: {elapsed_time:.2f} 秒")
    print(f"平均位置RMSE: {avg_pos_rmse:.2f} m")
    print(f"平均速度RMSE: {avg_vel_rmse:.2f} m/s")

    # 将结果整合为一个字典
    mc_results = {
        'true_states': avg_true_states,
        'estimated_states': avg_estimated_states,
        'estimated_covs': avg_estimated_covs,
        'observer_trajectory': result['observer_trajectory'],  # 使用最后一次仿真的观测者轨迹
        'nis': avg_nis,
        'innovation': avg_innovation,
        'innovation_covariance': avg_innovation_covariance,
        'pos_rmse': avg_pos_rmse,
        'vel_rmse': avg_vel_rmse,
        'dt': result['dt'],
        'steps': result['steps'],
        'num_runs': num_runs
    }

    # 可视化结果
    if animate:
        visualize_results(mc_results, animate=True)
    else:
        visualize_results(mc_results, animate=False)

    return mc_results


def visualize_results(results, animate=True):
    """
    可视化仿真结果

    参数:
    results: 包含仿真结果的字典
    animate: 是否创建动画
    """
    true_states = results['true_states']
    estimated_states = results['estimated_states']
    estimated_covs = results['estimated_covs']
    observer_trajectory = results['observer_trajectory']
    nis = results['nis']
    innovation = results['innovation']
    innovation_covariance = results['innovation_covariance']
    dt = results['dt']
    steps = results['steps']
    num_runs = results.get('num_runs', 1)

    # 创建动画或静态图
    if animate:
        # 创建动画
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111)

        # 绘制完整的真实轨迹和观测者轨迹
        ax.plot(true_states[:, 0], true_states[:, 1], 'b-', alpha=0.3, label='真实轨迹')
        ax.plot(observer_trajectory[:, 0], observer_trajectory[:, 1], 'k-', alpha=0.5, label='观测者轨迹')

        # 初始化绘图元素
        target_true, = ax.plot([], [], 'bo', markersize=6)
        target_est, = ax.plot([], [], 'ro', markersize=6)
        observer, = ax.plot([], [], 'ko', markersize=6)
        estimation_line, = ax.plot([], [], 'r-', alpha=0.5)
        ellipse = Ellipse((0, 0), 0, 0, 0, color='r', alpha=0.3)
        ax.add_patch(ellipse)

        # 设置坐标轴
        ax.set_xlim(min(min(true_states[:, 0]), min(observer_trajectory[:, 0])) - 200,
                    max(max(true_states[:, 0]), max(observer_trajectory[:, 0])) + 200)
        ax.set_ylim(min(min(true_states[:, 1]), min(observer_trajectory[:, 1])) - 200,
                    max(max(true_states[:, 1]), max(observer_trajectory[:, 1])) + 200)
        ax.grid(True)
        ax.set_aspect('equal')
        ax.set_xlabel('X 位置 (m)')
        ax.set_ylabel('Y 位置 (m)')
        ax.set_title(f'纯方位目标运动分析 - CKF ({num_runs}次Monte Carlo仿真平均结果)')
        ax.legend()

        # 初始化文本标签
        time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
        mc_text = ax.text(0.02, 0.90, f'Monte Carlo仿真次数: {num_runs}', transform=ax.transAxes)

        def init():
            target_true.set_data([], [])
            target_est.set_data([], [])
            observer.set_data([], [])
            estimation_line.set_data([], [])
            time_text.set_text('')
            ellipse.center = (0, 0)
            ellipse.width = 0
            ellipse.height = 0
            ellipse.angle = 0
            return target_true, target_est, observer, estimation_line, ellipse, time_text, mc_text

        def animate(i):
            # 更新目标和观测者位置
            target_true.set_data([true_states[i, 0]], [true_states[i, 1]])
            target_est.set_data([estimated_states[i, 0]], [estimated_states[i, 1]])
            observer.set_data([observer_trajectory[i, 0]], [observer_trajectory[i, 1]])

            # 更新估计轨迹
            estimation_line.set_data(estimated_states[:i + 1, 0], estimated_states[:i + 1, 1])

            # 更新误差椭圆
            cov = estimated_covs[i, 0:2, 0:2]
            vals, vecs = np.linalg.eig(cov)
            angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
            ellipse.center = (estimated_states[i, 0], estimated_states[i, 1])
            # 95% 置信区间对应于 chi-square 分布的 5.991
            ellipse.width = 2 * np.sqrt(5.991 * vals[0])
            ellipse.height = 2 * np.sqrt(5.991 * vals[1])
            ellipse.angle = angle

            # 更新文本信息
            time_text.set_text(f'时间: {i * dt:.1f} s')

            return target_true, target_est, observer, estimation_line, ellipse, time_text, mc_text

        # 创建动画
        anim = animation.FuncAnimation(fig, animate, init_func=init, frames=steps + 1,
                                       interval=100, blit=True)

        # 保存动画
        # anim.save(f'ckf_mc_{num_runs}_runs.mp4', writer='ffmpeg', fps=10)

        plt.show()

    else:
        # 创建静态图
        plt.figure(figsize=(12, 10))

        # 绘制轨迹
        plt.subplot(2, 1, 1)
        plt.plot(true_states[:, 0], true_states[:, 1], 'b-', label='真实轨迹')
        plt.plot(estimated_states[:, 0], estimated_states[:, 1], 'r--', label=f'{num_runs}次平均估计轨迹')
        plt.plot(observer_trajectory[:, 0], observer_trajectory[:, 1], 'k-', label='观测者轨迹')

        # 每10步绘制误差椭圆
        for k in range(0, steps + 1, 10):
            cov = estimated_covs[k, 0:2, 0:2]
            vals, vecs = np.linalg.eig(cov)
            angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
            ellipse = Ellipse(xy=(estimated_states[k, 0], estimated_states[k, 1]),
                              width=2 * np.sqrt(5.991 * vals[0]), height=2 * np.sqrt(5.991 * vals[1]),
                              angle=angle, alpha=0.3, color='red')
            plt.gca().add_patch(ellipse)

        plt.axis('equal')
        plt.grid(True)
        plt.xlabel('X 位置 (m)')
        plt.ylabel('Y 位置 (m)')
        plt.title(f'{num_runs}次Monte Carlo仿真的平均轨迹')
        plt.legend()

        # 绘制位置误差
        plt.subplot(2, 2, 3)
        pos_error = np.sqrt(np.sum((true_states[:, 0:2] - estimated_states[:, 0:2]) ** 2, axis=1))
        plt.plot(np.arange(0, steps * dt + dt, dt), pos_error)
        plt.grid(True)
        plt.xlabel('时间 (s)')
        plt.ylabel('位置误差 (m)')
        plt.title('位置估计误差')

        # 绘制速度误差
        plt.subplot(2, 2, 4)
        vel_error = np.sqrt(np.sum((true_states[:, 2:4] - estimated_states[:, 2:4]) ** 2, axis=1))
        plt.plot(np.arange(0, steps * dt + dt, dt), vel_error)
        plt.grid(True)
        plt.xlabel('时间 (s)')
        plt.ylabel('速度误差 (m/s)')
        plt.title('速度估计误差')

        plt.tight_layout()
        plt.show()

    # 绘制一致性分析图
    plt.figure(figsize=(12, 8))

    # 绘制NIS值
    plt.subplot(2, 1, 1)
    plt.plot(np.arange(1, steps + 1) * dt, nis)
    chi2_1 = chi2.ppf(0.95, 1)  # 95% 上界值
    plt.axhline(y=chi2_1, color='r', linestyle='--', label='95% 上界')
    plt.axhline(y=chi2.ppf(0.05, 1), color='g', linestyle='--', label='5% 下界')
    plt.grid(True)
    plt.xlabel('时间 (s)')
    plt.ylabel('NIS')
    plt.title(f'{num_runs}次Monte Carlo仿真的平均归一化创新平方 (NIS)')
    plt.legend()

    # 绘制创新序列
    plt.subplot(2, 1, 2)
    plt.plot(np.arange(1, steps + 1) * dt, innovation)
    plt.fill_between(np.arange(1, steps + 1) * dt,
                     -2 * np.sqrt(innovation_covariance), 2 * np.sqrt(innovation_covariance),
                     color='gray', alpha=0.3, label='2σ 范围')
    plt.grid(True)
    plt.xlabel('时间 (s)')
    plt.ylabel('创新 (rad)')
    plt.title(f'{num_runs}次Monte Carlo仿真的平均方位角创新序列')
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 设置蒙特卡洛模拟次数
    num_monte_carlo_runs = 100

    # 运行蒙特卡洛仿真
    run_monte_carlo_simulation(num_runs=num_monte_carlo_runs, animate=True)