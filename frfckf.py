import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.animation as animation
from scipy.stats import chi2

plt.rcParams['font.sans-serif'] = ['SimSun']  # 使用宋体字体
plt.rcParams['axes.unicode_minus'] = False  # 显示负号


class BearingOnlyCKF:
    """
    使用立方卡尔曼滤波器(CKF)进行纯方位目标运动分析
    """

    def __init__(self, x0, P0, Q, R, dt, observer_trajectory, backward=False, name="CKF"):
        """
        初始化CKF参数

        参数:
        x0: 初始状态向量 [x, y, vx, vy]
        P0: 初始协方差矩阵
        Q: 过程噪声协方差矩阵
        R: 测量噪声协方差 (标量)
        dt: 时间步长
        observer_trajectory: 观测者轨迹，每行为一个时间步的位置 [x, y]
        backward: 是否为逆向滤波
        name: 滤波器名称，用于区分不同实例
        """
        self.n = len(x0)  # 状态维度
        self.x = x0.copy()  # 状态向量
        self.P = P0.copy()  # 协方差矩阵
        self.Q = Q.copy()  # 过程噪声协方差矩阵
        self.R = R  # 测量噪声协方差 (标量)
        self.dt = dt  # 时间步长
        self.observer_trajectory = observer_trajectory  # 观测者轨迹
        self.current_step = 0  # 当前步
        self.backward = backward  # 是否为逆向滤波
        self.name = name  # 滤波器名称

        # CKF参数
        self.num_points = 2 * self.n  # CKF使用2n个立方点
        self.weight = 1.0 / (2 * self.n)  # 所有点的权重相等

        # 保存诊断信息
        self.innovation = []  # 保存创新序列
        self.innovation_covariance = []  # 保存创新协方差
        self.nees = []  # 归一化估计误差平方
        self.nis = []  # 归一化创新平方

        # 保存状态估计和协方差的历史
        self.state_history = [x0.copy()]
        self.covariance_history = [P0.copy()]

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
        支持正向和逆向转移
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
        返回以弧度表示的方位角 (相对于y轴)
        """
        if step is None:
            step = self.current_step

        observer_pos = self.observer_trajectory[step]
        dx = x[0] - observer_pos[0]
        dy = x[1] - observer_pos[1]
        # 修改为相对于y轴的方位角计算
        bearing = np.arctan2(dx, dy)  # 交换dx和dy的位置，使方位角相对于y轴

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

        # 保存当前状态和协方差
        self.state_history.append(self.x.copy())
        self.covariance_history.append(self.P.copy())

        # 更新当前步
        if self.backward:
            self.current_step -= 1
        else:
            self.current_step += 1

        return self.x, self.P


def run_simulation(seed=None, backward_start_step=50, animate=True):
    """
    运行模拟，包括：
    1. 传统的前向CKF
    2. 从指定步骤开始的逆向CKF
    3. 使用优化初值的新前向CKF

    参数:
    seed: 随机种子
    backward_start_step: 开始逆向滤波的步骤
    animate: 是否生成动画
    """
    # 设置随机种子以保证可重复性
    np.random.seed(seed)

    # 设置仿真参数
    dt = 1.0  # 时间步长 (秒)
    simulation_time = 1000  # 仿真时长
    steps = int(simulation_time / dt)  # 仿真步数

    # 观测者轨迹 (弯曲轨迹以提高可观测性)
    observer_trajectory = np.zeros((steps + 1, 2))
    for k in range(steps + 1):
        t = k * dt
        observer_trajectory[k, 0] = 100 * np.sin(t / 20)
        observer_trajectory[k, 1] = 50 * np.sin(t / 10)

    # 真实目标初始状态和运动
    true_x0 = np.array([1000.0, 1500.0, 10.0, -5.0])  # [x, y, vx, vy]
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
    R = 0.005  # 测量噪声方差 (弧度)

    for k in range(steps + 1):
        observer_pos = observer_trajectory[k]
        dx = true_states[k, 0] - observer_pos[0]
        dy = true_states[k, 1] - observer_pos[1]
        # 使用相对于y轴的方位角定义
        true_bearing = np.arctan2(dx, dy)
        measurements[k, 0] = true_bearing + np.sqrt(R) * np.random.randn()

    # 初始化参数
    # 初始状态估计 (假设我们有一个不太准确的初始猜测)
    x0 = np.array([900.0, 1400.0, 8.0, -4.0])
    x0 = np.array([900.0, 1400.0, 0.0, -0.0])

    # 初始状态不确定性
    P0 = np.diag([100.0 ** 2, 100.0 ** 2, 10.0 ** 2, 10.0 ** 2])

    # 过程噪声协方差 (滤波器中使用的估计值，可能与真实值不完全相同)
    Q = np.diag([0.1, 0.1, 0.01, 0.01])

    # 1. 实例化传统的前向CKF
    forward_ckf = BearingOnlyCKF(x0, P0, Q, R, dt, observer_trajectory, name="CKF")

    # 运行前向滤波
    forward_states = np.zeros((steps + 1, 4))
    forward_covs = np.zeros((steps + 1, 4, 4))

    forward_states[0] = x0
    forward_covs[0] = P0

    for k in range(1, steps + 1):
        forward_ckf.step(measurements[k], true_states[k])
        forward_states[k] = forward_ckf.x
        forward_covs[k] = forward_ckf.P

    # 2. 运行逆向滤波获取优化初值
    optimized_initial_state = None
    optimized_initial_covariance = None

    if backward_start_step > 0 and backward_start_step <= steps:
        print("\n=== 开始逆向滤波优化初值 ===")
        print(f"从第 {backward_start_step} 步开始逆向滤波")

        # 使用前向滤波在指定步骤的估计作为逆向滤波的起点
        backward_x0 = forward_states[backward_start_step].copy()
        backward_P0 = forward_covs[backward_start_step].copy()

        # 创建逆向轨迹 (从backward_start_step到0)
        backward_trajectory = observer_trajectory[:(backward_start_step + 1)][::-1]

        # 创建逆向测量
        backward_measurements = measurements[:(backward_start_step + 1)][::-1]

        # 实例化逆向CKF
        backward_ckf = BearingOnlyCKF(
            backward_x0, backward_P0, Q, R, dt, backward_trajectory, backward=True, name="逆向CKF"
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

        # 打印真实初值、前向滤波初始估计和逆向滤波优化后的初值
        print("\n=== 初值比较 ===")
        print(f"真实初值:       {true_x0}")
        print(f"前向滤波初始估计: {x0}")
        print(f"逆向滤波优化初值: {optimized_initial_state}")

        # 计算误差
        forward_init_error = np.linalg.norm(x0 - true_x0)
        backward_init_error = np.linalg.norm(optimized_initial_state - true_x0)
        print(f"\n前向滤波初值误差: {forward_init_error:.2f}")
        print(f"逆向滤波初值误差: {backward_init_error:.2f}")
        improvement = (forward_init_error - backward_init_error) / forward_init_error * 100
        print(f"误差改善: {improvement:.2f}%")

    # 3. 使用优化的初值重新运行前向滤波
    if optimized_initial_state is not None:
        print("\n=== 使用优化初值重新运行前向滤波 ===")

        # 实例化新的前向CKF，使用优化的初值
        optimized_forward_ckf = BearingOnlyCKF(
            optimized_initial_state,
            optimized_initial_covariance,
            Q, R, dt, observer_trajectory,
            name="FRCKF"
        )

        # 运行优化初值的前向滤波
        optimized_forward_states = np.zeros((steps + 1, 4))
        optimized_forward_covs = np.zeros((steps + 1, 4, 4))

        optimized_forward_states[0] = optimized_initial_state
        optimized_forward_covs[0] = optimized_initial_covariance

        for k in range(1, steps + 1):
            optimized_forward_ckf.step(measurements[k], true_states[k])
            optimized_forward_states[k] = optimized_forward_ckf.x
            optimized_forward_covs[k] = optimized_forward_ckf.P
    else:
        # 如果没有运行逆向滤波，则优化后的状态就是原始状态
        optimized_forward_states = forward_states
        optimized_forward_covs = forward_covs

    # 计算RMSE
    # 传统CKF的RMSE
    forward_pos_rmse = np.sqrt(np.mean(np.sum((true_states[:, 0:2] - forward_states[:, 0:2]) ** 2, axis=1)))
    forward_vel_rmse = np.sqrt(np.mean(np.sum((true_states[:, 2:4] - forward_states[:, 2:4]) ** 2, axis=1)))

    # 优化初值CKF的RMSE
    optimized_pos_rmse = np.sqrt(np.mean(np.sum((true_states[:, 0:2] - optimized_forward_states[:, 0:2]) ** 2, axis=1)))
    optimized_vel_rmse = np.sqrt(np.mean(np.sum((true_states[:, 2:4] - optimized_forward_states[:, 2:4]) ** 2, axis=1)))

    # 计算每个时间步的误差，用于绘制误差随时间变化图
    forward_pos_error = np.sqrt(np.sum((true_states[:, 0:2] - forward_states[:, 0:2]) ** 2, axis=1))
    forward_vel_error = np.sqrt(np.sum((true_states[:, 2:4] - forward_states[:, 2:4]) ** 2, axis=1))

    optimized_pos_error = np.sqrt(np.sum((true_states[:, 0:2] - optimized_forward_states[:, 0:2]) ** 2, axis=1))
    optimized_vel_error = np.sqrt(np.sum((true_states[:, 2:4] - optimized_forward_states[:, 2:4]) ** 2, axis=1))

    # 输出性能对比
    print("\n=== 性能对比 ===")
    print(f"CKF - 位置RMSE: {forward_pos_rmse:.2f} m, 速度RMSE: {forward_vel_rmse:.2f} m/s")
    print(f"FRCKF - 位置RMSE: {optimized_pos_rmse:.2f} m, 速度RMSE: {optimized_vel_rmse:.2f} m/s")

    pos_improvement = (forward_pos_rmse - optimized_pos_rmse) / forward_pos_rmse * 100
    vel_improvement = (forward_vel_rmse - optimized_vel_rmse) / forward_vel_rmse * 100

    print(f"位置RMSE改善: {pos_improvement:.2f}%")
    print(f"速度RMSE改善: {vel_improvement:.2f}%")

    # 可视化轨迹和误差
    plt.figure(figsize=(12, 10))

    # 绘制轨迹
    plt.subplot(2, 1, 1)
    plt.plot(true_states[:, 0], true_states[:, 1], 'k-', label='真实轨迹')
    plt.plot(forward_states[:, 0], forward_states[:, 1], 'b--', label='CKF估计')
    plt.plot(optimized_forward_states[:, 0], optimized_forward_states[:, 1], 'r-.', label='FRCKF估计')
    plt.plot(observer_trajectory[:, 0], observer_trajectory[:, 1], 'g-', alpha=0.5, label='观测者轨迹')

    # 绘制初始点
    plt.plot(true_x0[0], true_x0[1], 'ko', markersize=10, label='真实初值')
    plt.plot(x0[0], x0[1], 'bo', markersize=10, label='原始初值估计')
    if optimized_initial_state is not None:
        plt.plot(optimized_initial_state[0], optimized_initial_state[1], 'ro', markersize=10, label='优化初值估计')

    plt.grid(True)
    plt.axis('equal')
    plt.xlabel('X 位置 (m)')
    plt.ylabel('Y 位置 (m)')
    plt.title('目标轨迹对比')
    plt.legend()

    # 绘制误差随时间变化
    time_steps = np.arange(0, simulation_time + dt, dt)

    # 位置误差
    plt.subplot(2, 2, 3)
    plt.plot(time_steps, forward_pos_error, 'b-', label='CKF')
    plt.plot(time_steps, optimized_pos_error, 'r-', label='FRCKF')
    plt.grid(True)
    plt.xlabel('时间 (s)')
    plt.ylabel('位置误差 (m)')
    plt.title('位置估计误差对比')
    plt.legend()

    # 速度误差
    plt.subplot(2, 2, 4)
    plt.plot(time_steps, forward_vel_error, 'b-', label='CKF')
    plt.plot(time_steps, optimized_vel_error, 'r-', label='FRCKF')
    plt.grid(True)
    plt.xlabel('时间 (s)')
    plt.ylabel('速度误差 (m/s)')
    plt.title('速度估计误差对比')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # 绘制每个状态变量的误差对比
    plt.figure(figsize=(14, 10))

    # X位置误差
    plt.subplot(2, 2, 1)
    x_error_forward = np.abs(true_states[:, 0] - forward_states[:, 0])
    x_error_optimized = np.abs(true_states[:, 0] - optimized_forward_states[:, 0])
    plt.plot(time_steps, x_error_forward, 'b-', label='CKF')
    plt.plot(time_steps, x_error_optimized, 'r-', label='FRCKF')
    plt.grid(True)
    plt.xlabel('时间 (s)')
    plt.ylabel('X位置误差 (m)')
    plt.title('X位置估计误差对比')
    plt.legend()

    # Y位置误差
    plt.subplot(2, 2, 2)
    y_error_forward = np.abs(true_states[:, 1] - forward_states[:, 1])
    y_error_optimized = np.abs(true_states[:, 1] - optimized_forward_states[:, 1])
    plt.plot(time_steps, y_error_forward, 'b-', label='CKF')
    plt.plot(time_steps, y_error_optimized, 'r-', label='FRCKF')
    plt.grid(True)
    plt.xlabel('时间 (s)')
    plt.ylabel('Y位置误差 (m)')
    plt.title('Y位置估计误差对比')
    plt.legend()

    # X速度误差
    plt.subplot(2, 2, 3)
    vx_error_forward = np.abs(true_states[:, 2] - forward_states[:, 2])
    vx_error_optimized = np.abs(true_states[:, 2] - optimized_forward_states[:, 2])
    plt.plot(time_steps, vx_error_forward, 'b-', label='CKF')
    plt.plot(time_steps, vx_error_optimized, 'r-', label='FRCKF')
    plt.grid(True)
    plt.xlabel('时间 (s)')
    plt.ylabel('X速度误差 (m/s)')
    plt.title('X速度估计误差对比')
    plt.legend()

    # Y速度误差
    plt.subplot(2, 2, 4)
    vy_error_forward = np.abs(true_states[:, 3] - forward_states[:, 3])
    vy_error_optimized = np.abs(true_states[:, 3] - optimized_forward_states[:, 3])
    plt.plot(time_steps, vy_error_forward, 'b-', label='CKF')
    plt.plot(time_steps, vy_error_optimized, 'r-', label='FRCKF')
    plt.grid(True)
    plt.xlabel('时间 (s)')
    plt.ylabel('Y速度误差 (m/s)')
    plt.title('Y速度估计误差对比')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # 创建动画 (如果需要)
    if animate:
        # 创建动画
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111)

        # 绘制观测者轨迹
        ax.plot(observer_trajectory[:, 0], observer_trajectory[:, 1], 'g-', alpha=0.5, label='观测者轨迹')

        # 绘制完整的真实轨迹
        ax.plot(true_states[:, 0], true_states[:, 1], 'k-', alpha=0.5, label='真实轨迹')

        # 绘制传统CKF和优化初值CKF的完整估计轨迹
        ax.plot(forward_states[:, 0], forward_states[:, 1], 'b--', alpha=0.5, label='CKF估计')
        ax.plot(optimized_forward_states[:, 0], optimized_forward_states[:, 1], 'r-.', alpha=0.5, label='FRCKF估计')

        # 绘制初始点
        ax.plot(true_x0[0], true_x0[1], 'ko', markersize=10, label='真实初值')
        ax.plot(x0[0], x0[1], 'bo', markersize=10, label='原始初值估计')
        if optimized_initial_state is not None:
            ax.plot(optimized_initial_state[0], optimized_initial_state[1], 'ro', markersize=10, label='优化初值估计')

        # 初始化动态绘图元素
        target_true, = ax.plot([], [], 'ko', markersize=8)
        target_forward, = ax.plot([], [], 'bo', markersize=8)
        target_optimized, = ax.plot([], [], 'ro', markersize=8)
        observer, = ax.plot([], [], 'go', markersize=8)
        bearing_line, = ax.plot([], [], 'g-', alpha=0.7)

        # 初始化文本标签
        time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
        error_text_forward = ax.text(0.02, 0.90, '', transform=ax.transAxes)
        error_text_optimized = ax.text(0.02, 0.85, '', transform=ax.transAxes)

        # 设置坐标轴
        ax.set_xlim(min(min(true_states[:, 0]), min(observer_trajectory[:, 0])) - 200,
                    max(max(true_states[:, 0]), max(observer_trajectory[:, 0])) + 200)
        ax.set_ylim(min(min(true_states[:, 1]), min(observer_trajectory[:, 1])) - 200,
                    max(max(true_states[:, 1]), max(observer_trajectory[:, 1])) + 200)
        ax.grid(True)
        ax.set_aspect('equal')
        ax.set_xlabel('X 位置 (m)')
        ax.set_ylabel('Y 位置 (m)')
        ax.set_title('纯方位目标运动分析 - CKF vs FRCKF')
        ax.legend(loc='upper left')

        def init():
            target_true.set_data([], [])
            target_forward.set_data([], [])
            target_optimized.set_data([], [])
            observer.set_data([], [])
            bearing_line.set_data([], [])
            time_text.set_text('')
            error_text_forward.set_text('')
            error_text_optimized.set_text('')
            return target_true, target_forward, target_optimized, observer, bearing_line, time_text, error_text_forward, error_text_optimized

        def animate(i):
            # 更新真实目标位置
            target_true.set_data([true_states[i, 0]], [true_states[i, 1]])

            # 更新传统CKF估计位置
            target_forward.set_data([forward_states[i, 0]], [forward_states[i, 1]])

            # 更新FRCKF估计位置
            target_optimized.set_data([optimized_forward_states[i, 0]], [optimized_forward_states[i, 1]])

            # 更新观测者位置
            observer.set_data([observer_trajectory[i, 0]], [observer_trajectory[i, 1]])

            # 更新方位线
            obs_x, obs_y = observer_trajectory[i]
            bearing = measurements[i, 0]
            line_length = 2000  # 方位线长度
            # 使用新的方位角定义（相对于y轴的角度）
            bearing_x = obs_x + line_length * np.sin(bearing)
            bearing_y = obs_y + line_length * np.cos(bearing)
            bearing_line.set_data([obs_x, bearing_x], [obs_y, bearing_y])

            # 更新文本信息
            time_text.set_text(f'时间: {i * dt:.1f} s')
            forward_error = np.sqrt(np.sum((true_states[i, 0:2] - forward_states[i, 0:2]) ** 2))
            optimized_error = np.sqrt(np.sum((true_states[i, 0:2] - optimized_forward_states[i, 0:2]) ** 2))
            error_text_forward.set_text(f'CKF误差: {forward_error:.1f} m')
            error_text_optimized.set_text(f'FRCKF误差: {optimized_error:.1f} m')

            return target_true, target_forward, target_optimized, observer, bearing_line, time_text, error_text_forward, error_text_optimized

        # 创建动画
        anim = animation.FuncAnimation(fig, animate, init_func=init, frames=steps + 1,
                                       interval=100, blit=True)

        # 显示动画
        plt.show()

        # 可以保存动画
        # anim.save('ckf_comparison.mp4', writer='ffmpeg', fps=10)

    return {
        'true_states': true_states,
        'forward_states': forward_states,
        'optimized_forward_states': optimized_forward_states,
        'forward_pos_rmse': forward_pos_rmse,
        'forward_vel_rmse': forward_vel_rmse,
        'optimized_pos_rmse': optimized_pos_rmse,
        'optimized_vel_rmse': optimized_vel_rmse,
        'pos_improvement': pos_improvement,
        'vel_improvement': vel_improvement
    }


if __name__ == "__main__":
    run_simulation(seed=None, backward_start_step=200, animate=True)