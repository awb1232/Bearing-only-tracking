from angel_process import *


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

    #def normalize_angle(self, angle):
    #    """将角度归一化到[-pi, pi]范围内"""
    #    return (angle + np.pi) % (2 * np.pi) - np.pi

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
        #z_residual = z - z_pred
        #z_residual[0] = self.normalize_angle(z_residual[0])
        z_residual = rad2rad2sub1(z, z_pred)

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

    def __init__(self, x0, P0, Q, R, dt, observer_trajectory, measurements, backward=False):
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

        #self.P = P0.copy()  # 协方差矩阵
        # self.x = np.array([0,0,0,0])
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

    def construct_pseudo_linear_measurement(self):
        """
        构造伪线性测量方程
        z = arctan2(dy, dx) => sin(z)*dx - cos(z)*dy = 0

        返回:
        - 伪线性测量矩阵H
        - 伪线性测量值z_pl (通常为0)
        """
        z = self.measurements[self.current_step]
        observer_pos = self.observer_trajectory[self.current_step]
        sin_z = np.sin(z)
        cos_z = np.cos(z)

        # 构造伪线性测量矩阵 H = [sin(z), -cos(z), 0, 0]，该H是对于x轴夹角方位角而言的，对于y轴夹角定义的方位角需要额外考虑
        H = np.zeros((1, self.n))
        #H[0, 0] = sin_z
        #H[0, 1] = -cos_z
        H[0, 0] = -cos_z
        H[0, 1] = sin_z

        # 伪线性测量中的偏置项
        #bias_term = sin_z * observer_pos[0] - cos_z * observer_pos[1]
        bias_term = -cos_z * observer_pos[0] + sin_z * observer_pos[1]

        # 伪线性测量值(通常为0)
        z_pl = np.array([bias_term])

        return H, z_pl

    def normalize_angle(self, angle):
        """将角度归一化到[-pi, pi]范围内"""
        return (angle + np.pi) % (2 * np.pi) - np.pi


    def step(self):
        """执行完整的PLKF步骤：预测和更新"""

        # 预测状态
        Xpre = self.state_transition(self.x)

        # 预测协方差
        Ppre = self.cov_transition(self.P)

        # 构造伪线性测量系统
        H, z_pl = self.construct_pseudo_linear_measurement()

        S = H @ Ppre @ H.T + self.R

        K = Ppre @ H.T / S[0, 0]

        z_pre = H @ Xpre
        z_res = z_pl[0][0] - z_pre[0]

        add1 = K * (z_res)
        add = np.squeeze(K * (z_res))
        #add = np.squeeze(K * (z - H @ Xpre))

        self.x = Xpre + add

        self.P = Ppre - K @ H @ Ppre

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