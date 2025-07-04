from angel_process import *

"""
说明：
1.假设你要进行n次卡尔曼滤波递推，那么observer_trajectory和measurements应该包含从初始状态和往后n个状态下的量测和坐标
2.observer_trajectory可以送入形状为{(n,2)}的二维位置坐标数组，也可以送入形状为{(n,4)}的四维状态坐标数组，但要保证前两个元素是x,y
3.噪声矩阵R取角度或弧度时对滤波器的性能有明显影响，对于EKF/UKF/CKF，三者在R取角度弧度时都能运行，但对于PLKF，R矩阵取角度时可以运行，取弧度时跟踪会发散
4.默认距离单位是米，速度单位是米/秒
5.这个代码没有考虑Q阵，因为仿真的时候被跟踪目标是按照严格的匀速直线运动设置并生成轨迹的
"""

class BearingOnlyEKF:

    def __init__(self, x0, P0, Q, R, dt, observer_trajectory, measurements, backward=False):
        """
        纯方位拓展卡尔曼滤波EKF

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

        if self.backward:
            # 逆向状态转移矩阵
            self.F = np.array([
                [1, 0, -self.dt, 0],
                [0, 1, 0, -self.dt],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])
        else:
            # 正向状态转移矩阵
            self.F = np.array([
                [1, 0, self.dt, 0],
                [0, 1, 0, self.dt],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])

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

    def step(self):

        o_k = self.observer_trajectory[self.current_step]

        # X(k|k-1)
        Xpre = self.F @ self.x

        # P(k|k-1)
        Ppre = self.F @ self.P @ self.F.T
        # Ppre = self.F @ self.P @ self.F.T + self.Q

        # H观测矩阵
        H = np.array([(Xpre[1] - o_k[1])/((Xpre[1] - o_k[1]) ** 2 + (Xpre[0] - o_k[0]) ** 2),
                      -(Xpre[0] - o_k[0])/((Xpre[1] - o_k[1]) ** 2 + (Xpre[0] - o_k[0]) ** 2),
                      0,
                      0])

        # Z(k|k-1) 和 Z(k)
        #Zpre = H @ Xpre 错误的预测方位的方法，因为这个Xpre不是用相对距离和相对速度创建的估计量，而是绝对距离和绝对速度
        Zpre = self.predict_bearing(Xpre)[0]
        Z = self.measurements[self.current_step][0]     # 取索引0是因为measurements是一系列array
        Z_residual = rad1rad2sub1(Z, Zpre)

        # S(k)
        S = H @ self.P @ H.T + self.R
        # S = H @ self.P @ H.T + np.rad2deg(self.R)

        # K(k)
        K = Ppre @ H.T / S

        # X(k|k)
        self.x = Xpre + K * (Z_residual)

        # P(k|k)
        self.P = Ppre - np.outer(K,H) @ Ppre # K和H的形状都是{ndarray(4,)}，不能直接用@相乘会报错

        # 更新步数
        self.current_step += 1


class BearingOnlyPLKF:

    def __init__(self, x0, P0, Q, R, dt, observer_trajectory, measurements, backward=False):
        """
        纯方位伪线性卡尔曼滤波器(PLKF)

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

        if self.backward:
            # 逆向状态转移矩阵
            self.F = np.array([
                [1, 0, -self.dt, 0],
                [0, 1, 0, -self.dt],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])
        else:
            # 正向状态转移矩阵
            self.F = np.array([
                [1, 0, self.dt, 0],
                [0, 1, 0, self.dt],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])

        if len(self.observer_trajectory) != len(self.measurements):
            raise ValueError("传入的坐标序列和方位序列不等长")

        self.current_step = 1  # 当前步（用于索引observer_trajectory和measurements）

    def step(self):

        opos_k = self.observer_trajectory[self.current_step]
        Z_k = self.measurements[self.current_step][0]
        cos_z = np.cos(Z_k)
        sin_z = np.sin(Z_k)

        # X(k|k-1)
        Xpre = self.F @ self.x

        # P(k|k-1)
        Ppre = self.F @ self.P @ self.F.T
        # Ppre = self.F @ self.P @ self.F.T + self.Q

        # d(k|k-1)
        dst_pre_2 = (Xpre[0] - opos_k[0]) ** 2 + (Xpre[1] - opos_k[1]) ** 2

        # n(k)
        R_pl = dst_pre_2 * np.rad2deg(self.R)

        # 伪线性量测方程
        H_pl = np.array([cos_z, -sin_z, 0, 0])

        # S(k)
        S = H_pl @ Ppre @ H_pl.T + R_pl

        # K(k)
        K = Ppre @ H_pl.T / S

        # Z(k|k-1) 和 Z(k)
        Zpl_k = H_pl @ opos_k   # Z(k)
        # Zpre = self.predict_bearing(Xpre)
        Zpre = H_pl @ Xpre
        #Zpre = np.outer(H_pl, Xpre) # 这样Zpre会得到一个4×4矩阵，是不对的
        Z_residual = Zpl_k - Zpre

        # X(k|k)
        self.x = Xpre + K * (Z_residual)

        # P(k|k)
        self.P = Ppre - np.outer(K, H_pl) @ Ppre  # K和H的形状都是{ndarray(4,)}，不能直接用@相乘会报错

        # 更新步数
        self.current_step += 1


class BearingOnlyUKF:

    def __init__(self, x0, P0, Q, R, dt, observer_trajectory, measurements, backward=False):
        """
        纯方位无迹卡尔曼滤波 UKF

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
        self.kappa = 0  # 次要参数，通常设为0 (x为单变量时设置为0 )

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

    def step(self):

        # 生成sigma采样点
        sigma_points = self.generate_sigma_points()

        # 对各采样点进行状态转移计算
        sigma_points_pred = np.array([self.state_transition(sigma, add_noise=False) for sigma in sigma_points])

        # X(k|k-1)
        x_pred = np.sum(self.weights_m.reshape(-1, 1) * sigma_points_pred, axis=0)

        # P(k|k-1)
        P_pred = np.zeros((self.n, self.n))
        for i in range(len(sigma_points_pred)):
            diff = sigma_points_pred[i] - x_pred
            P_pred += self.weights_c[i] * np.outer(diff, diff)

        # P_pred += self.Q

        # Z(k|k-1)
        z_pred = np.array([self.measurement_function(x) for x in sigma_points_pred])
        z_mean = np.sum(self.weights_m.reshape(-1, 1) * z_pred, axis=0)

        # 计算测量预测协方差
        P_zz = 0
        for i in range(len(z_pred)):
            diff = rad1rad2sub1(z_pred[i], z_mean)
            P_zz += self.weights_c[i] * diff ** 2

        P_zz += self.R
        # P_zz += np.rad2deg(self.R)

        # 计算状态与测量的互相关矩阵
        P_xz = np.zeros(self.n)
        for i in range(len(sigma_points_pred)):
           diff_x = sigma_points_pred[i] - self.x
           diff_z = rad1rad2sub1(z_pred[i], z_mean)
           P_xz += self.weights_c[i] * diff_x * diff_z

        # 计算卡尔曼增益
        K = P_xz / P_zz

        Z = self.measurements[self.current_step]
        Z_residual =rad1rad2sub1(Z, z_mean)

        # X(k|k)
        self.x = x_pred + K * Z_residual[0]

        # P(k|k)
        self.P = P_pred - np.outer(K, K) * P_zz

        self.current_step += 1


class BearingOnlyCKF:

    def __init__(self, x0, P0, Q, R, dt, observer_trajectory, measurements, backward=False):
        """
        纯方位容积卡尔曼滤波 CKF

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
        self.weight = 1.0 / self.num_points  # 所有点的权重相等

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
                # cubature_points[i] = self.x + scaled_directions[i] @ sqrt_P

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

    def step(self):

        # 生成立方点
        cubature_points = self.generate_cubature_points()

        # 传播立方点
        propagated_points = np.array([self.state_transition(x, add_noise=False) for x in cubature_points])

        # X(k|k-1)
        x_pred = np.sum(propagated_points * self.weight, axis=0)

        # P(k|k-1)
        P_pred = np.zeros((self.n, self.n))
        for i in range(len(propagated_points)):
            # diff = propagated_points[i] - x_pred
            # P_pred += self.weight * np.outer(diff, diff)

            P_pred += self.weight * np.outer(propagated_points[i], propagated_points[i])

        # 添加过程噪声协方差
        # P_pred += (-np.outer(x_pred, x_pred) + self.Q)
        P_pred += (-np.outer(x_pred, x_pred))  # 0627，注释掉+ self.Q以后CKF能正常跟踪

        # 上述计算与下面等价：
        # for i in range(len(propagated_points)):
        #     diff = propagated_points[i] - x_pred
        #     P_pred += self.weight * np.outer(diff, diff)
        # P_pred += self.Q

        # 通过测量函数变换传播点
        z_points = np.array([self.measurement_function(x) for x in propagated_points])

        # 计算预测测量均值
        z_pred = np.sum(z_points * self.weight, axis=0)

        # 计算预测测量协方差
        # P_zz = 0
        # for i in range(len(z_points)):
        #     diff = z_points[i] - z_pred
        #     diff[0] = self.normalize_angle(diff[0])
        #     P_zz += self.weight * diff[0] ** 2

        P_zz = 0
        for i in range(len(z_points)):
            diff = rad1rad2sub1(z_points[i], z_pred)
            P_zz += self.weight * diff[0] ** 2 #有没有平方影响不大

        # 添加测量噪声协方差
        P_zz += self.R
        # P_zz += np.rad2deg(self.R)

        # 计算状态与测量的互相关矩阵
        P_xz = np.zeros(self.n)
        for i in range(len(propagated_points)):
            #diff_x = propagated_points[i] - self.x
            diff_x = propagated_points[i] - x_pred
            diff_z = rad1rad2sub1(z_points[i], z_pred)
            P_xz += self.weight * diff_x * diff_z[0]

        # 计算卡尔曼增益
        K = P_xz / P_zz

        # 计算测量残差
        z = self.measurements[self.current_step]  # 获取当前回合的量测方位角
        z_residual = rad1rad2sub1(z, z_pred)

        # X(k|k)
        self.x = x_pred + K * z_residual[0]

        # P(k|k)
        self.P = P_pred - np.outer(K, K) * P_zz

        self.current_step += 1
