"""
    reference of author Russell Shomberg's mpcekf
    # 0306 ekf successfully refered as the thesis
      previous mistake: the unit of _DST_SIGMA/ _SPD_SIGMA_2/ was incorrect, the same problem occured when initialing class CEKF dst_est.
    # 0306 mgekf successfully refered as the thesis
      previous mistake: missing a negative sign on the second elemet of matrix H in mgekf update process
    # 0309 测试：
      把速度调为20kmh，60kmh采样时间15秒，算法会发散
"""

import numpy as np
import matplotlib.pyplot as plt

_SONAR_BEARING_ERROR = 0.1
_MEAN_RANGE = 50 * 1000
_RANGE_VARIANCE = .2
_SPEED_VARIANCE = 0.01

_MIN_DETECTION_RANGE = 10.0
_MAX_TARGET_SPEED = 5.0

_OBSERVER_SPD = 600 * 1000 / 3600 # kmph to mps
_TGT_SPD = 200 * 1000 / 3600 # kmph to mps
_TGT_DST = 70 * 1000    # km to m
_TGT_CRS = 45 # deg

_DST_SIGMA = 15 # KM
#_SPD_SIGMA = np.sqrt(0.2*3600)
_SPD_SIGMA_2 = 0.02

R = np.diag([1, 1, 1, 1]) * np.deg2rad(_SONAR_BEARING_ERROR) ** 2
#R = np.diag([1, 1, 1, 1]) * _SONAR_BEARING_ERROR

class CEKF:

    def __init__(self, bearing):
        """
        生成EKF的初始估计

        :param bearing:
        """
        dst_est = 60 * 1000 if _TGT_DST  == 70 * 1000 else 50 * 1000
        dst_est += np.random.normal(0, _DST_SIGMA)
        self.Xest = np.array([dst_est * np.sin(bearing),
                              dst_est * np.cos(bearing),
                              0,
                              0])
        self.Pest = np.diag([_DST_SIGMA ** 2, _DST_SIGMA ** 2, _SPD_SIGMA_2, _SPD_SIGMA_2])

    def update(self, bearing, Xo, dT):
        """
        用拓展卡尔曼滤波更新X和P

        :param bearing: 最新的方位角(弧度)
        :param Xo: 观测者的状态历史
        :param dT: 采样间隔
        :return:
        """
        U = np.array([Xo[-1][0] - Xo[-2][0] - dT * Xo[-2][2],
                      Xo[-1][1] - Xo[-2][1] - dT * Xo[-2][3],
                      Xo[-1][2] - Xo[-2][2],
                      Xo[-1][3] - Xo[-2][3]])

        # 状态转移矩阵
        A = np.array([[1, 0, dT, 0],
                      [0, 1, 0, dT],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])

        # X的先验估计
        Xpre = A @ self.Xest - U

        # P的先验估计
        Ppre = A @ self.Pest @ A.T

        # 方位角的先验估计
        bearingPred = np.arctan2(Xpre[0], Xpre[1])

        bearingError = bearing - bearingPred

        # 观测矩阵
        H = np.array(
            [Xpre[1] / (Xpre[0] ** 2 + Xpre[1] ** 2),
             -Xpre[0] / (Xpre[0] ** 2 + Xpre[1] ** 2),
             0,
             0])

        # 基于P的先验计算新息协方差矩阵Sk = Hk Pk|k-1 HkT + Rk
        S = H @ Ppre @ H.T + R

        # 卡尔曼增益
        Kg = Ppre @ H.T @ np.linalg.inv(S)

        # X的后验估计
        self.Xest= Xpre + Kg * (bearingError)

        # P的后验估计
        self.Pest = (np.eye(4) - Kg @ H) @ Ppre

        #return self.Xest


class MGEKF:

    def __init__(self, bearing):
        """
        生成MGEKF的初始估计（与EKF应相同）

        :param bearing:
        """

        dst_est = 60 * 1000 if _TGT_DST  == 70 * 1000 else 50 * 1000
        dst_est += np.random.normal(0, _DST_SIGMA)
        self.Xest = np.array([dst_est * np.sin(bearing),
                              dst_est * np.cos(bearing),
                              0,
                              0])
        self.Pest = np.diag([_DST_SIGMA ** 2, _DST_SIGMA ** 2, _SPD_SIGMA_2, _SPD_SIGMA_2])


    def update(self, bearing, Xo, dT):
        """

        :param bearing:
        :param Xo:
        :param dT:
        :return:
        """

        U = np.array([Xo[-1][0] - Xo[-2][0] - dT * Xo[-2][2],
                      Xo[-1][1] - Xo[-2][1] - dT * Xo[-2][3],
                      Xo[-1][2] - Xo[-2][2],
                      Xo[-1][3] - Xo[-2][3]])

        # 状态转移矩阵
        A = np.array([[1, 0, dT, 0],
                      [0, 1, 0, dT],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]

                      ])

        # X的先验估计
        Xpre = A @ self.Xest - U

        # P的先验估计
        Ppre = A @ self.Pest @ A.T

        # 方位角的先验估计
        bearingPred = np.arctan2(Xpre[0], Xpre[1])

        bearingError = bearing - bearingPred

        # MGEKF对H做出修改
        H = np.array([np.cos(bearingPred) / (Xpre[0] * np.sin(bearingPred) + Xpre[1] * np.cos(bearingPred)),
                      -np.cos(bearingPred) / (Xpre[0] * np.sin(bearingPred) + Xpre[1] * np.cos(bearingPred)),
                      0,
                      0])

        # 基于P的先验计算新息协方差矩阵Sk = Hk Pk|k-1 HkT + Rk
        S = H @ Ppre @ H.T + R

        # 卡尔曼增益
        Kg = Ppre @ H.T @ np.linalg.inv(S)

        # X的后验估计
        self.Xest = Xpre + Kg * (bearingError)

        # P的后验估计
        self.Pest = (np.eye(4) - Kg @ H) @ Ppre


class PLEKF:
    def __init__(self, bearing):
        """


        :param bearing:
        """

        self.Xest = np.array([0,
                              0,
                              0,
                              0])
        self.Pest = np.diag([1, 1, 1, 1])

    def update(self, bearing, Xo, dT):
        """
        用伪线性卡尔曼滤波更新X和P

        :param bearing: 最新的方位角(弧度)
        :param Xo: 观测者的状态历史
        :param dT: 采样间隔
        :return:
        """

        U = np.array([Xo[-1][0] - Xo[-2][0] - dT * Xo[-2][2],
                      Xo[-1][1] - Xo[-2][1] - dT * Xo[-2][3],
                      Xo[-1][2] - Xo[-2][2],
                      Xo[-1][3] - Xo[-2][3]])

        # 状态转移矩阵
        A = np.array([[1, 0, dT, 0],
                      [0, 1, 0, dT],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]

                      ])

        # X的先验估计
        Xpre = A @ self.Xest - U

        # P的先验估计
        Ppre = A @ self.Pest @ A.T

        H = np.array([np.cos(bearing),
                      -np.sin(bearing),
                      0,
                      0])

        # 基于P的先验计算新息协方差矩阵Sk = Hk Pk|k-1 HkT + Rk
        S = H @ Ppre @ H.T + R
        #S = H @ Ppre @ H.T + np.diag([1, 1, 1, 1])

        # 卡尔曼增益
        Kg = Ppre @ H.T @ np.linalg.inv(S)

        # X的后验估计
        #self.Xest = Xpre + Kg * (bearingError)
        self.Xest = Xpre + Kg * H @ Xpre

        # P的后验估计
        self.Pest = (np.eye(4) - Kg @ H) @ Ppre


class MPCEKF:

    def __init__(self, bearing):
        """

        :param bearing:
        """

        dst_est = 60 * 1000 if _TGT_DST % 35 == 0 else 50 * 1000
        dst_est += np.random.normal(0, _DST_SIGMA)

        self.Yest = np.array([0,
                              0,
                              bearing,
                              1/_TGT_DST])
        self.Pest = np.diag([6e-3, 3.6e-7, 9e-3, 1e-11])
        """
        self.Pest = np.diag([_SPD_SIGMA_2 / (dst_est ** 2),
                              _SPD_SIGMA_2 / (dst_est ** 2),
                              0.001,
                              (_DST_SIGMA ** 2) / (dst_est ** 4) ])
        """
    def update(self, bearing, Xo, dT):
        """
                用伪线性卡尔曼滤波更新X和P

                :param bearing: 最新的方位角(弧度)
                :param Xo: 观测者的状态历史
                :param dT: 采样间隔
                :return:
                """

        U = np.array([Xo[-1][0] - Xo[-2][0] - dT * Xo[-2][2],
                      Xo[-1][1] - Xo[-2][1] - dT * Xo[-2][3],
                      Xo[-1][2] - Xo[-2][2],
                      Xo[-1][3] - Xo[-2][3]])

        # Y的先验估计和状态转移矩阵
        Ypre,A = f_yPre_and_phi(bearing, self.Yest, U, dT)

        # P的先验估计
        Ppre = A @ self.Pest @ A.T

        # MP观测矩阵
        H = np.array([0, 0, 1, 0])

        S = H @ Ppre @ H.T + np.deg2rad(_SONAR_BEARING_ERROR) ** 2

        G = Ppre @ H.T * (S) ** -1

        self.Yest = Ypre + G * (bearing - H @ Ypre)
        self.Pest = (np.eye(4) - G @ H) @ Ppre


class Ship:

    def __init__(self, state):
        self.X = state

    def update(self, dT, newCourse=None):
        '''step ship forward and record path'''

        Qsim = np.random.randn(4) * .1
        Qsim[2:3] = 0
        xPrev = self.X.copy()

        if newCourse:
            self.X[2] = newCourse[0]
            self.X[3] = newCourse[1]

        motionModel = np.array([[1, 0, dT, 0],
                                [0, 1, 0, dT],
                                [0, 0, 1, 0],
                                [0, 0, 0, 1]])

        self.X = motionModel @ self.X# + Qsim

        # Ship motion as input to relative motion of other ships
        U = np.array([self.X[0] - (xPrev[0] + dT * xPrev[2]),
                      self.X[1] - (xPrev[1] + dT * xPrev[3]),
                      self.X[2] - xPrev[2],
                      self.X[3] - xPrev[3]
                      ])

        return self.X, U


def sonar_bearing(ownship, ship):
    '''Return a bearing affected by noise to a target ship'''

    xRel = ship.X - ownship.X
    bearing = np.arctan2(xRel[0], xRel[1])
    noise = np.random.normal(0, 1) * np.deg2rad(_SONAR_BEARING_ERROR)
    return bearing + noise


def xy2mpc(X):
    '''Converts to modified polar coords'''

    Y = np.array([(X[2] * X[1] - X[3] * X[0]) / (X[0] ** 2 + X[1] ** 2),  # bearing_rate
                  (X[2] * X[0] + X[3] * X[1]) / (X[0] ** 2 + X[1] ** 2),  # range_rate /range
                  np.arctan2(X[0], X[1]),  # bearing
                  1.0 / np.sqrt(X[0] ** 2 + X[1] ** 2)  # 1/range
                  ])
    return Y


def mpc2xy(Y):
    '''convert from modified polar coords'''

    X = (1 / Y[3]) * np.array([np.sin(Y[2]),
                               np.cos(Y[2]),
                               Y[1] * np.sin(Y[2]) + Y[0] * np.cos(Y[2]),
                               Y[1] * np.cos(Y[2]) - Y[0] * np.sin(Y[2])
                               ])
    return X


def f_yPre_and_phi(B, Y, U, dT):
    """
    按照论文公式37和38计算MP坐标系下Y状态的先验估计

    :param B: 方位角
    :param Y: 状态Y
    :param U:
    :param dT:
    :return:
    """

    """计算Y的先验估计"""
    a = np.array([dT * Y[0] - Y[3] * (U[0] * np.cos(B) - U[1] * np.sin(B)),
                  1 + dT * Y[1] - Y[3] * (U[0] * np.sin(B) + U[1] * np.cos(B)),
                  Y[0] - Y[3] * (U[2] * np.cos(B) - U[3] * np.sin(B)),
                  Y[1] - Y[3] * (U[2] * np.sin(B) - U[3] * np.cos(B))])

    yPred = np.array([(a[1] * a[2] - a[0] * a[3]) / (a[0] ** 2 + a[1] ** 2),
                      (a[0] * a[2] + a[1] * a[3]) / (a[0] ** 2 + a[1] ** 2),
                      Y[2] + np.arctan2(a[0], a[1]),
                      Y[3] / np.sqrt(a[0] ** 2 + a[1] ** 2)])

    """计算修正极坐标系下的状态转移矩阵"""
    # 公式A6
    d11 = (-a[0] * (a[1] * a[2] - a[0] * a[3]) - a[1] * (a[0] * a[2] + a[1] * a[3])) / (a[0] ** 2 + a[1] ** 2) ** 2
    d21 = (-a[0] * (a[0] * a[2] + a[1] * a[3]) + a[1] * (a[1] * a[2] - a[0] * a[3])) / (a[0] ** 2 + a[1] ** 2) ** 2
    d31 = a[1] / (a[0] ** 2 + a[1] ** 2)
    d41 = -a[2] * Y[3] / (a[0] ** 2 + a[1] ** 2) ** (3.0 / 2.0)
    d32 = -a[0] / (a[0] ** 2 + a[1] ** 2)
    d42 = -a[1] * Y[3] / (a[0] ** 2 + a[1] ** 2) ** (3.0 / 2.0)
    d13 = a[1] / (a[0] ** 2 + a[1] ** 2)

    # 公式A8
    e14 = -(U[0] * np.cos(Y[2]) - U[1] * np.sin(Y[2]))
    e24 = -(U[0] * np.sin(Y[2]) + U[1] * np.cos(Y[2]))
    e34 = -(U[2] * np.cos(Y[2]) - U[3] * np.sin(Y[2]))
    e44 = -(U[2] * np.sin(Y[2]) + U[3] * np.cos(Y[2]))
    e13 = -Y[3] * e24
    e23 = Y[3] * e14
    e33 = -Y[3] * e44
    e43 = Y[3] * e34

    # 公式A4
    C = np.array([[0, 0, 0, 0],
                  [0, 0, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1.0 / np.sqrt(a[0] ** 2 + a[1] ** 2)]
                  ])

    # 公式A5
    D = np.array([[d11, -d21, d13, d32],
                  [d21, d11, -d32, d13],
                  [d31, d32, 0, 0],
                  [d41, d42, 0, 0]])

    # 公式A7
    E = np.array([[dT, 0, e13, e14],
                  [0, dT, e23, e24],
                  [1, 0, e33, e34],
                  [0, 1, e43, e44]])

    phi_mp = C + D @ E

    return yPred, phi_mp


def mpc2polar(Y):
    '''Returns array of [Bearing, Bearing Rate, Range, Range Rate] to target'''
    return np.array([np.rad2deg(Y[2]),
                     np.rad2deg(Y[0]),
                     # Y[3],
                     1 / Y[3],
                     # Y[2]
                     Y[2] / Y[3]
                     ])


def convert_hist_polar(hist, ownshipHist):
    '''Returns array of vectors [Bearing, Bearing Rate, Range, Range Rate] to target'''
    if len(hist) != len(ownshipHist):
        print("Lengths must match")
        return 0
    xRel = hist - ownshipHist
    Y = np.zeros_like(xRel)
    ii = 0
    for val in xRel:
        Y[ii] = mpc2polar(xy2mpc(val))
        ii += 1

    return Y


def build_plots(contactHist, targetHist, ownshipHist):
    '''build and display some error checking plots at the end'''
    contactRelHist = convert_hist_polar(contactHist, ownshipHist)
    fig, ax = plt.subplots(2, 2)
    plt.tight_layout()
    ax[0, 0].plot(contactRelHist[:, 0], '.')
    ax[0, 1].plot(contactRelHist[:, 1], '.')
    ax[1, 0].plot(contactRelHist[:, 2], '.')
    ax[1, 1].plot(contactRelHist[:, 3], '.')

    targetRelHist = convert_hist_polar(targetHist, ownshipHist)
    ax[0, 0].plot(targetRelHist[:, 0], '-')
    ax[0, 1].plot(targetRelHist[:, 1], '-')
    ax[1, 0].plot(targetRelHist[:, 2], '-')
    ax[1, 1].plot(targetRelHist[:, 3], '-')

    # [Bearing, Bearing Rate, Range, Range Rate] to target
    ax[0, 0].set_title("Bearing")

    ax[0, 1].set_title("Bearing Rate")

    ax[1, 0].set_title("Range")

    ax[1, 1].set_title("Range Rate")
    '''
    ax[0,0].set_ylim((-180,180))
    ax[0,1].set_ylim((-180,180))
    ax[1,0].set_ylim(0, max(targetRelHist[:,2]))
    ax[1,1].set_ylim(0, max(targetRelHist[:,3]))
    '''


runTime = 300
dT = 3 # seconds

def main():
    # Generate ownship and target ship
    ownship = Ship(np.array([0, 0, _OBSERVER_SPD, 0]))
    target = Ship(np.array([_TGT_DST * np.sin(np.deg2rad(_TGT_CRS)),
                            _TGT_DST * np.cos(np.deg2rad(_TGT_CRS)),
                            _TGT_SPD * np.sin(np.deg2rad(_TGT_CRS)),
                            _TGT_SPD * np.sin(np.deg2rad(_TGT_CRS))]))


    ekf = CEKF(sonar_bearing(ownship, target))
    mgekf = MGEKF(sonar_bearing(ownship, target))
    plekf = PLEKF(sonar_bearing(ownship, target))
    mpcekf = MPCEKF(sonar_bearing(ownship, target))

    # Record history
    ownshipHist = ownship.X;
    targetHist = target.X;

    ekf_Xest_record = ekf.Xest
    ekf_result_record = ownship.X + ekf.Xest
    mgekf_result_record = ownship.X + mgekf.Xest
    plekf_result_record = ownship.X + plekf.Xest
    mpcekf_result_record = ownship.X + mpc2xy(mpcekf.Yest)

    time = 0
    times = np.array([time])

    while time < runTime:
        # move ships


        ii = int(time / (10 * dT))
        """
        new_course_x = _OBSERVER_SPD if ((-1) ** ii) > 0 else 0
        new_course_y = 0 if ((-1) ** ii) > 0 else _OBSERVER_SPD
        """
        if 0 <= time < 10 * dT:
            new_course_x = _OBSERVER_SPD
            new_course_y = 0
        elif 10 * dT <= time < 30 * dT:
            new_course_x = 0
            new_course_y = _OBSERVER_SPD
        elif 30 * dT <= time < 50 * dT:
            new_course_x = _OBSERVER_SPD
            new_course_y = 0
        elif 50 * dT <= time < 70 * dT:
            new_course_x = 0
            new_course_y = _OBSERVER_SPD
        elif 70 * dT <= time < 90 * dT:
            new_course_x = _OBSERVER_SPD
            new_course_y = 0
        else:
            new_course_x = 0
            new_course_y = _OBSERVER_SPD

        Xo, U = ownship.update(dT, newCourse=[new_course_x, new_course_y])

        Xt, _ = target.update(dT)

        # Record history
        ownshipHist = np.vstack((ownshipHist, ownship.X))

        # update contact
        bearing = sonar_bearing(ownship, target)
        ekf.update(bearing=bearing, Xo=ownshipHist, dT=dT)
        mgekf.update(bearing=bearing, Xo=ownshipHist, dT=dT)
        #if time > 30 * dT:
        #    plekf.update(bearing=bearing, Xo=ownshipHist, dT=dT)
        mpcekf.update(bearing=bearing, Xo=ownshipHist, dT=dT)



        targetHist = np.vstack((targetHist, target.X))
        ekf_Xest_record = np.vstack((ekf_Xest_record, ekf.Xest))
        ekf_result_record = np.vstack((ekf_result_record, ekf.Xest + ownship.X))
        plekf_result_record = np.vstack((plekf_result_record, plekf.Xest + ownship.X))
        mgekf_result_record = np.vstack((mgekf_result_record, mgekf.Xest + ownship.X))
        mpcekf_result_record = np.vstack((mpcekf_result_record, mpc2xy(mpcekf.Yest) + ownship.X))

        time += dT
        times = np.vstack((times, time))

        # Plot progress
        plt.cla()
        plt.plot(ownshipHist[:, 0], ownshipHist[:, 1], 'b-', label="OwnShip")
        plt.plot(targetHist[:, 0], targetHist[:, 1], 'r-', label="Target")
        plt.plot(ekf_result_record[:, 0], ekf_result_record[:, 1], 'b.', label="ekf")
        plt.plot(mgekf_result_record[:, 0], ekf_result_record[:, 1], 'r.', label="mgekf")
        #plt.plot(plekf_result_record[:, 0], plekf_result_record[:, 1], 'b.', label="plekf")
        #plt.plot(mpcekf_result_record[:, 0], ekf_result_record[:, 1], 'r.', label="mpcekf")
        plt.axis("equal")
        # plt.ylim((-10,500))
        # plt.xlim((-10,500))
        # plt.axis((-10,5000,-10,5000))
        plt.title("COP")
        plt.legend()
        plt.pause(.01 * dT)

    build_plots(ekf_result_record, targetHist, ownshipHist)
    build_plots(mgekf_result_record, targetHist, ownshipHist)
    build_plots(mpcekf_result_record, targetHist, ownshipHist)

    # 评价距离估计性能
    dst_ekf = np.sqrt(ekf_Xest_record[:, 0] ** 2 + ekf_Xest_record[:, 1] ** 2)
    dst_real = np.sqrt((targetHist[:, 0] - ownshipHist[:, 0]) ** 2 + (targetHist[:, 1] - ownshipHist[:, 1]) ** 2)
    dst_err = dst_ekf - dst_real
    plt.figure(figsize=(10, 6))
    plt.title("dst tracking performance")
    # 计算距离：
    plt.subplot(1, 2, 1)
    plt.title("ekf dst estimate and true dst")
    plt.plot(times, dst_ekf, label="ekf dst")
    plt.plot(times, dst_real, label="real dst")
    plt.xlabel('Time (s)')
    plt.ylabel('dst (meter)')
    plt.subplot(1, 2, 2)
    plt.title("dst error")
    plt.plot(times, dst_err, label="dst err")
    plt.xlabel('Time (s)')
    plt.ylabel('dst (meter)')
    plt.title("dst tracking performance")
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.title("crs tracking performance")
    crs_ekf = np.rad2deg(np.arctan2(ekf_result_record[:, 2], ekf_result_record[:, 3]))
    plt.subplot(1, 2, 1)
    plt.title("ekf crs estimate and true spd")
    plt.xlabel('Time (s)')
    plt.ylabel('crs (deg)')
    plt.plot(times, crs_ekf, label="ekf crs")
    plt.plot(times, np.ones_like(crs_ekf) * _TGT_CRS, label="real crs")
    plt.subplot(1, 2, 2)
    plt.title("ekf crs err")
    plt.xlabel('Time (s)')
    plt.ylabel('crs err(deg)')
    plt.plot(times, crs_ekf - np.ones_like(crs_ekf) * _TGT_CRS, label="ekf crs err")

    plt.show()

    plt.figure(figsize=(10, 6))
    spd_ekf = np.sqrt(ekf_result_record[:, 2] ** 2 + ekf_result_record[:, 3] ** 2)
    plt.title("spd tracking performance")
    plt.subplot(1,2,1)
    plt.title("ekf spd estimate and true spd")
    plt.plot(times, spd_ekf, label="ekf spd")
    plt.plot(times, np.ones_like(spd_ekf) * _TGT_SPD, label="real spd")
    plt.xlabel('Time (s)')
    plt.ylabel('spd (meters per second)')
    plt.subplot(1,2,2)
    plt.title("ekf spd err")
    plt.xlabel('Time (s)')
    plt.ylabel('spd err(meters per second)')
    plt.plot(times, spd_ekf - np.ones_like(spd_ekf) * _TGT_SPD, label="ekf spd err")
    plt.show()

    plt.figure(figsize=(10, 6))
    brg_ekf = np.rad2deg(np.arctan2(ekf_Xest_record[:, 0], ekf_Xest_record[:, 1]))
    brg_real = np.rad2deg(np.arctan2((targetHist[:, 0] - ownshipHist[:, 0]), (targetHist[:, 1] - ownshipHist[:, 1])))
    plt.plot(times, brg_ekf, label="ekf brg")
    plt.plot(times, brg_real, label="obs brg")
    plt.title("brg tracking performance")
    plt.show()
    return


if __name__ == '__main__':
    for _ in range(1):
        print("starting SIM")
        main()
        print()
        print("Complete")
        print()
        plt.show()