import numpy as np

from angel_process import *
from scipy.linalg import cho_factor, cho_solve

def predict_positions(times, sensor_states, estimation):
    """

    :param times:
    :param sensor_states:
    :param estimation:
    :return:
    """
    x, y, vx, vy = estimation

    x_ests = x + vx * times
    y_ests = y + vy * times

    sensor_x = sensor_states[:, 0]
    sensor_y = sensor_states[:, 1]

    rx = x_ests - sensor_x
    ry = y_ests - sensor_y

    return rx, ry

def predict_bearings(times, sensor_states, estimation):
    """
    由当前状态估计estimation，预测每个times时刻下的对应方位角

    :param times:{ndarray:(n,)}
    :param sensor_states: 形状为{ndarray:(n,4)}或{ndarray:(n,2)}
    :param estimation:
    :return: 方位角（弧度）序列({ndarray:(n,)})
    """

    rx, ry = predict_positions(times, sensor_states, estimation)

    bearings_pred = np.arctan2(rx, ry)

    return bearings_pred

def compute_predict_error(bearing_pred, measurements):
    """

    :param bearing_pred:
    :param measurements:
    :return:
    """

    res = rad1rad2sub1(bearing_pred, measurements)

    return np.sum(res ** 2)

def jaccobian(times, sensor_states, estimation):
    """

    :param times:
    :param sensor_states:
    :param estimation:
    :return:
    """

    J = []
    r_x, r_y = predict_positions(times, sensor_states, estimation)
    R2 = r_x ** 2 + r_y ** 2

    J.append(r_y / R2)
    J.append(-r_x / R2)
    J.append((times * r_y) / R2)
    J.append(-(times * r_x) / R2)

    return np.array(J).T


def lev_mar(times, sensor_states, measurements, par,
            lam=1e-2,
            down_factor=0.5,
            up_factor=3,
            max_it=1000,
            ftol=1e-8, ):

    i = 0  # 2025 04 16 迭代次数，最大为 max_it
    nf = 1  #  2025 04 16 迭代次数？
    status = -1

    # 由初始估计预测方位角，计算预测误差
    f_par = predict_bearings(times, sensor_states, par)
    err = compute_predict_error(f_par, measurements)

    while i < max_it:
        J = jaccobian(times, sensor_states, par)

        b = J.T.dot(rad1rad2sub1(measurements, f_par))

        H = J.T.dot(J)

        step = False

        while (not step) and (i < max_it):

            # 计算直到 delta_err < 0
            try:
                A = H + lam * np.diag(np.diag(H))  # Marquardt modification 20250416 列文伯格-马夸尔特最优化算法的改进

                L, low = cho_factor(A)
                delta_par = cho_solve((L, low), b)

                new_par = par + delta_par
                f_par = predict_bearings(times, sensor_states, new_par)

                nf += 1  # 获得一个新的par后nf+1

                new_err = compute_predict_error(f_par, measurements)

                delta_err = err - new_err

                step = delta_err >= 0.0

                if not step:
                    lam *= up_factor

            except np.linalg.LinAlgError:
                lam *= up_factor

        par = new_par
        err = new_err
        i += 1

        lam *= down_factor

        if delta_err < ftol:
            status = 0
            break

    if status == -1:
        status = 1

    #J = jaccobian(times, sensor_states, f_par)
    #H = J.T.dot(J)

    return par

def lstsq1(times, sensor_states, measurements):

    b0 = measurements[0]

    n = len(measurements)

    bearings = measurements
    bearing_origin = np.array([b0] * n)

    H = [-np.sin(bearing_origin - bearings)]
    H.append(np.sin(bearings) * times)
    H.append(-np.cos(bearings) * times)
    H = np.array(H).T

    d = sensor_states[:,0] * np.sin(
            bearings
        ) - sensor_states[:,1] * np.cos(bearings)

    res = np.linalg.solve(H.T.dot(H), H.T.dot(d))
    res = np.insert(res, 0, b0)  # res

    b, d = b0, res[1]
    res[0] = d * np.sin(b)
    res[1] = d * np.cos(b)

    return res

def lstsq(times, sensor_states, measurements):
    """

    :param times:
    :param sensor_states:
    :param measurements:
    :return:
    """

    # A = []
    # B = []
    #
    # for j in range(len(times)):
    #     cos_B = np.cos(measurements[j])
    #     sin_B = np.sin(measurements[j])
    #     delta_t = times[j] - times[0]
    #     pw_x_0, pw_y_0 = sensor_states[0][0], sensor_states[0][1]
    #     pw_x_t, pw_y_t = sensor_states[j][0], sensor_states[j][1]
    #
    #     A.append([cos_B, -sin_B, delta_t * cos_B, -delta_t * sin_B])
    #
    #     B_i = (pw_x_t - pw_x_0) * cos_B - (pw_y_t - pw_y_0) * sin_B
    #     B.append(B_i)
    #
    # A = np.array(A)
    # B = np.array(B)

    t0 = times[0]

    delta_t_i = times - t0 * np.ones_like(times)
    cos_b_i = np.cos(measurements)
    sin_b_i = np.sin(measurements)

    A = [cos_b_i, -sin_b_i, cos_b_i * delta_t_i, -sin_b_i * delta_t_i]
    A = np.array(A).T

    pos_x = sensor_states[:, 0]
    pos_y = sensor_states[:, 1]
    pos_x_0 = pos_x[0]
    pos_y_0 = pos_y[0]
    delta_pos_x_i = pos_x - pos_x_0 * np.ones_like(pos_x)
    delta_pos_y_i = pos_y - pos_y_0 * np.ones_like(pos_y)

    B = delta_pos_x_i * cos_b_i - delta_pos_y_i * sin_b_i
    B = np.array(B).T

    X = np.linalg.lstsq(A, B, rcond=None)[0]

    return X
