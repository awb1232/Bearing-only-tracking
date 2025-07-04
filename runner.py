import matplotlib.pyplot as plt
import time

import numpy as np
from tqdm import tqdm
from algorithms import *


class Runner:
    def __init__(self, algorithms):
        self.model = algorithms.model

        self.sensor_maneuver_types = {
            '8': self.model.generate_sensor_trajectory_8,
            'o': self.model.generate_sensor_trajectory_circle,
            's': self.model.generate_sensor_trajectory_s,

            'z': self.model.generate_sensor_trajectory_z,
        }

        self.method_name = None
        self.method_map = {
            "ekf": algorithms.ekf,
            "plkf": algorithms.plkf,
            "ukf": algorithms.ukf,
            "ckf": algorithms.ckf,
            "frekf": algorithms.frkf,
            "frplkf": algorithms.frkf,
            "frukf": algorithms.frkf,
            "frckf": algorithms.frkf,
            'lsfrekf': algorithms.lsfrkf,
            'lsfrplkf': algorithms.lsfrkf,
            'lsfrukf': algorithms.lsfrkf,
            'lsfrckf': algorithms.lsfrkf,
            'mle': algorithms.mle,
            'lstsq': algorithms.lstsq,
            'lstsq1': algorithms.lstsq1,
        }

        self.result = []

        self.measurements_generated = False  # 标记是否已生成测量数据
        self.mc_iterations = 0  # 蒙特卡洛迭代次数
        self.all_measurements = []  # 存储所有迭代的测量值

    def select_maneuver_type(self, type):
        self.maneuver_type = type
        # 检查方法是否有效
        if type not in self.sensor_maneuver_types:
            # 提取所有支持的算法名称，用逗号分隔
            supported_methods = ", ".join(sorted(self.sensor_maneuver_types.keys()))
            raise ValueError(
                f"Unknown maneuver: '{type}'. Supported maneuver types are: {supported_methods}"
            )

    def generate_monte_carlo_data(self, num):
        """
        为蒙特卡洛仿真生成固定的测量数据集

        :param num: 蒙特卡洛仿真次数
        """
        if self.measurements_generated:
            return  # 如果已经生成数据，直接返回

        # 生成轨迹（仅需一次）
        selected_maneuver = self.sensor_maneuver_types[self.model.Sensor.maneuver]

        selected_maneuver()
        self.model.generate_target_trajectory()
        self.model.generate_bearings()
        self.model.generate_crlb()

        print(f'正在生成{num}次蒙特卡洛仿真的测量数据...')

        # 保存所有迭代的测量值
        self.all_measurements = []
        for i in tqdm(range(num),
                      desc="数据生成进度",
                      unit="次",
                      bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{remaining}]"):
            # 生成新的含噪方位测量
            self.model.generate_measurements()
            # 保存本次迭代的测量值
            self.all_measurements.append(self.model.measurements.copy())

        self.measurements_generated = True
        self.mc_iterations = num
        print(f'测量数据生成完毕')

    def select_method(self, method):
        self.method_name = method
        # 检查方法是否有效
        if method not in self.method_map:
            # 提取所有支持的算法名称，用逗号分隔
            supported_methods = ", ".join(sorted(self.method_map.keys()))
            raise ValueError(
                f"Unknown method: '{method}'. Supported methods are: {supported_methods}"
            )

    def run_monte_carlo(self, x0, p0, num=None, **kwargs):
        """
        使用已经生成的数据集运行蒙特卡洛仿真

        :param num: 可选，如果提供则重新生成数据集
        """
        if num is not None and (not self.measurements_generated or num != self.mc_iterations):
            self.measurements_generated = False
            self.generate_monte_carlo_data(num)
        elif not self.measurements_generated:
            raise ValueError("请先调用 generate_monte_carlo_data 生成数据集")

        if "reverse_step" in kwargs:
            reverse_step = kwargs["reverse_step"]
        else:
            reverse_step = 600

        if "partical_rev_step" in kwargs:
            partical_rev_step = kwargs["partical_rev_step"]
        else:
            partical_rev_step = 20

        # 获取对应的函数
        target_method = self.method_map[self.method_name]

        # 初始状态设置
        init_Xest = x0
        init_Pest = p0
        Q = self.model.Q

        estimation_all = []
        square_error_all = []

        print(f'对{self.method_name}方法进行{self.mc_iterations}次蒙特卡洛仿真')
        start_time = time.time()

        for i in tqdm(range(self.mc_iterations),
                      desc="仿真进度",
                      unit="次",
                      bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{remaining}]"):

            # 使用保存的测量值
            self.model.measurements = self.all_measurements[i].copy()

            if self.method_name == 'ekf' or self.method_name == 'ukf' or self.method_name == 'ckf' :
                result = target_method(init_Xest, init_Pest, Q)

            elif self.method_name == 'frekf' or self.method_name == 'frukf' or self.method_name == 'frckf':
                keyword='fr'
                fliter_name = self.method_name.split(keyword, 1)[1]
                result = target_method(init_Xest, init_Pest, Q, fliter_name, reverse_step)

            elif self.method_name == 'lsfrekf' or self.method_name == 'lsfrukf' or self.method_name == 'lsfrckf':
                keyword = 'lsfr'
                fliter_name = self.method_name.split(keyword, 1)[1]
                result = target_method(init_Xest, init_Pest, Q, fliter_name, reverse_step, partical_rev_step)

            elif self.method_name == 'mle':
                result = target_method(init_Xest, init_Pest, Q, 'ckf', reverse_step, partical_rev_step)

            elif self.method_name == 'lstsq' or self.method_name == 'lstsq1':
                result = target_method()

            else:
                result = target_method(init_Xest, init_Pest, Q)

            estimation_all.append(result['states'])
            square_error_all.append(result['square_error'])

        time1 = time.time() - start_time
        print(f'仿真结束，用时{time1:.2f}')

        avg_estimation = np.mean(estimation_all, axis=0)

        avg_rse = np.sqrt((self.model.target_states - avg_estimation) ** 2)

        avg_pos_rmse = np.sqrt(np.sum((self.model.target_states[:, 0:2] - avg_estimation[:, 0:2]) ** 2, axis=1))
        avg_vel_rmse = np.sqrt(np.sum((self.model.target_states[:, 2:4] - avg_estimation[:, 2:4]) ** 2, axis=1))

        # 计算真实航向角
        true_crs = np.arctan2(self.model.target_states[:, 2], self.model.target_states[:, 3])  # arctan2(vx, vy)

        # 计算估计航向角
        avg_estimated_crs = np.arctan2(avg_estimation[:, 2], avg_estimation[:, 3])  # arctan2(vx, vy)

        # 计算航向误差（考虑角度的周期性）
        # 这种方法可以正确处理如359°和1°之间的差异（应该是2°而非358°）
        avg_crs_rmse = np.arctan2(np.sin(true_crs - avg_estimated_crs),
                                  np.cos(true_crs - avg_estimated_crs))
        avg_crs_rmse = np.rad2deg(avg_crs_rmse)

        rmse = np.sqrt(np.mean(square_error_all, axis=0))

        pos_rmse = np.sqrt(np.sum(rmse[:, :2]**2, axis=1))

        crs_error = []
        for arr in estimation_all:
            vx = arr[:, 2]
            vy = arr[:, 3]
            estimated_crs = np.arctan2(vx, vy)
            error = estimated_crs - true_crs
            error = np.arctan2(np.sin(true_crs - estimated_crs),
                               np.cos(true_crs - estimated_crs))

            crs_error.append(error)

        crs_rmse = np.mean(crs_error, axis=0)
        crs_rmse = np.rad2deg(crs_rmse)

        spd_rmse = np.sqrt(np.sum(rmse[:, 2:]**2, axis=1))

        if "color" in kwargs:
            color = kwargs["color"]
        else:
            color = result['color'],

        runner_result = {'name': self.method_name,
                         'num': self.mc_iterations,
                         'time': self.model.times,
                         'sensor_state': self.model.sensor_states,
                         'true_state': self.model.target_states,
                         'estimation': avg_estimation,
                         'avg_x_rmse': avg_rse[:, 0],
                         'avg_y_rmse': avg_rse[:, 1],
                         'avg_vx_rmse': avg_rse[:, 2],
                         'avg_vy_rmse': avg_rse[:, 3],
                         'avg_pos_rmse': avg_pos_rmse,
                         'avg_vel_rmse': avg_crs_rmse,
                         'avg_crs_rmse': avg_vel_rmse,
                         'x_rmse': rmse[:, 0],
                         'y_rmse': rmse[:, 1],
                         'vx_rmse': rmse[:, 2],
                         'vy_rmse': rmse[:, 3],
                         'pos_rmse': pos_rmse,
                         'crs_rmse': crs_rmse,
                         'spd_rmse': spd_rmse,
                         'color': color,
                         'crlb': self.model.crlb,
                         }

        self.result.append(runner_result)

        return avg_pos_rmse, avg_vel_rmse

    def visualize(self, crlb_analysis=True, subplot=False, **kwargs):

        num_of_methods_used = len(self.result)

        if num_of_methods_used < 1:
            raise ValueError('未使用任何方法进行仿真！')

        if "x_ylim" in kwargs:
            x_ylim = kwargs["x_ylim"]

        if "y_ylim" in kwargs:
            y_ylim = kwargs["y_ylim"]

        if "vx_ylim" in kwargs:
            vx_ylim = kwargs["vx_ylim"]

        if "vy_ylim" in kwargs:
            vy_ylim = kwargs["vy_ylim"]

        if "pos_ylim" in kwargs:
            pos_ylim = kwargs["pos_ylim"]

        if "crs_ylim" in kwargs:
            crs_ylim = kwargs["crs_ylim"]

        if "spd_ylim" in kwargs:
            spd_ylim = kwargs["spd_ylim"]

        # 绘制完整的真实轨迹和观测者轨迹
        true_states = self.result[0]['true_state']
        sensor_states = self.result[0]['sensor_state']
        num = self.result[0]['num']

        # 创建静态图
        plt.figure()

        # 绘制真实轨迹和估计轨迹

        plt.plot(true_states[:, 0], true_states[:, 1], 'red', label='真实轨迹')
        plt.plot(sensor_states[:, 0], sensor_states[:, 1], 'blue', label='传感器轨迹')

        for i in range(num_of_methods_used):
            estimation = self.result[i]['estimation']
            name = self.result[i]['name']
            color = self.result[i]['color']

            if name != 'mle' and name != 'lstsq':
                plt.plot(estimation[:, 0], estimation[:, 1], color=color, label=f"{name}算法{num}次平均估计轨迹")

        plt.axis('equal')
        plt.grid(True)
        plt.xlabel('X 位置 (m)')
        plt.ylabel('Y 位置 (m)')
        plt.title(f'{num}次Monte Carlo仿真的平均轨迹')
        plt.legend()
        plt.show()

        # 绘制XRMSE
        crlb = self.result[0]['crlb']
        crlb_x = np.sqrt(crlb[:, 0])
        crlb_y = np.sqrt(crlb[:, 1])
        crlb_vx = np.sqrt(crlb[:, 2])
        crlb_vy = np.sqrt(crlb[:, 3])

        plt.figure()
        times_range = self.result[0]['time']
        if subplot:
            plt.subplot(1, 2, 1)
        for i in range(num_of_methods_used):
            name = self.result[i]['name']
            color = self.result[i]['color']
            rmse = self.result[i]['x_rmse']
            if name != 'mle' and name != 'lstsq':
                plt.plot(times_range, rmse, color=color, label=f"{name}算法{num}次仿真 x RMSE")
            else:
                mle_estimation_x_init = self.result[i]['estimation'][:, 0]
                true_x_init = true_states[0][0]
                plt.plot(times_range, mle_estimation_x_init - true_x_init * np.ones_like(mle_estimation_x_init), color=color, label=f"{name}算法{num}次仿真 x RMSE")

        if crlb_analysis:
            plt.plot(times_range, crlb_x, color='black', label='状态估计x CRLB')

        if "x_ylim" in kwargs:
            plt.ylim(0, x_ylim)
        plt.xlabel('时间 (s)')
        plt.ylabel('位置x误差 (m)')
        plt.title('位置x估计RMSE')
        plt.legend()

        if subplot:
            plt.subplot(1, 2, 2)
            for i in range(num_of_methods_used):
                name = self.result[i]['name']
                color = self.result[i]['color']
                avg_rse = self.result[i]['avg_x_rmse']
                plt.plot(times_range, avg_rse, color=color, label=f"{name}算法{num}次仿真平均位置x误差")

            if crlb_analysis:
                plt.plot(times_range, crlb_x, color='black', label='状态估计x CRLB')

            if "x_ylim" in kwargs:
                plt.ylim(0, x_ylim)
            plt.xlabel('时间 (s)')
            plt.ylabel('位置x误差 (m)')
            plt.title('位置x平均估计误差')
            plt.legend()

        plt.show()

        # Y
        plt.figure()
        times_range = self.result[0]['time']
        if subplot:
            plt.subplot(1, 2, 1)
        for i in range(num_of_methods_used):
            name = self.result[i]['name']
            color = self.result[i]['color']
            rmse = self.result[i]['y_rmse']
            if name != 'mle' and name != 'lstsq':
                plt.plot(times_range, rmse, color=color, label=f"{name}算法{num}次仿真 y RMSE")
            else:
                mle_estimation_y_init = self.result[i]['estimation'][:, 1]
                true_y_init = true_states[0][1]
                plt.plot(times_range, mle_estimation_y_init - true_y_init * np.ones_like(mle_estimation_y_init),
                         color=color, label=f"{name}算法{num}次仿真 y RMSE")

        if crlb_analysis:
            plt.plot(times_range, crlb_y, color='black', label='状态估计y CRLB')

        if "y_ylim" in kwargs:
            plt.ylim(0, y_ylim)
        plt.xlabel('时间 (s)')
        plt.ylabel('位置y误差 (m)')
        plt.title('位置y估计RMSE')
        plt.legend()

        if subplot:
            plt.subplot(1, 2, 2)
            for i in range(num_of_methods_used):
                name = self.result[i]['name']
                color = self.result[i]['color']
                avg_rse = self.result[i]['avg_y_rmse']
                plt.plot(times_range, avg_rse, color=color, label=f"{name}算法{num}次仿真平均位置y误差")

            if crlb_analysis:
                plt.plot(times_range, crlb_y, color='black', label='状态估计y CRLB')

            if "y_ylim" in kwargs:
                plt.ylim(0, y_ylim)
            plt.xlabel('时间 (s)')
            plt.ylabel('位置y误差 (m)')
            plt.title('位置y平均估计误差')
            plt.legend()

        plt.show()

        # Vx
        plt.figure()
        times_range = self.result[0]['time']
        if subplot:
            plt.subplot(1, 2, 1)
        for i in range(num_of_methods_used):
            name = self.result[i]['name']
            color = self.result[i]['color']
            rmse = self.result[i]['vx_rmse']
            plt.plot(times_range, rmse, color=color, label=f"{name}算法{num}次仿真 vx RMSE")

        if crlb_analysis:
            plt.plot(times_range, crlb_vx, color='black', label='状态估计vx CRLB')
            plt.ylim(0, vx_ylim)

        if "vx_ylim" in kwargs:
            plt.ylim(0, vx_ylim)

        plt.xlabel('时间 (s)')
        plt.ylabel('速度x误差 (m/s)')
        plt.title('速度x估计RMSE')
        plt.legend()

        if subplot:
            plt.subplot(1, 2, 2)
            for i in range(num_of_methods_used):
                name = self.result[i]['name']
                color = self.result[i]['color']
                avg_rse = self.result[i]['avg_vx_rmse']
                plt.plot(times_range, avg_rse, color=color, label=f"{name}算法{num}次仿真平均速度x误差")

            if crlb_analysis:
                plt.plot(times_range, crlb_vx, color='black', label='状态估计vx CRLB')
                plt.ylim(0, vx_ylim)

            if "vx_ylim" in kwargs:
                plt.ylim(0, vx_ylim)
            plt.xlabel('时间 (s)')
            plt.ylabel('速度x误差 (m/s)')
            plt.title('速度x平均估计误差')
            plt.legend()

        plt.show()

        # Vy
        plt.figure()
        times_range = self.result[0]['time']
        if subplot:
            plt.subplot(1, 2, 1)
        for i in range(num_of_methods_used):
            name = self.result[i]['name']
            color = self.result[i]['color']
            rmse = self.result[i]['vy_rmse']
            plt.plot(times_range, rmse, color=color, label=f"{name}算法{num}次仿真 vy RMSE")

        if crlb_analysis:
            plt.plot(times_range, crlb_vy, color='black', label='状态估计vy CRLB')
            plt.ylim(0, vy_ylim)

        if "vy_ylim" in kwargs:
            plt.ylim(0, vy_ylim)
        plt.xlabel('时间 (s)')
        plt.ylabel('速度y误差 (m/s)')
        plt.title('速度y估计RMSE')
        plt.legend()

        if subplot:
            plt.subplot(1, 2, 2)
            for i in range(num_of_methods_used):
                name = self.result[i]['name']
                color = self.result[i]['color']
                avg_rse = self.result[i]['avg_vy_rmse']
                plt.plot(times_range, avg_rse, color=color, label=f"{name}算法{num}次仿真平均速度y误差")

            if crlb_analysis:
                plt.plot(times_range, crlb_vy, color='black', label='状态估计vy CRLB')
                plt.ylim(0, vy_ylim)

            if "vy_ylim" in kwargs:
                plt.ylim(0, vy_ylim)
            plt.xlabel('时间 (s)')
            plt.ylabel('速度y误差 (m/s)')
            plt.title('速度y平均估计误差')
            plt.legend()

        plt.show()

        # 绘制位置RMSE
        plt.figure()

        times_range = self.result[0]['time']
        if subplot:
            plt.subplot(1, 2, 1)
        for i in range(num_of_methods_used):
            name = self.result[i]['name']
            color = self.result[i]['color']
            rmse = self.result[i]['pos_rmse']
            if name != 'mle' and name != 'lstsq':
                plt.plot(times_range, rmse, color=color, label=f"{name}算法{num}次仿真 pos RMSE")
            else:
                mle_estimation = self.result[i]['estimation']
                true_state_init = true_states[0]
                true_states_init = np.tile(true_state_init, (len(mle_estimation), 1))
                init_pos_error = np.sum(np.sqrt((mle_estimation[:, :2] - true_states_init[:, :2]) ** 2), axis=1)

                plt.plot(times_range, init_pos_error, color=color, label=f"{name}算法{num}次仿真初始 pos RMSE")

        if "pos_ylim" in kwargs:
            plt.ylim(0, pos_ylim)
        plt.xlabel('时间 (s)')
        plt.ylabel('位置误差 (m)')
        plt.title('位置估计RMSE')
        plt.legend()

        if subplot:
            plt.subplot(1, 2, 2)
            for i in range(num_of_methods_used):
                name = self.result[i]['name']
                color = self.result[i]['color']
                avg_rse = self.result[i]['avg_pos_rmse']
                plt.plot(times_range, avg_rse, color=color, label=f"{name}算法{num}次仿真平均位置误差")
                plt.ylim(0, pos_ylim)

            if "pos_ylim" in kwargs:
                plt.ylim(0, pos_ylim)
            plt.xlabel('时间 (s)')
            plt.ylabel('位置误差 (m)')
            plt.title('位置平均估计误差')
            plt.legend()

        plt.show()

        # 绘制速度RMSE
        plt.figure()
        times_range = self.result[0]['time']
        if subplot:
            plt.subplot(1, 2, 1)
        for i in range(num_of_methods_used):
            name = self.result[i]['name']
            color = self.result[i]['color']
            rmse = self.result[i]['spd_rmse']
            plt.plot(times_range, rmse, color=color, label=f"{name}算法{num}次仿真 spd RMSE")

        if "spd_ylim" in kwargs:
            plt.ylim(0, spd_ylim)
        plt.xlabel('时间 (s)')
        plt.ylabel('速度误差 (m/s)')
        plt.title('速度估计RMSE')
        plt.legend()

        if subplot:
            plt.subplot(1, 2, 2)
            for i in range(num_of_methods_used):
                name = self.result[i]['name']
                color = self.result[i]['color']
                avg_rse = self.result[i]['avg_vel_rmse']
                plt.plot(times_range, avg_rse, color=color, label=f"{name}算法{num}次仿真平均速度误差")

            if "spd_ylim" in kwargs:
                plt.ylim(0, spd_ylim)
            plt.xlabel('时间 (s)')
            plt.ylabel('速度误差 (m/s)')
            plt.title('速度平均估计误差')
            plt.legend()

        plt.show()

        # 绘制航向RMSE
        plt.figure()
        times_range = self.result[0]['time']
        if subplot:
            plt.subplot(1, 2, 1)
        for i in range(num_of_methods_used):
            name = self.result[i]['name']
            color = self.result[i]['color']
            rmse = self.result[i]['crs_rmse']
            plt.plot(times_range, rmse, color=color, label=f"{name}算法{num}次仿真 crs RMSE")

        if "crs_ylim" in kwargs:
            plt.ylim(0, crs_ylim)
        plt.xlabel('时间 (s)')
        plt.ylabel('航向误差 (deg)')
        plt.title('航向估计RMSE')
        plt.legend()

        if subplot:
            plt.subplot(1, 2, 2)
            for i in range(num_of_methods_used):
                name = self.result[i]['name']
                color = self.result[i]['color']
                avg_rse = self.result[i]['avg_crs_rmse']
                plt.plot(times_range, avg_rse, color=color, label=f"{name}算法{num}次仿真平均速度误差")

            if "crs_ylim" in kwargs:
                plt.ylim(0, crs_ylim)
            plt.xlabel('时间 (s)')
            plt.ylabel('航向误差 (deg)')
            plt.title('航向平均估计误差')
            plt.legend()

        plt.show()