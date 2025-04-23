import numpy as np
import copy
from bearing_only_track.other_function import *

class ship2d:

    def __init__(self, initial_position, initial_speed, acceleration, initial_course, omega):
        """
        通过2维坐标，航速，航向创建一个二维平面的动点

        :param initial_position:
        :param initial_speed:
        :param acceleration:
        :param initial_course:
        :param omgea
        """
        self.position = np.array(initial_position, dtype=float)
        self.speed = initial_speed
        self.course = initial_course
        self.acceleration = acceleration
        self.omega = omega

        # 记录点的位置速度和航向的变化
        self.trajectory = [self.position.copy()]
        self.speed_history = [self.speed.copy()]
        self.course_history = [deg1adddeg2(self.course.copy, 0)]

    def uniform_linear_motion(self, dt):
        delta_position = self.speed * np.array(
            [np.sin(self.course), np.cos(self.course)]) * dt
        self.position += delta_position
        self.trajectory.append(self.position.copy())
        self.speed_history.append(copy.copy(self.speed))
        self.course_history.append(deg1adddeg2(self.course.copy, 0))

    def accelerated_linear_motion(self, dt):
        """匀加速直线运动"""
        self.speed += self.acceleration * dt
        self.uniform_linear_motion(dt)

    def uniform_circular_motion(self, dt, turn_direction):
        """匀速圆周运动"""
        delta_angular = self.omega * dt
        if turn_direction == 0:
            self.course -= delta_angular
        else:
            self.course += delta_angular
        self.uniform_linear_motion(dt)

    def get_trajectory(self):
        return np.array(self.trajectory)

    def auto_hold_course_and_speed(self, dt, target_course, target_speed):
        # 检查是否需要转向
        delta_crs = deg1subdeg2a(target_course, self.course)
        if abs(deg1subdeg2a(target_course, self.course)) > 1e-3:
            turn_direction = 0 if delta_crs < 0 else 1
            self.uniform_circular_motion(dt, turn_direction)

        # 检查是否需要变速
        elif abs(self.speed - target_speed) > 1e-3:  # 调整航速

            acceleration = self.acceleration if self.speed < target_speed else -self.acceleration
            self.accelerated_linear_motion(dt)
        else:  # 直线航行

            self.uniform_linear_motion(dt)


# 模拟的我艇类
class Point:
    def __init__(self, initial_position, speed, course):
        # 初始化点的位置、速度和航向
        self.position = np.array(initial_position, dtype=float)
        self.speed = speed
        self.course = course

        # 记录点的位置速度和航向的变化
        self.trajectory = [self.position.copy()]
        self.speed_history = [self.speed.copy()]
        self.course_history = [Bear1AddBear2(copy.copy(self.course), 0)]

    def uniform_linear_motion(self, dt):
        """
        执行一次匀速直线运动，记录点的位置、航速、航向

        :param dt: 模拟最小的时间步长
        :return: 无
        """
        delta_position = self.speed * np.array(
            [sind(self.course), cosd(self.course)]) * dt / 360
        self.position += delta_position
        self.trajectory.append(self.position.copy())
        self.speed_history.append(copy.copy(self.speed))
        self.course_history.append(Bear1AddBear2(copy.copy(self.course), 0))

    def accelerated_linear_motion(self, dt, acceleration):
        """匀加速直线运动"""
        self.speed += acceleration * dt
        self.uniform_linear_motion(dt)

    def uniform_circular_motion(self, dt, angular_speed, turn_direction):
        """匀速圆周运动"""
        delta_angular = angular_speed * dt
        if turn_direction == 0:
            self.course -= delta_angular
        else:
            self.course += delta_angular
        self.uniform_linear_motion(dt)

