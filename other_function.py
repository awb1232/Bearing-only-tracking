import numpy as np


def rad1addrad2(rad1, rad2):
    """
    对两个弧度角求和，返回0-2pi范围的和弧度

    :param rad1:
    :param rad2:
    :return: rad1 + rad2
    """
    add = rad1 + rad2
    add = add % (2 * np.pi)
    return add


def rad1subrad2(rad1, rad2):
    """

    :param rad1:
    :param rad2:
    :return: rad1 - rad2
    """
    sub = rad1 - rad2
    sub = sub % (2 * np.pi)
    return sub


def rad1subrad2a(rad1, rad2):
    """

    :param rad1:
    :param rad2:
    :return: rad1 - rad2
    """
    sub = rad1 - rad2
    sub = (sub + np.pi) % (2 * np.pi) - np.pi
    return sub


def deg1adddeg2(deg1, deg2):
    """
    对两个角度求和，返回0-360范围的和角度

    :param deg1:
    :param deg2:
    :return:
    """

    add = deg1 + deg2
    add = add % 360
    return add


def deg1subdeg2(deg1, deg2):
    """
    对两个角度求差，返回0-360范围的差角度

    :param deg1:
    :param deg2:
    :return: deg1 - deg2
    """
    sub = deg1 - deg2
    sub = sub % 360
    return sub


def deg1subdeg2a(deg1, deg2):
    """
   对两个角度求差，返回-180-180范围的差角度

   :param deg1:
   :param deg2:
   :return: deg1 - deg2
   """
    sub = deg1 - deg2
    sub = (sub + 180) % 360 - 180
    return sub