import numpy as np

def deg1deg2add(deg1, deg2):
    """
    求角度和，返回0-360度的和角

    :param deg1:
    :param deg2:
    :return:
    """
    return (deg1 + deg2) % 360

def deg1deg2sub(deg1, deg2):
    """
    求角度差，返回0-360度的差角

    :param deg1:
    :param deg2:
    :return:
    """
    return (deg1 - deg2) % 360

def deg1deg2sub1(deg1, deg2):
    """
    求角度差，返回-180~+180范围的差角

    :param deg1:
    :param deg2:
    :return:
    """

    return (deg1 - deg2 + 180) % 360 - 180

def rad1rad2add(rad1, rad2):

    return (rad1 + rad2) % (2 * np.pi)

def rad1rad2sub(rad1, rad2):

    return (rad1 - rad2) % (2 * np.pi)

def rad1rad2sub1(rad1, rad2):

    return (rad1 - rad2 + np.pi) % (2 * np.pi) - np.pi