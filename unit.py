import numpy as np


def set_position2meter(object_pos, pos_unit):
    """
    将不同单位的对象位置转换为米

    :param object_pos: 对象位置坐标
    :param pos_unit: 位置单位 ('cab', 'nmile', 'meter'等)
    :return: 以米为单位的对象位置坐标
    """
    # 创建转换系数字典（单位到米的转换）
    conversion_factors = {
        'cab': 185.2,
        'nmile': 1852,
        'meter': 1
    }

    if pos_unit in conversion_factors:
        factor = conversion_factors[pos_unit]
        # 如果不是meter单位，应用转换系数
        if pos_unit != 'meter':
            object_pos = [object_pos[0] * factor, object_pos[1] * factor]
        return object_pos
    else:
        # 生成所有支持单位的列表
        supported_units = ", ".join(conversion_factors.keys())
        raise ValueError(f"Unsupported distance unit! Current support: {supported_units}")


def set_distance2meter(dst, unit):
    """


    :param dst:
    :param unit:
    :return:
    """
    # 创建转换系数字典（单位到米的转换）
    conversion_factors = {
        'cab': 185.2,
        'nmile': 1852,
        'meter': 1
    }

    if unit in conversion_factors:
        factor = conversion_factors[unit]
        if unit != 'meter':
            dst = dst * factor
        return dst
    else:
        supported_units = ", ".join(conversion_factors.keys())
        raise ValueError(f"Unsupported distance unit! Current support: {supported_units}")

dst = set_distance2meter(100, 'yard')
print(dst)