from runner import *

plt.rcParams['font.sans-serif'] = ['SimSun']  # 使用宋体字体
plt.rcParams['axes.unicode_minus'] = False  # 显示负号

def main():

    dt = 1
    maxt = 2000
    noise_mean = 0
    noise_std = 0.2
    Sensor = Point2D(np.array([0, 0, 45, 2]), mode='bdcv', acceleration=1, omega=1, maneuver='o')
    Target = Point2D(np.array([5000, 5000, 0, -5]), mode='xyvv', acceleration=1, omega=1)
    model = Model(Sensor, Target, dt=dt, maxt=maxt, brg_noise_mean=noise_mean, brg_noise_std=noise_std)

    algorithms = Algorithms(model)

    number = 20
    runner = Runner(algorithms)

    # 生成固定的蒙特卡洛仿真数据
    runner.generate_monte_carlo_data(number)

    methods = [
                'plkf',
                'ekf',
                #'ukf',
                #'ckf',
                #'frckf',
                #'frfrckf',
                ]
    for method in methods:
        runner.select_method(method)
        runner.run_monte_carlo()

    runner.visualize()

if __name__ == '__main__':
    main()