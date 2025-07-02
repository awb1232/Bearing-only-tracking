from runner import *

plt.rcParams['font.sans-serif'] = ['SimSun']  # 使用宋体字体
plt.rcParams['axes.unicode_minus'] = False  # 显示负号

def main():

    dt = 1
    maxt = 2000
    noise_mean = 0
    noise_std = 0.2
    Sensor = Point2D(np.array([0, 0, 90, 2]), mode='bdcv', acceleration=1, omega=1, maneuver='s')
    Target = Point2D(np.array([5000, 5000, 0, -5]), mode='xyvv', acceleration=1, omega=1)
    model = Model(Sensor, Target, dt=dt, maxt=maxt, brg_noise_mean=noise_mean, brg_noise_std=noise_std)

    algorithms = Algorithms(model)

    number = 10
    runner = Runner(algorithms)

    # 生成固定的蒙特卡洛仿真数据
    runner.generate_monte_carlo_data(number)
    x0 = np.array([5000.0, 5000.0, 0.0, -5.0])
    P0 = np.diag([100.0 ** 2, 100.0 ** 2, 1.0 ** 2, 50.0 ** 2])

    methods = {
        # 'ekf':'palegreen',
        'plkf': 'grey',
        # 'ukf': 'violet',
        # 'ckf': 'cyan',
        #
        # 'frekf': 'forestgreen',
        # 'frukf': 'darkviolet',
        # 'frckf': 'c',
        #
        # 'lsfrekf':'green',
        # 'lsfrukf': 'purple',
        # 'lsfrckf': 'darkcyan',

        # 'mle': 'orange',
        # 'lstsq': 'blue',
        # 'lstsq1': 'red',

    }

    for method, color in methods.items():
        runner.select_method(method)
        runner.run_monte_carlo(x0, P0, reverse_step=600, partical_rev_step=20, color=color)

    runner.visualize(x_ylim=1500,
                     y_ylim=1500,
                     vx_ylim=5,
                     vy_ylim=5,
                     pos_ylim=2000,
                     crs_ylim=10,
                     spd_ylim=5,
                     crlb_analysis=True,
                     subplot=False)

if __name__ == '__main__':
    main()