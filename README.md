1.使用拓展卡尔曼滤波器（EKF）伪线性卡尔曼滤波器（PLKF）无味卡尔曼滤波器（UKF）和容积卡尔曼滤波器（PLKF）进行对匀速直线运动目标的纯方位目标运动分析（BOTMA），绘制蒙特卡洛试验的轨迹和状态变量的RMSE，CRLB（但不是很严谨）

2.基于正逆向滤波思想呈现了正逆向滤波和长短结合正逆向滤波，提高了四个滤波器的跟踪精度，参考为链接论文的第四章https://kns.cnki.net/kcms2/article/abstract?v=AA8hwJ51-CQYrsI92xyDS5jsUXptwdqKJ771bJy7N1fmiWz46EwMYyhoVLBhKBsMRMxbKAuwLcQFKlvmrTZomFsCezr6ktSE9KHNoipzybnn0RZEKNTwxIPoIt041pwsb7oDaKbho4la7eKm9P6jHtpC8rVgp5eMKXrZNQCBoCU=&uniplatform=NZKPT

3.整合了一种基于列文伯格-马夸尔特算法的纯方位极大似然估计（MLE）方法，源代码仓库见https://github.com/orl1k/mle-in-bearing-only-estimation

已知问题：

1.在生成观测数据和更新迭代的时候没有添加过程噪声，因此在卡尔曼滤波器P矩阵迭代更新的过程中也没有加上Q阵，如果强行添加会导致滤波结果变差，原理未知

2.噪声方差矩阵R阵取弧度和角度会对EKF/CKF/UKF的解算性能造成影响，迭代时R取弧度的滤波器性能要显著好于R取角度的性能；对于PLKF，在计算伪线性方差的时候R取弧度则无法正常跟踪，解算结果发散，R取角度的时候PLKF能正常跟踪，但性能弱于EKF/UKF/CKF，该问题的原理未知

3.在正逆向滤波和长短正逆向滤波优化过程中，噪声方差矩阵R阵取弧度和角度会长短正逆向滤波的精度提升幅度造成影响，迭代时R取弧度的滤波器，经过正逆向滤波后解算精度提升幅度明显高于迭代时R取角度的滤波器，原理未知

4.尝试整合最小二乘法（LSTSQ），也是上述链接仓库里提到的“n-bearings”方法，但未成功，我认为代码的表达式没有问题，但是分析不出问题所在。

如果您能发现代码的问题，请不吝赐教！
