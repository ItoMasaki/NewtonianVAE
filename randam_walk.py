import numpy as np
import matplotlib.pyplot as plt

NUMBER_OF_POINTS = 100000

# 3次元 幾何ランダム・ウォーク
def geometric_random_walk3D():
    # 乱数発生
    dnx = np.random.choice([-1,0,1], size=NUMBER_OF_POINTS)
    dny = np.random.choice([-1,0,1], size=NUMBER_OF_POINTS)
    dnz = np.random.choice([-1,0,1], size=NUMBER_OF_POINTS)
    init_val = 100

    #幾何ランダムウォーク
    ratio = 0.01
    swalkx = np.cumsum(dnx)+init_val    # 累積和
    swalky = np.cumsum(dny)+init_val    # 累積和
    gwalkz = np.cumprod(np.exp(dnz * ratio))*init_val   # 累積積

    # 表示
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title('Geometric random walk 3D')
    ax.plot(swalkx, swalky, gwalkz, color='#FF33FF')
    plt.show()

geometric_random_walk3D()