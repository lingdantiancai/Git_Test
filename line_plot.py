import matplotlib.pyplot as plt

# 手动改 y 轴范围；任一端可为 None（该端仍由数据自动留白）
Y_MIN, Y_MAX = -0.5, 6

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

plt.plot([0, 1, 2, 3, 4], [0, 5, 1, 3, 2])
plt.xlabel("x")
plt.ylabel("y")
plt.title("折线图")
if Y_MIN is not None or Y_MAX is not None:
    plt.ylim(Y_MIN, Y_MAX)
plt.tight_layout()
plt.show()