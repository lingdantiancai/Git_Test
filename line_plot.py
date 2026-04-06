import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

plt.plot([0, 1, 2, 3, 4], [0, 2, 1, 3, 2])
plt.xlabel("x")
plt.ylabel("y")
plt.title("折线图")
plt.tight_layout()
plt.show()