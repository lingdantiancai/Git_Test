import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

fig, ax = plt.subplots()
fig.subplots_adjust(bottom=0.22)

x = [0, 1, 2, 3, 4]
y = [0, 5, 1, 3, 2]
ax.plot(x, y)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("折线图")

y_lo, y_hi = min(y), max(y)
pad = (y_hi - y_lo) * 0.1 or 1.0
init_lo, init_hi = y_lo - pad, y_hi + pad

ax_min = fig.add_axes([0.15, 0.12, 0.7, 0.03])
ax_max = fig.add_axes([0.15, 0.07, 0.7, 0.03])
slider_lo = Slider(ax_min, "Y 下限", -10, 10, valinit=init_lo, valstep=0.1)
slider_hi = Slider(ax_max, "Y 上限", -10, 10, valinit=init_hi, valstep=0.1)


def on_slider(_):
    lo, hi = sorted((slider_lo.val, slider_hi.val))
    if hi - lo < 1e-6:
        hi = lo + 0.01
    ax.set_ylim(lo, hi)
    fig.canvas.draw_idle()


slider_lo.on_changed(on_slider)
slider_hi.on_changed(on_slider)
ax.set_ylim(init_lo, init_hi)

plt.show()
