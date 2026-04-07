import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# 手动改 y 轴范围；任一端可为 None（该端仍由数据自动留白）
Y_MIN, Y_MAX = -0.5, 6

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

fig, ax = plt.subplots()
# 四条滑块占底部约 0.28，主图从下沿 0.30 开始，避免滑块叠在主图里点不到
fig.subplots_adjust(bottom=0.30)

x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
y = [0, 5, 1, 3, 2, 6, 4, 7, 3, 8, 2, 5, 9, 1, 6, 4, 7, 3, 8, 5, 2]
ax.plot(x, y)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("折线图")

y_lo, y_hi = min(y), max(y)
pad = (y_hi - y_lo) * 0.1 or 1.0
auto_lo, auto_hi = y_lo - pad, y_hi + pad
init_lo = Y_MIN if Y_MIN is not None else auto_lo
init_hi = Y_MAX if Y_MAX is not None else auto_hi

x_lo, x_hi = min(x), max(x)
xpad = (x_hi - x_lo) * 0.1 or 1.0
x_init_lo, x_init_hi = x_lo - xpad, x_hi + xpad

# 自下而上：Y 上限、Y 下限、x 下限、x 上限（均在主图下方）
ax_y_max = fig.add_axes([0.15, 0.07, 0.7, 0.03])
ax_y_min = fig.add_axes([0.15, 0.12, 0.7, 0.03])
ax_x_min = fig.add_axes([0.15, 0.17, 0.7, 0.03])
ax_x_max = fig.add_axes([0.15, 0.22, 0.7, 0.03])

y_slider_lo = Slider(ax_y_min, "Y 下限", -10, 10, valinit=init_lo, valstep=0.1)
y_slider_hi = Slider(ax_y_max, "Y 上限", -10, 10, valinit=init_hi, valstep=0.1)
x_slider_lo = Slider(ax_x_min, "x 下限", -50, 50, valinit=x_init_lo, valstep=0.1)
x_slider_hi = Slider(ax_x_max, "x 上限", -50, 50, valinit=x_init_hi, valstep=0.1)


def on_y_slider(_):
    lo, hi = sorted((y_slider_lo.val, y_slider_hi.val))
    if hi - lo < 1e-6:
        hi = lo + 0.01
    ax.set_ylim(lo, hi)
    fig.canvas.draw_idle()


def on_x_slider(_):
    xlo, xhi = sorted((x_slider_lo.val, x_slider_hi.val))
    if xhi - xlo < 1e-6:
        xhi = xlo + 0.01
    ax.set_xlim(xlo, xhi)
    fig.canvas.draw_idle()


y_slider_lo.on_changed(on_y_slider)
y_slider_hi.on_changed(on_y_slider)
x_slider_lo.on_changed(on_x_slider)
x_slider_hi.on_changed(on_x_slider)
ax.set_ylim(init_lo, init_hi)
ax.set_xlim(x_init_lo, x_init_hi)

plt.show()
