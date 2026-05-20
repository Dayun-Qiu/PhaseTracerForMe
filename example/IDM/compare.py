import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

data_dir = "/home/dayun/data"
raw_file = os.path.join(data_dir, "M2_1.txt")

# 读取原始数据
raw = np.loadtxt(raw_file, skiprows=1)
phi_raw = raw[:, 0]
T_raw = raw[:, 1]

for k in range(13):
    orig = raw[:, k+2]                     # 第 k 个质量（0‑based）
    interp_file = os.path.join(data_dir, f"interp_{k}.txt")
    if not os.path.exists(interp_file):
        print(f"File {interp_file} not found, skip")
        continue

    # 读取插值结果
    interp_data = np.loadtxt(interp_file, skiprows=1)
    phi_interp = interp_data[:, 0]
    T_interp = interp_data[:, 1]
    interp_val = interp_data[:, 2]

    # 剔除原始数据中的 NaN（坏点）
    mask = ~np.isnan(orig)
    phi_raw_clean = phi_raw[mask]
    T_raw_clean = T_raw[mask]
    orig_clean = orig[mask]

    # 将插值结果重塑为二维网格（假设 phi 和 T 规则且按相同顺序）
    phi_vals = np.unique(phi_interp)
    T_vals = np.unique(T_interp)
    interp_grid = interp_val.reshape(len(phi_vals), len(T_vals))
    Phi, T_grid = np.meshgrid(phi_vals, T_vals, indexing='ij')

    # 创建单个 3D 图
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 1. 绘制插值曲面（半透明，便于观察背后的散点）
    surf = ax.plot_surface(Phi, T_grid, interp_grid,
                           cmap='viridis', alpha=0.6, edgecolor='none')
    # 2. 绘制原始数据散点（红色，稍大）
    ax.scatter(phi_raw_clean, T_raw_clean, orig_clean,
               c='red', s=10, alpha=0.9, label='Original data')

    ax.set_xlabel(r'$\phi$')
    ax.set_ylabel('$T$')
    ax.set_zlabel(f'Mass {k} value')
    ax.set_title(f'C++ bicubic spline surface + original points (mass index {k})')
    fig.colorbar(surf, ax=ax, shrink=0.5, label='Interpolated value')
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, f"combined_surface_{k}.png"), dpi=150)
    plt.close()
    print(f"Saved combined_surface_{k}.png")

print("All done.")