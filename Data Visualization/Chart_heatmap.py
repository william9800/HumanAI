import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.colors import Normalize
from matplotlib import gridspec

# ==== 原始数据 ====
data = {
    'Dataset': [
        '500stylegan+500unsplashed', '1stylegan+500unsplashed',
        '21Dall3+1FFHQ', '21Dall3+21FFHQ',
        '40flux+1FFHQ', '40flux+40FFHQ',
        '100doubao+100FFHQ', '100SDXL+1FFHQ',
        '100SDXL+100FFHQ', '500stable+500FFHQ',
        '1000Lexica+1000FFHQ', '1000randomweb+1000FFHQ',
        '2000generated+1FFHQ', '2000generated+2000FFHQ',
        '2000stylegan2+1FFHQ', '2000stylegan2+2000FFHQ',
        '2000stylegan+1FFHQ', '2000stylegan+2000CelebA',
        '2000stylegan+2000FFHQ', '2000stylegan+2000Star'
    ],
    'MD': [99.90, 99.60, 31.82, 64.29, 97.56, 98.75, 82.50, 47.52, 73.50, 100,
           99.85, 99.85, np.nan, 96.03, np.nan, 99.92, np.nan, 100, 99.92, 99.98],
    'EP': [100.00, 99.10, 4.55, 50.00, 7.32, 52.50, 49.00, 12.87, 56.00, 49.60,
           65.15, 64.65, np.nan, 49.23, np.nan, 84.30, np.nan, np.nan, 98.40, np.nan],
    'EfficientNet-B7': [99.90, 99.90, 4.55, 50.00, 2.44, 50.00, 50.00, 0.99, 50.00, 49.80,
                        49.95, 53.50, 1.40, 50.52, 50.92, 75.30, 99.60, 96.05, 99.58, 99.65]
}
df = pd.DataFrame(data)

# ==== 差值计算 ====
diff_md = df['MD'] - df['EfficientNet-B7']
diff_ep = df['EP'] - df['EfficientNet-B7']
heatmap_data = pd.DataFrame({'MD': diff_md, 'EP': diff_ep}).T
heatmap_data.columns = df['Dataset']

# ==== 分两行显示 ====
top_half = heatmap_data.iloc[:, :10]
bottom_half = heatmap_data.iloc[:, 10:]

# ==== 绘图布局 ====
fig = plt.figure(figsize=(18, 6))
gs = gridspec.GridSpec(2, 2, width_ratios=[20, 1], height_ratios=[1, 1], wspace=0.05, hspace=0.5)

# ==== 色彩设定 ====
vmin = np.nanmin(heatmap_data.values)
vmax = np.nanmax(heatmap_data.values)
norm = Normalize(vmin=vmin, vmax=vmax)
cmap = "RdYlGn"

# ==== 子图 1 ====
ax1 = fig.add_subplot(gs[0, 0])
sns.heatmap(top_half, cmap=cmap, norm=norm, annot=True, fmt='.1f',
            cbar=False, ax=ax1, linewidths=0.5)
ax1.set_title("Top 10 Dataset Comparison (ΔAccuracy)", fontsize=14)
ax1.set_ylabel("Model", fontsize=12)
ax1.set_xlabel("")  # ✅ 去除 X 轴标签
ax1.set_xticklabels(top_half.columns, rotation=15, ha='right', fontsize=10)

# ==== 子图 2 ====
ax2 = fig.add_subplot(gs[1, 0])
sns.heatmap(bottom_half, cmap=cmap, norm=norm, annot=True, fmt='.1f',
            cbar=False, ax=ax2, linewidths=0.5)
ax2.set_title("Bottom 10 Dataset Comparison (ΔAccuracy)", fontsize=14)
ax2.set_ylabel("Model", fontsize=12)
ax2.set_xlabel("")  # ✅ 去除 X 轴标签
ax2.set_xticklabels(bottom_half.columns, rotation=15, ha='right', fontsize=10)

# ==== Colorbar 右侧 ====
cax = fig.add_subplot(gs[:, 1])
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, cax=cax, ticks=[round(vmin, 1), round(vmax, 1)])
cbar.ax.set_title("ΔAcc", fontsize=12)

# ==== 总体布局 ====
plt.suptitle("Accuracy Gain of MD and EP over EfficientNet-B7", fontsize=18)
plt.tight_layout(rect=[0, 0.05, 0.95, 0.95])
plt.savefig("heatmap_accuracy_comparison_split_cleaned.png", dpi=300)
plt.show()
