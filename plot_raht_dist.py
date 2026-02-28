import torch
import matplotlib.pyplot as plt
import numpy as np

# 加载数据
file_path = "raht_C_iter0.pt"
print(f"Loading RAHT features from {file_path} ...")
C = torch.load(file_path).numpy()
print(f"C shape: {C.shape}")

# 特征维度定义 (opacity+euler+f_dc+f_rest+scale = 1+3+3+45+3 = 55)
# f_rest 分为三部分: sh_1 (9), sh_2 (15), sh_3 (21)
features = {
    'opacity': (0, 1),
    'euler': (1, 4),
    'f_dc': (4, 7),
    'f_rest_0 (sh_1)': (7, 16),      # 9
    'f_rest_1 (sh_2)': (16, 31),     # 15
    'f_rest_2 (sh_3)': (31, 52),     # 21
    'scale': (52, 55)
}

fig, axs = plt.subplots(len(features), 2, figsize=(15, 4 * len(features)))
fig.suptitle('RAHT Features Distribution (Before Quantization) - Iteration 0\nDC vs AC Coefficients', fontsize=16)

for i, (name, (start, end)) in enumerate(features.items()):
    # 提取特定特征的数据
    feat_data = C[:, start:end]
    
    # 分离 DC 和 AC
    dc_data = feat_data[0].flatten()
    ac_data = feat_data[1:].flatten()
    
    # 绘制 DC (只有一个点，画个柱形表示值)
    axs[i, 0].bar(range(len(dc_data)), dc_data, color='blue', alpha=0.7)
    axs[i, 0].set_title(f'DC Coefficient(s) - {name}')
    axs[i, 0].set_xticks(range(len(dc_data)))
    axs[i, 0].grid(True, alpha=0.3)
    
    # 绘制 AC (直方图)
    # 为了更好可视化，排除极端的 top/bottom 0.1% 的离群值
    if len(ac_data) > 0:
        p1, p99 = np.percentile(ac_data, [0.1, 99.9])
        filtered_ac = ac_data[(ac_data >= p1) & (ac_data <= p99)]
        
        axs[i, 1].hist(filtered_ac, bins=100, color='orange', alpha=0.7, density=True)
        axs[i, 1].set_title(f'AC Coefficients Distribution - {name}')
        axs[i, 1].set_xlabel('Value')
        axs[i, 1].set_ylabel('Density')
        axs[i, 1].grid(True, alpha=0.3)
        
        # 添加统计信息文本
        stats_text = f"Min: {ac_data.min():.3f}\nMax: {ac_data.max():.3f}\nMean: {ac_data.mean():.3f}\nStd: {ac_data.std():.3f}"
        axs[i, 1].text(0.05, 0.95, stats_text, transform=axs[i, 1].transAxes, 
                      fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout(rect=[0, 0.03, 1, 0.97])
plt.savefig('raht_distribution_iter0.png', dpi=150)
print("Plot saved as 'raht_distribution_iter0.png'")
