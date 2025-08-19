import matplotlib.pyplot as plt
import numpy as np

# 读取数据
data = np.loadtxt('data_acm.txt', delimiter=',')

# 提取列
node_drop = data[:, 0]
ma_f1 = data[:, 1]
mi_f1 = data[:, 2]
auc = data[:, 3]

# 创建图表
plt.figure(figsize=(8, 8))  # 设置正方形画布大小

# 绘制三条曲线：设置 linewidth 和 markersize 控制视觉紧凑度
plt.plot(node_drop, ma_f1, 'b-o', label='Macro F1', linewidth=2, markersize=10)
plt.plot(node_drop, mi_f1, 'r-s', label='Micro F1', linewidth=2, markersize=10)
plt.plot(node_drop, auc, 'g-^', label='AUC', linewidth=2, markersize=10)

# 设置横轴、纵轴字体大小
plt.xlabel('Node Drop Rate', fontsize=30)
plt.ylabel('Score', fontsize=30)

# 设置坐标轴刻度字体大小
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)

# 图例字体设置大些
plt.legend(fontsize=20)

# 添加网格
plt.grid(True, linestyle='--', alpha=0.6)

# 自动布局
# 设置纵轴范围让曲线更集中
y_min = min(ma_f1.min(), mi_f1.min(), auc.min()) - 1
y_max = max(ma_f1.max(), mi_f1.max(), auc.max()) + 1
plt.ylim(y_min, y_max)

plt.tight_layout()

# 显示图
plt.show()

# 保存图像（如需）
plt.savefig('acm_node_drop.pdf', format='pdf', bbox_inches='tight')
