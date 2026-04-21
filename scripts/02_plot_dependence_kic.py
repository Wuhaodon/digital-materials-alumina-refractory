"""
从保存的SHAP数据生成自定义dependence plot
适用于KIC_ablation数据
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]

warnings.filterwarnings("ignore")
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False

# 数据路径
data_dir = PROJECT_ROOT / "results" / "cv" / "shap_exports" / "KIC_ablation" / "shap_exports"
output_dir = PROJECT_ROOT / "results" / "cv" / "shap_exports" / "KIC_ablation"

# 读取数据
print("读取SHAP数据...")
X_orig = pd.read_csv(f"{data_dir}/X_explain_original_units.csv")
shap_vals = pd.read_csv(f"{data_dir}/shap_values.csv")

# 去掉row_index列
features = [c for c in X_orig.columns if c != "row_index"]
X_data = X_orig[features]
shap_data = shap_vals[features]

print(f"特征: {features}")
print(f"样本数: {len(X_data)}")

# 生成所有特征的dependence plot
for feat in features:
    print(f"\n生成 {feat} 的dependence plot...")
    
    # 提取数据
    x_vals = X_data[feat].values
    shap_vals_feat = shap_data[feat].values
    
    # 按SHAP值正负分组
    positive_mask = shap_vals_feat > 0
    negative_mask = shap_vals_feat <= 0
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 绘制散点
    ax.scatter(x_vals[positive_mask], shap_vals_feat[positive_mask], 
               c='#d62728', s=50, alpha=0.7, label='SHAP > 0', edgecolors='none')
    ax.scatter(x_vals[negative_mask], shap_vals_feat[negative_mask], 
               c='#1f77b4', s=50, alpha=0.7, label='SHAP < 0', edgecolors='none')
    
    # 添加y=0参考线
    ax.axhline(y=0, color='black', linewidth=1.5, linestyle='-')
    
    # 设置标签和标题
    ax.set_xlabel(feat, fontsize=14, fontweight='bold')
    ax.set_ylabel(f'SHAP value for {feat}', fontsize=14, fontweight='bold')
    ax.set_title(f'SHAP Dependence Plot for KIC - {feat} (original units)', 
                 fontsize=16, fontweight='bold')
    
    # 图例
    ax.legend(fontsize=12, frameon=True, fancybox=True, shadow=True, loc='best')
    
    # 网格
    ax.grid(False)
    ax.spines['top'].set_linewidth(1.5)
    ax.spines['right'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    
    # 刻度
    ax.tick_params(labelsize=12, width=1.5, length=6)
    
    plt.tight_layout()
    
    # 保存
    safe_feat = feat.replace('/', '_').replace(' ', '_')
    plt.savefig(f"{output_dir}/Custom_Dependence_{safe_feat}.pdf", 
                format="pdf", bbox_inches="tight", dpi=300)
    plt.savefig(f"{output_dir}/Custom_Dependence_{safe_feat}.png", 
                dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"  ✓ 保存至 Custom_Dependence_{safe_feat}.pdf")

print(f"\n{'='*60}")
print(f"所有图表已生成!")
print(f"保存位置: {output_dir}")
print(f"{'='*60}")
