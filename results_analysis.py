"""
结果分析模块
分析优化结果，可视化Pareto前沿，输出最优解
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from model_training import train_gbr_model
from objective_functions import compute_objectives

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def analyze_results(pareto_front, pareto_objectives, feature_names,
                   X_train, y_train, X_test, y_test, history):
    """
    分析优化结果
    
    Parameters:
    -----------
    pareto_front : array
        Pareto前沿解
    pareto_objectives : array
        Pareto前沿目标函数值
    feature_names : list
        特征名称列表
    X_train : array
        训练集特征
    y_train : array
        训练集目标
    X_test : array
        测试集特征
    y_test : array
        测试集目标
    history : dict
        优化历史
    """
    print("\n" + "=" * 60)
    print("结果分析")
    print("=" * 60)
    
    # 1. 可视化Pareto前沿
    visualize_pareto_front(pareto_objectives, history)
    
    # 2. 选择最终解（使用加权和方法）
    best_solution_idx = select_best_solution(pareto_objectives)
    best_solution = pareto_front[best_solution_idx]
    best_objectives = pareto_objectives[best_solution_idx]
    
    print(f"\n选择的最优解索引: {best_solution_idx}")
    print(f"目标函数值: f1={best_objectives[0]:.6f}, f2={best_objectives[1]}")
    
    # 3. 分析最优解
    analyze_best_solution(best_solution, feature_names, X_train, y_train, 
                         X_test, y_test)
    
    # 4. 输出所有Pareto解的统计信息
    print_pareto_statistics(pareto_front, pareto_objectives, feature_names)


def visualize_pareto_front(pareto_objectives, history):
    """
    可视化Pareto前沿
    
    Parameters:
    -----------
    pareto_objectives : array
        Pareto前沿目标函数值
    history : dict
        优化历史
    """
    fig = plt.figure(figsize=(16, 10))
    
    # 子图1：Pareto前沿
    ax1 = plt.subplot(2, 2, 1)
    ax1.scatter(pareto_objectives[:, 0], pareto_objectives[:, 1], 
               c='red', s=50, alpha=0.7, label='Pareto前沿')
    ax1.set_xlabel('f1: 预测误差均值', fontsize=12)
    ax1.set_ylabel('f2: 特征数量', fontsize=12)
    ax1.set_title('最终Pareto前沿', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 子图2：f1的演化过程
    ax2 = plt.subplot(2, 2, 2)
    if history and 'generations' in history:
        if 'best_f1' in history:
            ax2.plot(history['generations'], history['best_f1'], 
                     'b-', label='最佳f1', linewidth=2, marker='o', markersize=4)
        if 'mean_f1' in history:
            ax2.plot(history['generations'], history['mean_f1'], 
                     'b--', label='平均f1', linewidth=1.5, alpha=0.7)
        if 'median_f1' in history:
            ax2.plot(history['generations'], history['median_f1'], 
                     'b:', label='中位数f1', linewidth=1.5, alpha=0.7)
    ax2.set_xlabel('迭代次数', fontsize=12)
    ax2.set_ylabel('f1值', fontsize=12)
    ax2.set_title('f1演化过程', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # 子图3：f2的演化过程
    ax3 = plt.subplot(2, 2, 3)
    if history and 'generations' in history:
        if 'best_f2' in history:
            ax3.plot(history['generations'], history['best_f2'], 
                     'r-', label='最佳f2', linewidth=2, marker='s', markersize=4)
        if 'mean_f2' in history:
            ax3.plot(history['generations'], history['mean_f2'], 
                     'r--', label='平均f2', linewidth=1.5, alpha=0.7)
        if 'median_f2' in history:
            ax3.plot(history['generations'], history['median_f2'], 
                     'r:', label='中位数f2', linewidth=1.5, alpha=0.7)
    ax3.set_xlabel('迭代次数', fontsize=12)
    ax3.set_ylabel('f2值', fontsize=12)
    ax3.set_title('f2演化过程', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # 子图4：Pareto前沿大小和可行解比例
    ax4 = plt.subplot(2, 2, 4)
    if history and 'generations' in history:
        if 'pareto_size' in history:
            ax4_twin = ax4.twinx()
            ax4.plot(history['generations'], history['pareto_size'], 
                    'g-', label='Pareto前沿大小', linewidth=2, marker='^', markersize=4)
            ax4.set_xlabel('迭代次数', fontsize=12)
            ax4.set_ylabel('Pareto前沿大小', fontsize=12, color='g')
            ax4.tick_params(axis='y', labelcolor='g')
            
            if 'feasible_ratio' in history:
                ax4_twin.plot(history['generations'], 
                            [r * 100 for r in history['feasible_ratio']], 
                            'orange', label='可行解比例(%)', linewidth=2, 
                            marker='d', markersize=4, linestyle='--')
                ax4_twin.set_ylabel('可行解比例 (%)', fontsize=12, color='orange')
                ax4_twin.tick_params(axis='y', labelcolor='orange')
            
            ax4.set_title('Pareto前沿大小和可行解比例', fontsize=14, fontweight='bold')
            ax4.grid(True, alpha=0.3)
            
            # 合并图例
            lines1, labels1 = ax4.get_legend_handles_labels()
            lines2, labels2 = ax4_twin.get_legend_handles_labels()
            ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.tight_layout()
    plt.savefig('results/pareto_front.png', dpi=300, bbox_inches='tight')
    print("Pareto前沿图已保存到 results/pareto_front.png")
    plt.close()


def select_best_solution(pareto_objectives, weight_f1=0.9, weight_f2=0.1):
    """
    选择最优解（使用加权和方法）
    
    Parameters:
    -----------
    pareto_objectives : array
        Pareto前沿目标函数值
    weight_f1 : float
        f1的权重
    weight_f2 : float
        f2的权重
    
    Returns:
    --------
    best_idx : int
        最优解的索引
    """
    # 归一化目标函数值
    f1_norm = (pareto_objectives[:, 0] - pareto_objectives[:, 0].min()) / \
              (pareto_objectives[:, 0].max() - pareto_objectives[:, 0].min() + 1e-10)
    f2_norm = (pareto_objectives[:, 1] - pareto_objectives[:, 1].min()) / \
              (pareto_objectives[:, 1].max() - pareto_objectives[:, 1].min() + 1e-10)
    
    # 计算加权和
    weighted_sum = weight_f1 * f1_norm + weight_f2 * f2_norm
    
    # 选择加权和最小的解
    best_idx = np.argmin(weighted_sum)
    
    return best_idx


def analyze_best_solution(solution, feature_names, X_train, y_train, 
                         X_test, y_test):
    """
    分析最优解
    
    Parameters:
    -----------
    solution : array
        最优解向量
    feature_names : list
        特征名称列表
    X_train : array
        训练集特征
    y_train : array
        训练集目标
    X_test : array
        测试集特征
    y_test : array
        测试集目标
    """
    n_features = len(feature_names)
    
    # 提取特征选择和资源分配
    feature_mask = solution[:n_features].astype(int)
    # 资源分配部分从n_features开始，长度等于测试集样本数
    n_test = len(y_test)
    resource_allocation = solution[n_features:n_features+n_test]
    
    # 获取选择的特征
    selected_features = [feature_names[i] for i in range(n_features) if feature_mask[i] == 1]
    selected_indices = np.where(feature_mask == 1)[0]
    
    print("\n" + "=" * 60)
    print("最优解分析")
    print("=" * 60)
    
    print(f"\n选择的特征数量: {len(selected_features)}")
    print(f"\n选择的特征列表:")
    for i, feat in enumerate(selected_features, 1):
        print(f"  {i}. {feat}")
    
    # 训练模型并评估
    model, y_pred_train, y_pred_test, r2 = train_gbr_model(
        X_train, y_train, X_test, y_test, selected_indices.tolist()
    )
    
    print(f"\n模型性能:")
    print(f"  测试集R²: {r2:.4f}")
    print(f"  测试集MAE: {np.mean(np.abs(y_test - y_pred_test)):.6f}")
    
    # 资源分配分析
    print(f"\n资源分配统计:")
    print(f"  最小分配比例: {resource_allocation.min():.4f}")
    print(f"  最大分配比例: {resource_allocation.max():.4f}")
    print(f"  平均分配比例: {resource_allocation.mean():.4f}")
    print(f"  分配总和: {resource_allocation.sum():.4f}")
    
    # 找出资源分配最多的前10个社区
    top_communities = np.argsort(-resource_allocation)[:10]
    print(f"\n资源分配最多的前10个社区:")
    for i, idx in enumerate(top_communities, 1):
        print(f"  社区 {idx}: {resource_allocation[idx]:.4f} ({resource_allocation[idx]*100:.2f}%)")
    
    # 可视化资源分配
    visualize_resource_allocation(resource_allocation, y_test, y_pred_test)
    
    # 保存结果到CSV
    save_solution_to_csv(solution, feature_names, selected_features, 
                        resource_allocation, r2)


def visualize_resource_allocation(resource_allocation, y_test, y_pred_test):
    """
    可视化资源分配
    
    Parameters:
    -----------
    resource_allocation : array
        资源分配向量
    y_test : array
        测试集真实值
    y_pred_test : array
        测试集预测值
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 子图1：资源分配分布
    axes[0, 0].hist(resource_allocation, bins=30, edgecolor='black', alpha=0.7)
    axes[0, 0].set_xlabel('资源分配比例', fontsize=12)
    axes[0, 0].set_ylabel('社区数量', fontsize=12)
    axes[0, 0].set_title('资源分配分布', fontsize=14, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 子图2：资源分配 vs 预测犯罪率
    axes[0, 1].scatter(y_pred_test, resource_allocation, alpha=0.6, s=30)
    axes[0, 1].set_xlabel('预测犯罪率', fontsize=12)
    axes[0, 1].set_ylabel('资源分配比例', fontsize=12)
    axes[0, 1].set_title('资源分配 vs 预测犯罪率', fontsize=14, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 子图3：资源分配 vs 真实犯罪率
    axes[1, 0].scatter(y_test, resource_allocation, alpha=0.6, s=30, c='red')
    axes[1, 0].set_xlabel('真实犯罪率', fontsize=12)
    axes[1, 0].set_ylabel('资源分配比例', fontsize=12)
    axes[1, 0].set_title('资源分配 vs 真实犯罪率', fontsize=14, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 子图4：资源分配后的犯罪率变化
    rk_after = y_pred_test * (1 - 0.6 * resource_allocation)
    reduction = y_pred_test - rk_after
    axes[1, 1].scatter(y_pred_test, reduction, alpha=0.6, s=30, c='green')
    axes[1, 1].set_xlabel('原始预测犯罪率', fontsize=12)
    axes[1, 1].set_ylabel('犯罪率降低量', fontsize=12)
    axes[1, 1].set_title('资源分配后的犯罪率降低', fontsize=14, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/resource_allocation.png', dpi=300, bbox_inches='tight')
    print("资源分配分析图已保存到 results/resource_allocation.png")
    plt.close()


def print_pareto_statistics(pareto_front, pareto_objectives, feature_names):
    """
    打印Pareto解的统计信息
    
    Parameters:
    -----------
    pareto_front : array
        Pareto前沿解
    pareto_objectives : array
        Pareto前沿目标函数值
    feature_names : list
        特征名称列表
    """
    n_features = len(feature_names)
    
    print("\n" + "=" * 60)
    print("Pareto前沿统计信息")
    print("=" * 60)
    
    # 统计特征数量分布
    feature_counts = []
    for solution in pareto_front:
        feature_mask = solution[:n_features].astype(int)
        feature_counts.append(int(np.sum(feature_mask)))
    
    feature_counts = np.array(feature_counts)
    
    print(f"\n特征数量统计:")
    print(f"  最小值: {feature_counts.min()}")
    print(f"  最大值: {feature_counts.max()}")
    print(f"  平均值: {feature_counts.mean():.2f}")
    print(f"  中位数: {np.median(feature_counts):.2f}")
    
    print(f"\n目标函数统计:")
    print(f"  f1 (预测误差):")
    print(f"    最小值: {pareto_objectives[:, 0].min():.6f}")
    print(f"    最大值: {pareto_objectives[:, 0].max():.6f}")
    print(f"    平均值: {pareto_objectives[:, 0].mean():.6f}")
    print(f"  f2 (特征数量):")
    print(f"    最小值: {pareto_objectives[:, 1].min()}")
    print(f"    最大值: {pareto_objectives[:, 1].max()}")
    print(f"    平均值: {pareto_objectives[:, 1].mean():.2f}")


def save_solution_to_csv(solution, feature_names, selected_features,
                         resource_allocation, r2):
    """
    保存解到CSV文件
    
    Parameters:
    -----------
    solution : array
        解向量
    feature_names : list
        特征名称列表
    selected_features : list
        选择的特征列表
    resource_allocation : array
        资源分配向量
    r2 : float
        R²分数
    """
    # 保存特征选择结果
    feature_df = pd.DataFrame({
        '特征名称': selected_features,
        '是否选择': [1] * len(selected_features)
    })
    feature_df.to_csv('results/selected_features.csv', index=False, encoding='utf-8-sig')
    
    # 保存资源分配结果
    resource_df = pd.DataFrame({
        '社区索引': range(len(resource_allocation)),
        '资源分配比例': resource_allocation,
        '资源分配百分比': resource_allocation * 100
    })
    resource_df = resource_df.sort_values('资源分配比例', ascending=False)
    resource_df.to_csv('results/resource_allocation.csv', index=False, encoding='utf-8-sig')
    
    # 保存解摘要
    summary = {
        '选择的特征数量': [len(selected_features)],
        'R²分数': [r2],
        '资源分配总和': [resource_allocation.sum()],
        '最小资源分配': [resource_allocation.min()],
        '最大资源分配': [resource_allocation.max()],
        '平均资源分配': [resource_allocation.mean()]
    }
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv('results/solution_summary.csv', index=False, encoding='utf-8-sig')
    
    print("\n结果已保存到CSV文件:")
    print("  - results/selected_features.csv")
    print("  - results/resource_allocation.csv")
    print("  - results/solution_summary.csv")


def visualize_results(pareto_objectives, history):
    """
    可视化所有结果（兼容性函数）
    """
    visualize_pareto_front(pareto_objectives, history)

