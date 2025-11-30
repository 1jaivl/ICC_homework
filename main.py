"""
主程序：治安资源配置双目标优化
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os
from data_preprocessing import load_data, preprocess_data
from optimization import NSGA2
from results_analysis import analyze_results, visualize_results

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def main():
    """
    主函数
    """
    print("=" * 60)
    print("治安资源配置双目标优化系统")
    print("=" * 60)
    
    # 1. 数据预处理
    print("\n[1/5] 数据预处理...")
    df = load_data('communities+and+crime/communities.data')
    X_train, X_test, y_train, y_test, feature_names, scaler = preprocess_data(
        df, missing_threshold=0.5, test_size=0.05, random_state=41
    )
    
    print(f"训练集大小: {X_train.shape}")
    print(f"测试集大小: {X_test.shape}")
    print(f"特征数量: {len(feature_names)}")
    
    # 注意：资源分配向量长度应该等于测试集样本数
    n_communities = len(y_test)
    print(f"社区数量（测试集）: {n_communities}")
    
    # 2. 初始化优化算法
    print("\n[2/5] 初始化NSGA-II算法...")
    optimizer = NSGA2(
        n_features=len(feature_names),
        n_communities=n_communities,  # 设置为None会在run方法中自动确定
        min_features=10,
        max_features=30,
        min_ratio=0.005,
        max_ratio=0.05,
        pop_size=100,  # 种群大小（可根据计算资源调整）
        max_gen=50,  # 最大迭代次数（增加迭代次数以观察更明显的收敛过程）
        crossover_rate=0.9,
        mutation_rate=0.1  # 稍微提高变异率以增加探索
    )
    
    # 3. 运行优化
    print("\n[3/5] 运行优化算法...")
    print("这可能需要一些时间，请耐心等待...")
    pareto_front, pareto_objectives, history = optimizer.run(
        X_train, y_train, X_test, y_test, verbose=True
    )
    
    print(f"\n找到 {len(pareto_front)} 个Pareto最优解")
    
    # 4. 保存结果
    print("\n[4/5] 保存结果...")
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    
    # 保存Pareto前沿
    np.save(f'{results_dir}/pareto_front.npy', pareto_front)
    np.save(f'{results_dir}/pareto_objectives.npy', pareto_objectives)
    
    # 保存历史
    with open(f'{results_dir}/history.pkl', 'wb') as f:
        pickle.dump(history, f)
    
    # 保存特征名称
    with open(f'{results_dir}/feature_names.pkl', 'wb') as f:
        pickle.dump(feature_names, f)
    
    # 保存数据信息
    data_info = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'feature_names': feature_names
    }
    with open(f'{results_dir}/data_info.pkl', 'wb') as f:
        pickle.dump(data_info, f)
    
    print(f"结果已保存到 {results_dir}/ 目录")
    
    # 5. 结果分析
    print("\n[5/5] 结果分析...")
    analyze_results(pareto_front, pareto_objectives, feature_names, 
                   X_train, y_train, X_test, y_test, history)
    
    print("\n" + "=" * 60)
    print("优化完成！")
    print("=" * 60)


if __name__ == '__main__':
    main()

