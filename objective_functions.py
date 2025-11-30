"""
目标函数计算模块
计算双目标优化问题的目标函数值
"""
import numpy as np
from model_training import train_gbr_model


def compute_objectives(X_train, y_train, X_test, y_test, feature_mask, resource_allocation,
                      learning_rate=0.1, n_estimators=50, random_state=42):
    """
    计算双目标函数值
    
    Parameters:
    -----------
    X_train : array
        训练集特征
    y_train : array
        训练集目标
    X_test : array
        测试集特征
    y_test : array
        测试集目标
    feature_mask : array
        特征选择掩码（二进制向量，122维）
    resource_allocation : array
        资源分配向量（95维，每个社区的资源分配比例）
    learning_rate : float
        学习率
    n_estimators : int
        树的数量
    random_state : int
        随机种子
    
    Returns:
    --------
    f1 : float
        目标函数1：资源分配后的预测误差均值
    f2 : int
        目标函数2：选择的特征数量
    r2 : float
        模型R²分数（用于约束检查）
    y_pred_test : array
        测试集预测值
    """
    # 获取选择的特征索引
    feature_indices = np.where(feature_mask == 1)[0].tolist()
    
    # 如果选择的特征数量为0，返回惩罚值
    if len(feature_indices) == 0:
        return 1e6, 0, 0.0, None
    
    # 训练模型
    model, y_pred_train, y_pred_test, r2 = train_gbr_model(
        X_train, y_train, X_test, y_test, feature_indices,
        learning_rate, n_estimators, random_state
    )
    
    # 计算f1：资源分配后的预测误差均值
    # hat_rk(x)：基于特征子集S预测的犯罪率
    hat_rk = y_pred_test.copy()
    
    # rk(y) = hat_rk(x) * (1 - 0.6 * yk)：资源分配后的实际犯罪率
    # 注意：resource_allocation的长度应该等于测试集样本数
    # 如果测试集样本数不等于95，需要调整resource_allocation
    n_test = len(y_test)
    if len(resource_allocation) != n_test:
        # 如果资源分配向量长度不匹配，使用前n_test个值或重复
        if len(resource_allocation) > n_test:
            resource_allocation = resource_allocation[:n_test]
        else:
            # 如果不足，重复填充
            repeat_times = (n_test // len(resource_allocation)) + 1
            resource_allocation = np.tile(resource_allocation, repeat_times)[:n_test]
    
    # 归一化资源分配向量，确保总和为1
    resource_allocation = resource_allocation / resource_allocation.sum()
    
    # 计算资源分配后的实际犯罪率
    rk = hat_rk * (1 - 0.6 * resource_allocation)
    
    # 计算误差均值
    f1 = np.mean(np.abs(hat_rk - rk))
    
    # 计算f2：选择的特征数量
    f2 = int(np.sum(feature_mask))
    
    return f1, f2, r2, y_pred_test


def compute_objectives_vectorized(X_train, y_train, X_test, y_test, population,
                                  learning_rate=0.1, n_estimators=50, random_state=42):
    """
    向量化的目标函数计算（用于优化算法）
    
    Parameters:
    -----------
    X_train : array
        训练集特征
    y_train : array
        训练集目标
    X_test : array
        测试集特征
    y_test : array
        测试集目标
    population : list
        种群，每个个体是一个解向量
        [前122位：特征选择二进制, 后95位：资源分配连续值]
    learning_rate : float
        学习率
    n_estimators : int
        树的数量
    random_state : int
        随机种子
    
    Returns:
    --------
    objectives : array
        目标函数值矩阵 (n_population, 2)
    constraints : array
        约束违反度矩阵 (n_population, n_constraints)
    """
    n_pop = len(population)
    objectives = np.zeros((n_pop, 2))
    constraints = np.zeros((n_pop, 3))  # 3个约束：特征数量下界、上界、R²
    
    for i, individual in enumerate(population):
        # 解码个体
        feature_mask = individual[:122].astype(int)
        resource_allocation = individual[122:122+95]
        
        # 计算目标函数
        f1, f2, r2, _ = compute_objectives(
            X_train, y_train, X_test, y_test, feature_mask, resource_allocation,
            learning_rate, n_estimators, random_state
        )
        
        objectives[i, 0] = f1
        objectives[i, 1] = f2
        
        # 计算约束违反度
        n_features = int(np.sum(feature_mask))
        # 约束1：特征数量 >= 10
        constraints[i, 0] = max(0, 10 - n_features)
        # 约束2：特征数量 <= 30
        constraints[i, 1] = max(0, n_features - 30)
        # 约束3：R² >= 0.7
        constraints[i, 2] = max(0, 0.7 - r2)
    
    return objectives, constraints

