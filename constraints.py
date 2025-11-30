"""
约束处理模块
处理优化问题的约束条件
"""
import numpy as np


def check_feature_count_constraint(feature_mask, min_features=10, max_features=30):
    """
    检查特征数量约束
    
    Parameters:
    -----------
    feature_mask : array
        特征选择掩码（二进制向量）
    min_features : int
        最小特征数量
    max_features : int
        最大特征数量
    
    Returns:
    --------
    is_valid : bool
        是否满足约束
    violation : float
        约束违反度（0表示满足）
    """
    n_features = int(np.sum(feature_mask))
    
    if n_features < min_features:
        violation = min_features - n_features
        return False, violation
    elif n_features > max_features:
        violation = n_features - max_features
        return False, violation
    else:
        return True, 0.0


def check_resource_allocation_constraint(resource_allocation, min_ratio=0.005, max_ratio=0.05):
    """
    检查资源分配约束
    
    Parameters:
    -----------
    resource_allocation : array
        资源分配向量
    min_ratio : float
        最小分配比例
    max_ratio : float
        最大分配比例
    
    Returns:
    --------
    is_valid : bool
        是否满足约束
    violation : float
        约束违反度（0表示满足）
    normalized_allocation : array
        归一化后的资源分配向量
    """
    # 归一化，确保总和为1
    total = np.sum(resource_allocation)
    if total == 0:
        # 如果总和为0，平均分配
        normalized_allocation = np.ones_like(resource_allocation) / len(resource_allocation)
    else:
        normalized_allocation = resource_allocation / total
    
    # 检查每个元素是否在范围内
    violations = []
    for i, val in enumerate(normalized_allocation):
        if val < min_ratio:
            violations.append(min_ratio - val)
        elif val > max_ratio:
            violations.append(val - max_ratio)
    
    if len(violations) > 0:
        return False, max(violations), normalized_allocation
    else:
        return True, 0.0, normalized_allocation


def check_r2_constraint(r2, min_r2=0.7):
    """
    检查R²约束
    
    Parameters:
    -----------
    r2 : float
        R²分数
    min_r2 : float
        最小R²要求
    
    Returns:
    --------
    is_valid : bool
        是否满足约束
    violation : float
        约束违反度（0表示满足）
    """
    if r2 < min_r2:
        violation = min_r2 - r2
        return False, violation
    else:
        return True, 0.0


def repair_individual(individual, min_features=10, max_features=30, 
                     min_ratio=0.005, max_ratio=0.05):
    """
    修复个体，使其满足约束
    
    Parameters:
    -----------
    individual : array
        个体向量 [前122位：特征选择, 后95位：资源分配]
    min_features : int
        最小特征数量
    max_features : int
        最大特征数量
    min_ratio : float
        最小资源分配比例
    max_ratio : float
        最大资源分配比例
    
    Returns:
    --------
    repaired_individual : array
        修复后的个体
    """
    repaired = individual.copy()
    
    # 修复特征选择部分
    feature_mask = repaired[:122]
    n_features = int(np.sum(feature_mask))
    
    if n_features < min_features:
        # 如果特征太少，随机添加特征
        zero_indices = np.where(feature_mask == 0)[0]
        n_to_add = min_features - n_features
        if len(zero_indices) >= n_to_add:
            selected = np.random.choice(zero_indices, n_to_add, replace=False)
            feature_mask[selected] = 1
    elif n_features > max_features:
        # 如果特征太多，随机删除特征
        one_indices = np.where(feature_mask == 1)[0]
        n_to_remove = n_features - max_features
        if len(one_indices) >= n_to_remove:
            selected = np.random.choice(one_indices, n_to_remove, replace=False)
            feature_mask[selected] = 0
    
    repaired[:122] = feature_mask
    
    # 修复资源分配部分
    resource_allocation = repaired[122:122+95]
    
    # 归一化
    total = np.sum(resource_allocation)
    if total == 0:
        resource_allocation = np.ones(95) / 95
    else:
        resource_allocation = resource_allocation / total
    
    # 裁剪到有效范围
    resource_allocation = np.clip(resource_allocation, min_ratio, max_ratio)
    
    # 重新归一化
    total = np.sum(resource_allocation)
    if total != 1.0:
        # 如果归一化后总和不为1，需要调整
        # 策略：按比例缩放，然后裁剪，重复直到收敛
        max_iter = 100
        for _ in range(max_iter):
            resource_allocation = resource_allocation / total
            resource_allocation = np.clip(resource_allocation, min_ratio, max_ratio)
            total = np.sum(resource_allocation)
            if abs(total - 1.0) < 1e-6:
                break
        
        # 如果仍未归一化，进行微调
        if abs(total - 1.0) > 1e-6:
            diff = 1.0 - total
            # 将差值均匀分配到所有元素（在范围内）
            adjustment = diff / len(resource_allocation)
            resource_allocation = resource_allocation + adjustment
            resource_allocation = np.clip(resource_allocation, min_ratio, max_ratio)
    
    repaired[122:122+95] = resource_allocation
    
    return repaired

