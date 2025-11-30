"""
模型训练模块
实现梯度提升回归模型
"""
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error


def train_gbr_model(X_train, y_train, X_test, y_test, feature_indices, 
                    learning_rate=0.1, n_estimators=50, random_state=42):
    """
    训练梯度提升回归模型
    
    Parameters:
    -----------
    X_train : array
        训练集特征
    X_test : array
        测试集特征
    y_train : array
        训练集目标
    y_test : array
        测试集目标
    feature_indices : list
        选择的特征索引列表
    learning_rate : float
        学习率
    n_estimators : int
        树的数量
    random_state : int
        随机种子
    
    Returns:
    --------
    model : GradientBoostingRegressor
        训练好的模型
    y_pred_train : array
        训练集预测值
    y_pred_test : array
        测试集预测值
    r2 : float
        测试集R²分数
    """
    # 选择特征子集
    X_train_selected = X_train[:, feature_indices]
    X_test_selected = X_test[:, feature_indices]
    
    # 训练模型
    model = GradientBoostingRegressor(
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        random_state=random_state,
        max_depth=3,
        subsample=0.8
    )
    
    model.fit(X_train_selected, y_train)
    
    # 预测
    y_pred_train = model.predict(X_train_selected)
    y_pred_test = model.predict(X_test_selected)
    
    # 计算R²
    r2 = r2_score(y_test, y_pred_test)
    
    return model, y_pred_train, y_pred_test, r2


def evaluate_model(y_true, y_pred):
    """
    评估模型性能
    
    Parameters:
    -----------
    y_true : array
        真实值
    y_pred : array
        预测值
    
    Returns:
    --------
    metrics : dict
        评估指标字典
    """
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    return {
        'MAE': mae,
        'R2': r2,
        'RMSE': rmse
    }

