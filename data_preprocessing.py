"""
数据预处理模块
处理Communities and Crime数据集
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


def load_data(data_path='communities+and+crime/communities.data'):
    """
    读取数据文件
    
    Parameters:
    -----------
    data_path : str
        数据文件路径
    
    Returns:
    --------
    df : DataFrame
        原始数据框
    """
    # 读取列名
    column_names = [
        'state', 'county', 'community', 'communityname', 'fold',
        'population', 'householdsize', 'racepctblack', 'racePctWhite', 'racePctAsian', 'racePctHisp',
        'agePct12t21', 'agePct12t29', 'agePct16t24', 'agePct65up',
        'numbUrban', 'pctUrban', 'medIncome', 'pctWWage', 'pctWFarmSelf', 'pctWInvInc', 
        'pctWSocSec', 'pctWPubAsst', 'pctWRetire', 'medFamInc', 'perCapInc',
        'whitePerCap', 'blackPerCap', 'indianPerCap', 'AsianPerCap', 'OtherPerCap', 'HispPerCap',
        'NumUnderPov', 'PctPopUnderPov', 'PctLess9thGrade', 'PctNotHSGrad', 'PctBSorMore',
        'PctUnemployed', 'PctEmploy', 'PctEmplManu', 'PctEmplProfServ', 'PctOccupManu', 'PctOccupMgmtProf',
        'MalePctDivorce', 'MalePctNevMarr', 'FemalePctDiv', 'TotalPctDiv', 'PersPerFam',
        'PctFam2Par', 'PctKids2Par', 'PctYoungKids2Par', 'PctTeen2Par', 'PctWorkMomYoungKids', 'PctWorkMom',
        'NumIlleg', 'PctIlleg', 'NumImmig', 'PctImmigRecent', 'PctImmigRec5', 'PctImmigRec8', 'PctImmigRec10',
        'PctRecentImmig', 'PctRecImmig5', 'PctRecImmig8', 'PctRecImmig10',
        'PctSpeakEnglOnly', 'PctNotSpeakEnglWell', 'PctLargHouseFam', 'PctLargHouseOccup',
        'PersPerOccupHous', 'PersPerOwnOccHous', 'PersPerRentOccHous', 'PctPersOwnOccup', 'PctPersDenseHous',
        'PctHousLess3BR', 'MedNumBR', 'HousVacant', 'PctHousOccup', 'PctHousOwnOcc',
        'PctVacantBoarded', 'PctVacMore6Mos', 'MedYrHousBuilt', 'PctHousNoPhone', 'PctWOFullPlumb',
        'OwnOccLowQuart', 'OwnOccMedVal', 'OwnOccHiQuart', 'RentLowQ', 'RentMedian', 'RentHighQ', 'MedRent',
        'MedRentPctHousInc', 'MedOwnCostPctInc', 'MedOwnCostPctIncNoMtg',
        'NumInShelters', 'NumStreet', 'PctForeignBorn', 'PctBornSameState',
        'PctSameHouse85', 'PctSameCity85', 'PctSameState85',
        'LemasSwornFT', 'LemasSwFTPerPop', 'LemasSwFTFieldOps', 'LemasSwFTFieldPerPop',
        'LemasTotalReq', 'LemasTotReqPerPop', 'PolicReqPerOffic', 'PolicPerPop',
        'RacialMatchCommPol', 'PctPolicWhite', 'PctPolicBlack', 'PctPolicHisp', 'PctPolicAsian', 'PctPolicMinor',
        'OfficAssgnDrugUnits', 'NumKindsDrugsSeiz', 'PolicAveOTWorked',
        'LandArea', 'PopDens', 'PctUsePubTrans', 'PolicCars', 'PolicOperBudg',
        'LemasPctPolicOnPatr', 'LemasGangUnitDeploy', 'LemasPctOfficDrugUn', 'PolicBudgPerPop',
        'ViolentCrimesPerPop'  # 目标变量
    ]
    
    # 读取数据
    df = pd.read_csv(data_path, header=None, names=column_names, na_values='?')
    
    return df


def preprocess_data(df, missing_threshold=0.5, test_size=0.2, random_state=42):
    """
    预处理数据：处理缺失值、标准化、划分数据集
    
    Parameters:
    -----------
    df : DataFrame
        原始数据框
    missing_threshold : float
        缺失值阈值，超过此比例的列将被删除
    test_size : float
        测试集比例
    random_state : int
        随机种子
    
    Returns:
    --------
    X_train : array
        训练集特征
    X_test : array
        测试集特征
    y_train : array
        训练集目标
    y_test : array
        测试集目标
    feature_names : list
        特征名称列表
    scaler : StandardScaler
        标准化器
    """
    # 分离非预测性特征和目标变量
    non_predictive = ['state', 'county', 'community', 'communityname', 'fold']
    target = 'ViolentCrimesPerPop'
    
    # 获取预测性特征
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    predictive_features = [f for f in numeric_features if f not in non_predictive + [target]]
    
    # 提取特征和目标
    X = df[predictive_features].copy()
    y = df[target].copy()
    
    # 处理缺失值
    # 1. 删除缺失率超过阈值的特征
    missing_ratio = X.isnull().sum() / len(X)
    features_to_drop = missing_ratio[missing_ratio > missing_threshold].index.tolist()
    X = X.drop(columns=features_to_drop)
    predictive_features = [f for f in predictive_features if f not in features_to_drop]
    
    print(f"删除了 {len(features_to_drop)} 个缺失率超过{missing_threshold*100}%的特征")
    
    # 2. 删除包含目标变量缺失值的行
    valid_mask = ~y.isnull()
    X = X[valid_mask].copy()
    y = y[valid_mask].copy()
    
    # 3. 用中位数填充剩余缺失值
    X = X.fillna(X.median())
    
    # 确保特征数量为122（如果不足，说明有些特征被删除了）
    print(f"最终特征数量: {len(predictive_features)}")
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return (X_train_scaled, X_test_scaled, y_train.values, y_test.values, 
            predictive_features, scaler)


def get_feature_indices(feature_names, selected_features):
    """
    根据特征名称获取特征索引
    
    Parameters:
    -----------
    feature_names : list
        所有特征名称列表
    selected_features : list
        选中的特征名称列表
    
    Returns:
    --------
    indices : list
        特征索引列表
    """
    indices = []
    for feat in selected_features:
        if feat in feature_names:
            indices.append(feature_names.index(feat))
    return indices

