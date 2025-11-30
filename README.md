# 控制算法课程实验项目

本项目包含5个优化算法实验，涵盖车辆路径问题(VRP)和治安资源配置双目标优化问题。

## 项目简介

本项目实现了多种优化算法来解决两类问题：

1. **车辆路径问题(VRP)**: 使用4种不同的优化算法（遗传算法、禁忌搜索、模拟退火、粒子群优化）求解带约束的车辆路径优化问题
2. **治安资源配置优化**: 使用NSGA-II多目标优化算法求解特征选择和资源分配的双目标优化问题

## 项目结构

```
.
├── README.md                      # 项目说明文档
├── main.py                        # 实验5：治安资源配置双目标优化（NSGA-II）
├── exp1.py                        # 实验1：遗传算法求解VRP
├── exp2_tabu.py                   # 实验2：禁忌搜索求解VRP
├── exp3_sa.py                     # 实验3：模拟退火求解VRP
├── exp4_pso.py                    # 实验4：粒子群优化求解VRP
├── optimization.py                # NSGA-II算法实现
├── objective_functions.py         # 目标函数计算模块
├── constraints.py                 # 约束处理模块
├── data_preprocessing.py           # 数据预处理模块
├── model_training.py               # 模型训练模块
├── results_analysis.py            # 结果分析模块
└── *.png                          # 实验结果可视化图片
```

## 实验说明

### 实验1：遗传算法求解VRP (exp1.py)

**算法**: 遗传算法 (Genetic Algorithm, GA)

**问题描述**: 
- 1个供应点(S0)和10个需求点(D1-D10)
- 5辆不同规格的车辆，每辆车有容量、单位距离成本、最大行驶时间和速度限制
- 目标：最小化总运输成本
- 约束条件：
  - 车辆载重约束
  - 车辆时间约束
  - D1需求点必须在4小时内送达（优先级约束）

**算法参数**:
- 种群大小: 100
- 迭代次数: 500
- 交叉率: 0.8
- 变异率: 0.1
- 精英保留数量: 10

**输出结果**:
- 最优运输方案（每辆车的路径、载重、距离、时间、成本）
- 收敛曲线图 (`convergence_curve.png`)

### 实验2：禁忌搜索求解VRP (exp2_tabu.py)

**算法**: 禁忌搜索 (Tabu Search, TS)

**问题描述**: 同实验1

**算法参数**:
- 最大迭代次数: 500
- 禁忌表长度: 20
- 最大无改进次数: 120
- 邻域大小: 80

**邻域操作**:
- 交换操作 (swap): 交换两条路径中的客户
- 重定位操作 (relocate): 将一个客户从一条路径移动到另一条路径

**输出结果**:
- 最优运输方案
- 收敛曲线图 (`TS_convergence_curve.png`)
- 路径可视化图 (`TS_VRP_Results.png`)

### 实验3：模拟退火求解VRP (exp3_sa.py)

**算法**: 模拟退火 (Simulated Annealing, SA)

**问题描述**: 同实验1

**算法参数**:
- 最大迭代次数: 2000
- 初始温度: 800.0
- 终止温度: 1e-3
- 冷却系数: 0.995
- 邻域大小: 60

**邻域操作**: 同实验2

**输出结果**:
- 最优运输方案
- 收敛曲线图 (`SA_convergence_curve.png`)
- 路径可视化图 (`SA_VRP_Results.png`)

### 实验4：粒子群优化求解VRP (exp4_pso.py)

**算法**: 粒子群优化 (Particle Swarm Optimization, PSO)

**问题描述**: 同实验1

**算法参数**:
- 粒子群大小: 60
- 最大迭代次数: 400
- 惯性权重: 0.7 (动态调整范围: 0.4-0.9)
- 认知参数 c1: 1.5
- 社会参数 c2: 1.5
- 速度限制: 0.3
- 停滞迭代次数: 50

**编码方式**: 
- 粒子位置为实数向量，通过排序解码为客户访问顺序
- 使用启发式方法将客户分配到车辆

**输出结果**:
- 最优运输方案
- 收敛曲线图 (`PSO_convergence_curve.png`)
- 路径可视化图 (`PSO_VRP_Results.png`)

### 实验5：治安资源配置双目标优化 (main.py)

**算法**: NSGA-II (Non-dominated Sorting Genetic Algorithm II)

**问题描述**:
- 数据集: Communities and Crime数据集
- 双目标优化：
  - f1: 最小化资源分配后的预测误差均值
  - f2: 最小化选择的特征数量
- 约束条件：
  - 特征数量: 10 ≤ n_features ≤ 30
  - 资源分配比例: 0.5% ≤ ratio ≤ 5%
  - 模型R² ≥ 0.7

**算法参数**:
- 种群大小: 100
- 最大迭代次数: 50
- 交叉率: 0.9
- 变异率: 0.1

**决策变量**:
- 特征选择: 二进制向量（122维）
- 资源分配: 连续值向量（社区数量维）

**输出结果**:
- Pareto前沿解集
- Pareto前沿可视化图 (`results/pareto_front.png`)
- 资源分配分析图 (`results/resource_allocation.png`)
- 最优解的特征选择和资源分配结果（CSV文件）

## 环境要求

### Python版本
- Python 3.7+

### 依赖库
```
numpy >= 1.19.0
matplotlib >= 3.3.0
pandas >= 1.2.0
scikit-learn >= 0.24.0
scipy >= 1.6.0
```

### 安装依赖
```bash
pip install numpy matplotlib pandas scikit-learn scipy
```

### 数据文件（实验5）
实验5需要Communities and Crime数据集，数据文件路径应为：
```
communities+and+crime/communities.data
```

## 运行方法

### 实验1：遗传算法求解VRP
```bash
python exp1.py
```

### 实验2：禁忌搜索求解VRP
```bash
python exp2_tabu.py
```

### 实验3：模拟退火求解VRP
```bash
python exp3_sa.py
```

### 实验4：粒子群优化求解VRP
```bash
python exp4_pso.py
```

### 实验5：治安资源配置双目标优化
```bash
python main.py
```

**注意**: 实验5需要先准备数据集文件 `communities+and+crime/communities.data`

## 结果说明

### VRP实验结果（实验1-4）

每个实验会输出：
1. **控制台输出**: 
   - 迭代过程中的最佳成本
   - 最终最优运输方案（每辆车的详细路径信息）
   - 总运输成本

2. **可视化结果**:
   - 收敛曲线图：展示算法优化过程中目标函数值的变化
   - 路径可视化图（实验2-4）：展示车辆路径方案的空间分布

### 双目标优化结果（实验5）

实验5会输出：
1. **控制台输出**:
   - 优化过程中的统计信息（最佳f1、f2值，Pareto前沿大小等）
   - 最优解的特征选择结果
   - 资源分配统计信息

2. **可视化结果**:
   - `results/pareto_front.png`: Pareto前沿和优化过程可视化
   - `results/resource_allocation.png`: 资源分配分析图

3. **CSV结果文件**:
   - `results/selected_features.csv`: 选择的特征列表
   - `results/resource_allocation.csv`: 各社区的资源分配比例
   - `results/solution_summary.csv`: 解摘要信息

## 文件说明

### 核心算法文件

- `exp1.py`: 遗传算法实现，包含问题定义、GA算法和结果输出
- `exp2_tabu.py`: 禁忌搜索实现，包含TS算法和邻域操作
- `exp3_sa.py`: 模拟退火实现，包含SA算法和温度控制
- `exp4_pso.py`: 粒子群优化实现，包含PSO算法和粒子更新机制
- `main.py`: 主程序，调用NSGA-II进行双目标优化
- `optimization.py`: NSGA-II算法核心实现，包含非支配排序、拥挤距离计算等

### 辅助模块文件

- `objective_functions.py`: 目标函数计算（用于实验5）
- `constraints.py`: 约束处理和修复（用于实验5）
- `data_preprocessing.py`: 数据预处理（用于实验5）
- `model_training.py`: 梯度提升回归模型训练（用于实验5）
- `results_analysis.py`: 结果分析和可视化（用于实验5）

### 结果文件

- `*_convergence_curve.png`: 各算法的收敛曲线
- `*_VRP_Results.png`: VRP路径可视化结果
- `results/`: 实验5的结果目录，包含Pareto前沿、资源分配等可视化结果和CSV文件

## 算法特点对比

| 算法 | 优点 | 缺点 | 适用场景 |
|------|------|------|----------|
| 遗传算法(GA) | 全局搜索能力强，易于并行化 | 收敛速度较慢，参数敏感 | 复杂优化问题，多解空间 |
| 禁忌搜索(TS) | 避免局部最优，搜索效率高 | 需要合理设置禁忌表长度 | 组合优化问题 |
| 模拟退火(SA) | 实现简单，能跳出局部最优 | 收敛速度慢，参数调优困难 | 连续/离散优化问题 |
| 粒子群优化(PSO) | 收敛速度快，参数少 | 容易早熟收敛 | 连续优化问题 |
| NSGA-II | 多目标优化，Pareto前沿 | 计算复杂度高 | 多目标优化问题 |

## 注意事项

1. **随机种子**: 各实验都设置了随机种子以保证结果可复现
2. **中文字体**: 代码中使用了SimHei字体显示中文，如果系统没有该字体，可能需要修改matplotlib的字体设置
3. **计算时间**: 实验5（NSGA-II）可能需要较长的计算时间，请耐心等待
4. **数据文件**: 实验5需要Communities and Crime数据集，请确保数据文件路径正确

## 作者

控制算法课程实验项目

## 许可证

本项目仅用于学习和研究目的。

