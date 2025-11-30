"""
NSGA-II多目标优化算法实现
"""
import numpy as np
from scipy.spatial.distance import cdist
from objective_functions import compute_objectives
from constraints import repair_individual, check_feature_count_constraint, check_resource_allocation_constraint, check_r2_constraint


class NSGA2:
    """
    NSGA-II多目标优化算法
    """
    
    def __init__(self, n_features=122, n_communities=None, 
                 min_features=10, max_features=30,
                 min_ratio=0.005, max_ratio=0.05,
                 pop_size=100, max_gen=100,
                 crossover_rate=0.9, mutation_rate=0.1):
        """
        初始化NSGA-II算法
        
        Parameters:
        -----------
        n_features : int
            特征数量
        n_communities : int, optional
            社区数量（如果为None，将在run方法中根据测试集大小确定）
        min_features : int
            最小特征数量
        max_features : int
            最大特征数量
        min_ratio : float
            最小资源分配比例
        max_ratio : float
            最大资源分配比例
        pop_size : int
            种群大小
        max_gen : int
            最大迭代次数
        crossover_rate : float
            交叉概率
        mutation_rate : float
            变异概率
        """
        self.n_features = n_features
        self.n_communities = n_communities
        self.min_features = min_features
        self.max_features = max_features
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.pop_size = pop_size
        self.max_gen = max_gen
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        
        # 决策变量维度将在run方法中确定
        self.n_vars = None
        
    def initialize_population(self):
        """
        初始化种群（增加多样性）
        
        Returns:
        --------
        population : array
            初始种群 (pop_size, n_vars)
        """
        population = np.zeros((self.pop_size, self.n_vars))
        
        for i in range(self.pop_size):
            # 特征选择部分：随机选择10-30个特征
            n_selected = np.random.randint(self.min_features, self.max_features + 1)
            feature_indices = np.random.choice(self.n_features, n_selected, replace=False)
            population[i, feature_indices] = 1
            
            # 资源分配部分：使用不同的初始化策略增加多样性
            if i < self.pop_size // 3:
                # 策略1：均匀分配
                resource_allocation = np.ones(self.n_communities) / self.n_communities
            elif i < 2 * self.pop_size // 3:
                # 策略2：随机生成，然后归一化
                resource_allocation = np.random.uniform(self.min_ratio, self.max_ratio, self.n_communities)
                resource_allocation = resource_allocation / resource_allocation.sum()
            else:
                # 策略3：偏向某些社区（模拟高风险区域）
                resource_allocation = np.random.exponential(scale=0.01, size=self.n_communities)
                resource_allocation = np.clip(resource_allocation, self.min_ratio, self.max_ratio)
                resource_allocation = resource_allocation / resource_allocation.sum()
            
            population[i, self.n_features:] = resource_allocation
        
        return population
    
    def binary_tournament_selection(self, fitness, n_selections):
        """
        二进制锦标赛选择
        
        Parameters:
        -----------
        fitness : array
            适应度值（越小越好）
        n_selections : int
            选择数量
        
        Returns:
        --------
        selected_indices : array
            选中的个体索引
        """
        selected = []
        for _ in range(n_selections):
            # 随机选择两个个体
            idx1, idx2 = np.random.choice(len(fitness), 2, replace=False)
            # 选择适应度更好的（更小的）
            if fitness[idx1] < fitness[idx2]:
                selected.append(idx1)
            else:
                selected.append(idx2)
        return np.array(selected)
    
    def crossover(self, parent1, parent2):
        """
        交叉操作
        
        Parameters:
        -----------
        parent1 : array
            父代1
        parent2 : array
            父代2
        
        Returns:
        --------
        child1 : array
            子代1
        child2 : array
            子代2
        """
        if np.random.rand() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        child1 = parent1.copy()
        child2 = parent2.copy()
        
        # 特征选择部分：单点交叉
        crossover_point = np.random.randint(1, self.n_features)
        child1[:crossover_point] = parent2[:crossover_point]
        child2[:crossover_point] = parent1[:crossover_point]
        
        # 资源分配部分：模拟二进制交叉（SBX）
        eta_c = 20  # 分布指数
        u = np.random.rand(self.n_communities)
        beta = np.where(u <= 0.5,
                        (2 * u) ** (1 / (eta_c + 1)),
                        (1 / (2 * (1 - u))) ** (1 / (eta_c + 1)))
        
        child1_res = 0.5 * ((1 + beta) * parent1[self.n_features:] + 
                           (1 - beta) * parent2[self.n_features:])
        child2_res = 0.5 * ((1 - beta) * parent1[self.n_features:] + 
                           (1 + beta) * parent2[self.n_features:])
        
        child1[self.n_features:] = child1_res
        child2[self.n_features:] = child2_res
        
        # 修复子代
        child1 = self.repair_individual(child1)
        child2 = self.repair_individual(child2)
        
        return child1, child2
    
    def mutate(self, individual):
        """
        变异操作
        
        Parameters:
        -----------
        individual : array
            个体
        
        Returns:
        --------
        mutated : array
            变异后的个体
        """
        mutated = individual.copy()
        
        # 特征选择部分：位翻转变异
        for i in range(self.n_features):
            if np.random.rand() < self.mutation_rate:
                mutated[i] = 1 - mutated[i]
        
        # 资源分配部分：多项式变异
        eta_m = 20  # 分布指数
        for i in range(self.n_communities):
            if np.random.rand() < self.mutation_rate:
                u = np.random.rand()
                delta = np.where(u < 0.5,
                                (2 * u) ** (1 / (eta_m + 1)) - 1,
                                1 - (2 * (1 - u)) ** (1 / (eta_m + 1)))
                mutated[self.n_features + i] += delta * (self.max_ratio - self.min_ratio)
                mutated[self.n_features + i] = np.clip(mutated[self.n_features + i],
                                                       self.min_ratio, self.max_ratio)
        
        # 修复个体
        mutated = self.repair_individual(mutated)
        
        return mutated
    
    def repair_individual(self, individual):
        """
        修复个体，使其满足约束（使用类属性）
        
        Parameters:
        -----------
        individual : array
            个体向量 [前n_features位：特征选择, 后n_communities位：资源分配]
        
        Returns:
        --------
        repaired_individual : array
            修复后的个体
        """
        repaired = individual.copy()
        
        # 修复特征选择部分
        feature_mask = repaired[:self.n_features]
        n_features = int(np.sum(feature_mask))
        
        if n_features < self.min_features:
            # 如果特征太少，随机添加特征
            zero_indices = np.where(feature_mask == 0)[0]
            n_to_add = self.min_features - n_features
            if len(zero_indices) >= n_to_add:
                selected = np.random.choice(zero_indices, n_to_add, replace=False)
                feature_mask[selected] = 1
        elif n_features > self.max_features:
            # 如果特征太多，随机删除特征
            one_indices = np.where(feature_mask == 1)[0]
            n_to_remove = n_features - self.max_features
            if len(one_indices) >= n_to_remove:
                selected = np.random.choice(one_indices, n_to_remove, replace=False)
                feature_mask[selected] = 0
        
        repaired[:self.n_features] = feature_mask
        
        # 修复资源分配部分
        resource_allocation = repaired[self.n_features:self.n_features+self.n_communities]
        
        # 归一化
        total = np.sum(resource_allocation)
        if total == 0:
            resource_allocation = np.ones(self.n_communities) / self.n_communities
        else:
            resource_allocation = resource_allocation / total
        
        # 裁剪到有效范围
        resource_allocation = np.clip(resource_allocation, self.min_ratio, self.max_ratio)
        
        # 重新归一化
        total = np.sum(resource_allocation)
        if total != 1.0:
            # 如果归一化后总和不为1，需要调整
            # 策略：按比例缩放，然后裁剪，重复直到收敛
            max_iter = 100
            for _ in range(max_iter):
                resource_allocation = resource_allocation / total
                resource_allocation = np.clip(resource_allocation, self.min_ratio, self.max_ratio)
                total = np.sum(resource_allocation)
                if abs(total - 1.0) < 1e-6:
                    break
            
            # 如果仍未归一化，进行微调
            if abs(total - 1.0) > 1e-6:
                diff = 1.0 - total
                # 将差值均匀分配到所有元素（在范围内）
                adjustment = diff / len(resource_allocation)
                resource_allocation = resource_allocation + adjustment
                resource_allocation = np.clip(resource_allocation, self.min_ratio, self.max_ratio)
        
        repaired[self.n_features:self.n_features+self.n_communities] = resource_allocation
        
        return repaired
    
    def fast_non_dominated_sort(self, objectives):
        """
        快速非支配排序（使用归一化的目标函数值以避免尺度问题）
        
        Parameters:
        -----------
        objectives : array
            目标函数值矩阵 (n_pop, n_obj)
        
        Returns:
        --------
        fronts : list
            前沿列表，每个前沿包含个体索引
        """
        n_pop = len(objectives)
        
        # 归一化目标函数值以避免尺度问题
        normalized_obj = objectives.copy()
        for obj_idx in range(objectives.shape[1]):
            obj_range = objectives[:, obj_idx].max() - objectives[:, obj_idx].min()
            if obj_range > 1e-10:
                normalized_obj[:, obj_idx] = (objectives[:, obj_idx] - objectives[:, obj_idx].min()) / obj_range
        
        dominated_by = [[] for _ in range(n_pop)]
        domination_count = np.zeros(n_pop, dtype=int)
        
        # 计算支配关系（使用归一化后的值）
        for i in range(n_pop):
            for j in range(n_pop):
                if i == j:
                    continue
                # i支配j：i的所有目标都不大于j，且至少有一个目标小于j
                if np.all(normalized_obj[i] <= normalized_obj[j]) and np.any(normalized_obj[i] < normalized_obj[j]):
                    dominated_by[i].append(j)
                    domination_count[j] += 1
        
        # 构建前沿
        fronts = []
        current_front = np.where(domination_count == 0)[0].tolist()
        
        while len(current_front) > 0:
            fronts.append(current_front)
            next_front = []
            
            for i in current_front:
                for j in dominated_by[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        next_front.append(j)
            
            current_front = next_front
        
        return fronts
    
    def crowding_distance(self, objectives, front):
        """
        计算拥挤距离（使用归一化的目标函数值）
        
        Parameters:
        -----------
        objectives : array
            目标函数值矩阵
        front : list
            前沿中的个体索引
        
        Returns:
        --------
        distances : array
            拥挤距离
        """
        n_obj = objectives.shape[1]
        n_front = len(front)
        distances = np.zeros(n_front)
        
        if n_front <= 2:
            distances[:] = np.inf
            return distances
        
        # 归一化目标函数值
        normalized_obj = np.zeros((n_front, n_obj))
        for obj_idx in range(n_obj):
            obj_values = [objectives[front[i], obj_idx] for i in range(n_front)]
            obj_min = min(obj_values)
            obj_max = max(obj_values)
            obj_range = obj_max - obj_min
            if obj_range > 1e-10:
                normalized_obj[:, obj_idx] = [(v - obj_min) / obj_range for v in obj_values]
            else:
                normalized_obj[:, obj_idx] = 0
        
        for obj_idx in range(n_obj):
            # 按当前目标函数值排序
            sorted_indices = np.argsort([objectives[front[i], obj_idx] for i in range(n_front)])
            
            # 边界个体距离设为无穷大
            distances[sorted_indices[0]] = np.inf
            distances[sorted_indices[-1]] = np.inf
            
            # 计算中间个体的距离（使用归一化后的值）
            if n_front > 2:
                for i in range(1, n_front - 1):
                    idx = sorted_indices[i]
                    prev_idx = sorted_indices[i - 1]
                    next_idx = sorted_indices[i + 1]
                    # 使用归一化后的值计算距离
                    distances[idx] += abs(normalized_obj[next_idx, obj_idx] - 
                                          normalized_obj[prev_idx, obj_idx])
        
        return distances
    
    def select_next_generation(self, population, objectives, constraints):
        """
        选择下一代种群（改进版：即使没有可行解也能继续优化）
        
        Parameters:
        -----------
        population : array
            当前种群
        objectives : array
            目标函数值
        constraints : array
            约束违反度
        
        Returns:
        --------
        selected_population : array
            选中的种群
        """
        # 计算约束违反度总和
        constraint_violation = np.sum(constraints, axis=1)
        
        # 如果所有解都不满足约束，使用惩罚函数方法
        if np.all(constraint_violation > 0):
            # 使用惩罚函数：目标函数值 + 惩罚项
            penalty_weight = 1000.0  # 惩罚权重
            penalized_obj = objectives.copy()
            penalized_obj[:, 0] += penalty_weight * constraint_violation
            penalized_obj[:, 1] += penalty_weight * constraint_violation
            
            # 对惩罚后的目标函数进行非支配排序
            fronts = self.fast_non_dominated_sort(penalized_obj)
        else:
            # 正常情况：使用原始目标函数
            fronts = self.fast_non_dominated_sort(objectives)
        
        selected = []
        remaining = self.pop_size
        
        for front in fronts:
            if len(selected) + len(front) <= self.pop_size:
                # 如果整个前沿都能加入
                selected.extend(front)
            else:
                # 需要从当前前沿中选择
                if len(selected) < self.pop_size:
                    # 计算拥挤距离
                    distances = self.crowding_distance(objectives, front)
                    
                    # 按拥挤距离和约束违反度排序
                    # 优先选择约束满足的，然后按拥挤距离排序
                    front_array = np.array(front)
                    feasible_mask = constraint_violation[front_array] == 0
                    
                    # 先选择满足约束的
                    feasible_indices = front_array[feasible_mask]
                    infeasible_indices = front_array[~feasible_mask]
                    
                    # 对满足约束的按拥挤距离排序
                    if len(feasible_indices) > 0:
                        feasible_distances = distances[feasible_mask]
                        sorted_feasible = feasible_indices[np.argsort(-feasible_distances)]
                        selected.extend(sorted_feasible[:remaining].tolist())
                        remaining -= len(sorted_feasible[:remaining])
                    
                    # 如果还需要，从不满足约束的中选择违反度最小的
                    if remaining > 0 and len(infeasible_indices) > 0:
                        infeasible_violations = constraint_violation[infeasible_indices]
                        # 同时考虑拥挤距离和约束违反度
                        infeasible_distances = distances[~feasible_mask]
                        # 使用加权和：优先选择约束违反度小且拥挤距离大的
                        combined_score = -infeasible_violations + 0.1 * infeasible_distances
                        sorted_infeasible = infeasible_indices[np.argsort(-combined_score)]
                        selected.extend(sorted_infeasible[:remaining].tolist())
                
                break
        
        return population[selected[:self.pop_size]]
    
    def evaluate_population(self, population, X_train, y_train, X_test, y_test):
        """
        评估种群
        
        Parameters:
        -----------
        population : array
            种群
        X_train : array
            训练集特征
        y_train : array
            训练集目标
        X_test : array
            测试集特征
        y_test : array
            测试集目标
        
        Returns:
        --------
        objectives : array
            目标函数值矩阵
        constraints : array
            约束违反度矩阵
        """
        n_pop = len(population)
        objectives = np.zeros((n_pop, 2))
        constraints = np.zeros((n_pop, 3))
        
        for i, individual in enumerate(population):
            feature_mask = individual[:self.n_features].astype(int)
            resource_allocation = individual[self.n_features:self.n_features+self.n_communities]
            
            # 计算目标函数
            f1, f2, r2, _ = compute_objectives(
                X_train, y_train, X_test, y_test, feature_mask, resource_allocation
            )
            
            objectives[i, 0] = f1
            objectives[i, 1] = f2
            
            # 计算约束违反度
            n_features = int(np.sum(feature_mask))
            constraints[i, 0] = max(0, self.min_features - n_features)  # 特征数量下界
            constraints[i, 1] = max(0, n_features - self.max_features)  # 特征数量上界
            constraints[i, 2] = max(0, 0.7 - r2)  # R²约束
        
        return objectives, constraints
    
    def run(self, X_train, y_train, X_test, y_test, verbose=True):
        """
        运行NSGA-II算法
        
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
        verbose : bool
            是否打印进度
        
        Returns:
        --------
        pareto_front : array
            Pareto前沿解
        pareto_objectives : array
            Pareto前沿目标函数值
        history : dict
            优化历史
        """
        # 根据测试集大小确定社区数量
        if self.n_communities is None:
            self.n_communities = len(y_test)
        
        # 确定决策变量维度
        self.n_vars = self.n_features + self.n_communities
        
        # 初始化种群
        population = self.initialize_population()
        
        # 评估初始种群
        objectives, constraints = self.evaluate_population(
            population, X_train, y_train, X_test, y_test
        )
        
        history = {
            'generations': [],
            'best_f1': [],           # 最佳f1（最小）
            'best_f2': [],             # 最佳f2（最小）
            'mean_f1': [],            # Pareto前沿平均f1
            'mean_f2': [],            # Pareto前沿平均f2
            'median_f1': [],          # Pareto前沿中位数f1
            'median_f2': [],          # Pareto前沿中位数f2
            'pareto_size': [],        # Pareto前沿大小
            'feasible_ratio': []      # 满足约束的解的比例
        }
        
        # 记录初始代
        fronts = self.fast_non_dominated_sort(objectives)
        pareto_indices = fronts[0] if len(fronts) > 0 else []
        feasible_mask = np.sum(constraints, axis=1) == 0
        feasible_ratio = np.sum(feasible_mask) / len(constraints)
        
        if np.any(feasible_mask):
            feasible_obj = objectives[feasible_mask]
            history['best_f1'].append(np.min(feasible_obj[:, 0]))
            history['best_f2'].append(np.min(feasible_obj[:, 1]))
        else:
            history['best_f1'].append(np.min(objectives[:, 0]))
            history['best_f2'].append(np.min(objectives[:, 1]))
        
        if len(pareto_indices) > 0:
            pareto_obj = objectives[pareto_indices]
            history['mean_f1'].append(np.mean(pareto_obj[:, 0]))
            history['mean_f2'].append(np.mean(pareto_obj[:, 1]))
            history['median_f1'].append(np.median(pareto_obj[:, 0]))
            history['median_f2'].append(np.median(pareto_obj[:, 1]))
        else:
            history['mean_f1'].append(np.mean(objectives[:, 0]))
            history['mean_f2'].append(np.mean(objectives[:, 1]))
            history['median_f1'].append(np.median(objectives[:, 0]))
            history['median_f2'].append(np.median(objectives[:, 1]))
        
        history['pareto_size'].append(len(pareto_indices))
        history['feasible_ratio'].append(feasible_ratio)
        history['generations'].append(0)
        
        # 添加R²统计信息
        r2_values = []
        for i in range(len(population)):
            feature_mask = population[i][:self.n_features].astype(int)
            resource_allocation = population[i][self.n_features:self.n_features+self.n_communities]
            _, _, r2, _ = compute_objectives(X_train, y_train, X_test, y_test, 
                                            feature_mask, resource_allocation)
            r2_values.append(r2)
        r2_values = np.array(r2_values)
        
        if verbose:
            print(f"Gen   0/{self.max_gen}: "
                  f"Best f1={history['best_f1'][-1]:.6f}, "
                  f"Best f2={history['best_f2'][-1]:2.0f}, "
                  f"Mean f1={history['mean_f1'][-1]:.6f}, "
                  f"Mean f2={history['mean_f2'][-1]:.1f}, "
                  f"Pareto={len(pareto_indices):2d}, "
                  f"Feasible={feasible_ratio:.1%}, "
                  f"R²: max={r2_values.max():.3f}, mean={r2_values.mean():.3f}, min={r2_values.min():.3f}")
        
        # 主循环
        for gen in range(self.max_gen):
            # 改进的父代选择：考虑约束违反度
            constraint_violation = np.sum(constraints, axis=1)
            
            # 归一化目标函数值用于选择（避免尺度问题）
            f1_range = objectives[:, 0].max() - objectives[:, 0].min()
            f2_range = objectives[:, 1].max() - objectives[:, 1].min()
            
            # 归一化（避免除零）
            if f1_range > 1e-10:
                f1_norm = (objectives[:, 0] - objectives[:, 0].min()) / f1_range
            else:
                f1_norm = np.zeros(len(objectives))
            
            if f2_range > 1e-10:
                f2_norm = (objectives[:, 1] - objectives[:, 1].min()) / f2_range
            else:
                f2_norm = np.zeros(len(objectives))
            
            # 归一化约束违反度
            cv_range = constraint_violation.max() - constraint_violation.min()
            if cv_range > 1e-10:
                cv_norm = (constraint_violation - constraint_violation.min()) / cv_range
            else:
                cv_norm = np.zeros(len(constraint_violation))
            
            # 使用归一化后的值进行选择
            # 优先选择约束违反度小的，然后考虑目标函数值
            combined_fitness = 0.8 * f1_norm + 0.1 * f2_norm + 0.1 * cv_norm
            
            # 选择父代（使用归一化的适应度，越小越好）
            parent_indices = self.binary_tournament_selection(
                combined_fitness, self.pop_size
            )
            parents = population[parent_indices]
            
            # 生成子代
            offspring = []
            for i in range(0, len(parents) - 1, 2):
                child1, child2 = self.crossover(parents[i], parents[i + 1])
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                offspring.extend([child1, child2])
            
            # 如果子代数量不足，补充
            while len(offspring) < self.pop_size:
                idx = np.random.randint(len(parents))
                offspring.append(self.mutate(parents[idx].copy()))
            
            offspring = np.array(offspring[:self.pop_size])
            
            # 评估子代
            offspring_obj, offspring_const = self.evaluate_population(
                offspring, X_train, y_train, X_test, y_test
            )
            
            # 合并父代和子代
            combined_pop = np.vstack([population, offspring])
            combined_obj = np.vstack([objectives, offspring_obj])
            combined_const = np.vstack([constraints, offspring_const])
            
            # 选择下一代
            population = self.select_next_generation(
                combined_pop, combined_obj, combined_const
            )
            objectives, constraints = self.evaluate_population(
                population, X_train, y_train, X_test, y_test
            )
            
            # 计算Pareto前沿
            fronts = self.fast_non_dominated_sort(objectives)
            pareto_indices = fronts[0] if len(fronts) > 0 else []
            
            # 记录历史
            feasible_mask = np.sum(constraints, axis=1) == 0
            feasible_ratio = np.sum(feasible_mask) / len(constraints)
            history['feasible_ratio'].append(feasible_ratio)
            
            # 记录最佳值（从满足约束的解中选择）
            if np.any(feasible_mask):
                feasible_obj = objectives[feasible_mask]
                history['best_f1'].append(np.min(feasible_obj[:, 0]))
                history['best_f2'].append(np.min(feasible_obj[:, 1]))
            else:
                # 如果没有满足约束的解，记录所有解中的最佳值
                history['best_f1'].append(np.min(objectives[:, 0]))
                history['best_f2'].append(np.min(objectives[:, 1]))
            
            # 记录Pareto前沿的统计信息
            if len(pareto_indices) > 0:
                pareto_obj = objectives[pareto_indices]
                history['mean_f1'].append(np.mean(pareto_obj[:, 0]))
                history['mean_f2'].append(np.mean(pareto_obj[:, 1]))
                history['median_f1'].append(np.median(pareto_obj[:, 0]))
                history['median_f2'].append(np.median(pareto_obj[:, 1]))
            else:
                history['mean_f1'].append(np.mean(objectives[:, 0]))
                history['mean_f2'].append(np.mean(objectives[:, 1]))
                history['median_f1'].append(np.median(objectives[:, 0]))
                history['median_f2'].append(np.median(objectives[:, 1]))
            
            history['pareto_size'].append(len(pareto_indices))
            history['generations'].append(gen)
            
            if verbose:
                if gen == 0 or (gen + 1) % 5 == 0 or gen == self.max_gen - 1:
                    # 计算f1和f2的改进
                    f1_improvement = ""
                    f2_improvement = ""
                    if gen > 0:
                        f1_change = history['best_f1'][-1] - history['best_f1'][-2]
                        f2_change = history['best_f2'][-1] - history['best_f2'][-2]
                        f1_improvement = f" (Δ{f1_change:+.6f})" if abs(f1_change) > 1e-8 else ""
                        f2_improvement = f" (Δ{f2_change:+.1f})" if abs(f2_change) > 0.1 else ""
                    
                    # 计算R²统计
                    r2_values = []
                    for i in range(len(population)):
                        feature_mask = population[i][:self.n_features].astype(int)
                        resource_allocation = population[i][self.n_features:self.n_features+self.n_communities]
                        _, _, r2, _ = compute_objectives(X_train, y_train, X_test, y_test, 
                                                        feature_mask, resource_allocation)
                        r2_values.append(r2)
                    r2_values = np.array(r2_values)
                    
                    print(f"Gen {gen + 1:3d}/{self.max_gen}: "
                          f"Best f1={history['best_f1'][-1]:.6f}{f1_improvement}, "
                          f"Best f2={history['best_f2'][-1]:2.0f}{f2_improvement}, "
                          f"Mean f1={history['mean_f1'][-1]:.6f}, "
                          f"Mean f2={history['mean_f2'][-1]:.1f}, "
                          f"Pareto={len(pareto_indices):2d}, "
                          f"Feasible={feasible_ratio:.1%}, "
                          f"R²: max={r2_values.max():.3f}, mean={r2_values.mean():.3f}")
        
        # 提取最终Pareto前沿
        fronts = self.fast_non_dominated_sort(objectives)
        pareto_indices = fronts[0] if len(fronts) > 0 else []
        
        pareto_front = population[pareto_indices]
        pareto_objectives = objectives[pareto_indices]
        
        return pareto_front, pareto_objectives, history

