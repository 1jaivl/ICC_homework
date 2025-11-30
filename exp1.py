#coding:utf-8
import numpy as np
import matplotlib.pyplot as plt
import random
from math import sqrt
import copy
from pylab import mpl
# 设置显示中文字体
mpl.rcParams["font.sans-serif"] = ["SimHei"]

# 问题数据定义
class ProblemData:
    def __init__(self):
        # 节点坐标 (供应点 + 10个需求点)
        self.coordinates = {
            'S0': (0, 0),
            'D1': (10, 8), 'D2': (15, 20), 'D3': (5, 12), 'D4': (20, 5),
            'D5': (8, 18), 'D6': (25, 15), 'D7': (12, 3), 'D8': (3, 22),
            'D9': (18, 25), 'D10': (7, 10)
        }

        # 需求量
        self.demands = {
            'D1': 25, 'D2': 18, 'D3': 30, 'D4': 12, 'D5': 22,
            'D6': 16, 'D7': 28, 'D8': 14, 'D9': 20, 'D10': 15
        }

        # 车辆信息
        self.vehicles = {
            'V1': {'capacity': 40, 'cost_per_km': 8, 'max_time': 6, 'speed': 60},
            'V2': {'capacity': 50, 'cost_per_km': 10, 'max_time': 7, 'speed': 60},
            'V3': {'capacity': 45, 'cost_per_km': 9, 'max_time': 6.5, 'speed': 60},
            'V4': {'capacity': 35, 'cost_per_km': 7, 'max_time': 5.5, 'speed': 60},
            'V5': {'capacity': 55, 'cost_per_km': 11, 'max_time': 7.5, 'speed': 60}
        }

class GeneticAlgorithmVRP:
    def __init__(self, problem_data, population_size=100, generations=500,
                 crossover_rate=0.8, mutation_rate=0.1, elite_size=10):
        self.data = problem_data
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size

        # 需求点列表（不包含供应点）
        self.demand_points = ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10']
        self.num_demands = len(self.demand_points)
        self.num_vehicles = 5

        # 预计算距离矩阵
        self.distance_matrix = self._calculate_distance_matrix()

    def _calculate_distance_matrix(self):
        """计算所有节点之间的距离矩阵"""
        nodes = ['S0'] + self.demand_points
        n = len(nodes)
        distance_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                if i != j:
                    x1, y1 = self.data.coordinates[nodes[i]]
                    x2, y2 = self.data.coordinates[nodes[j]]
                    distance_matrix[i][j] = sqrt((x2-x1)**2 + (y2-y1)**2)

        return distance_matrix

    def _calculate_route_distance(self, route):
        """计算单条路径的总距离"""
        if len(route) == 0:
            return 0

        total_distance = 0
        # 从供应点出发
        current_node = 0  # S0的索引

        for node in route:
            next_node = node + 1  # 需求点在距离矩阵中的索引
            total_distance += self.distance_matrix[current_node][next_node]
            current_node = next_node

        # 返回供应点
        total_distance += self.distance_matrix[current_node][0]

        return total_distance

    def _calculate_route_time(self, route, vehicle_speed):
        """计算单条路径的行驶时间"""
        distance = self._calculate_route_distance(route)
        return distance / vehicle_speed

    def _calculate_route_demand(self, route):
        """计算单条路径的总需求量"""
        total_demand = 0
        for node_idx in route:
            demand_point = self.demand_points[node_idx]
            total_demand += self.data.demands[demand_point]
        return total_demand

    def _is_route_feasible(self, route, vehicle_id):
        """检查单条路径是否满足约束条件"""
        if len(route) == 0:
            return True

        vehicle = self.data.vehicles[vehicle_id]

        # 检查载重约束
        total_demand = self._calculate_route_demand(route)
        if total_demand > vehicle['capacity']:
            return False

        # 检查时间约束
        route_time = self._calculate_route_time(route, vehicle['speed'])
        if route_time > vehicle['max_time']:
            return False

        # 检查D1的优先级约束
        if 'D1' in [self.demand_points[i] for i in route]:
            if route_time > 4:  # D1必须在4小时内送达
                return False

        return True

    def create_individual(self):
        """创建一个个体（解）"""
        # 创建所有需求点的列表
        all_points = list(range(self.num_demands))
        random.shuffle(all_points)

        # 为每辆车初始化空路径
        vehicle_routes = [[] for _ in range(self.num_vehicles)]
        vehicle_loads = [0 for _ in range(self.num_vehicles)]
        vehicle_capacities = [self.data.vehicles[f'V{i+1}']['capacity'] for i in range(self.num_vehicles)]

        # 分配需求点到车辆
        for point_idx in all_points:
            point_demand = self.data.demands[self.demand_points[point_idx]]

            # 找到可以承载该需求点的车辆
            feasible_vehicles = []
            for i in range(self.num_vehicles):
                if vehicle_loads[i] + point_demand <= vehicle_capacities[i]:
                    feasible_vehicles.append(i)

            if feasible_vehicles:
                # 随机选择一个可行车辆
                vehicle_idx = random.choice(feasible_vehicles)
            else:
                # 如果没有车辆可以承载，分配到负载最小的车辆
                vehicle_idx = np.argmin(vehicle_loads)

            vehicle_routes[vehicle_idx].append(point_idx)
            vehicle_loads[vehicle_idx] += point_demand

        # 将路径表示转换为个体表示
        individual = []
        for vehicle_idx, route in enumerate(vehicle_routes):
            for point_idx in route:
                individual.append((point_idx, vehicle_idx))

        return individual

    def decode_individual(self, individual):
        """将个体解码为车辆路径"""
        vehicle_routes = [[] for _ in range(self.num_vehicles)]

        for point_idx, vehicle_idx in individual:
            vehicle_routes[vehicle_idx].append(point_idx)

        return vehicle_routes

    def calculate_fitness(self, individual):
        """计算个体的适应度（总成本）"""
        vehicle_routes = self.decode_individual(individual)
        total_cost = 0
        penalty = 0

        for i, route in enumerate(vehicle_routes):
            vehicle_id = f'V{i+1}'
            vehicle = self.data.vehicles[vehicle_id]

            # 检查可行性
            if not self._is_route_feasible(route, vehicle_id):
                penalty += 10000  # 不可行解的惩罚

            # 计算成本
            route_distance = self._calculate_route_distance(route)
            route_cost = route_distance * vehicle['cost_per_km']
            total_cost += route_cost

        return total_cost + penalty

    def selection(self, population, fitnesses):
        """锦标赛选择"""
        tournament_size = 3
        selected = []

        for _ in range(self.population_size):
            tournament_indices = random.sample(range(len(population)), min(tournament_size, len(population)))
            tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
            winner_idx = tournament_indices[np.argmin(tournament_fitnesses)]
            selected.append(population[winner_idx])

        return selected

    def crossover(self, parent1, parent2):
        """顺序交叉"""
        if random.random() > self.crossover_rate:
            return parent1, parent2

        # 选择交叉点
        crossover_point = random.randint(1, len(parent1) - 2)

        child1 = parent1[:crossover_point]
        child2 = parent2[:crossover_point]

        # 从另一个父代补充缺失的点
        for point_vehicle in parent1[crossover_point:]:
            point_idx, _ = point_vehicle
            if point_idx not in [p[0] for p in child2]:
                child2.append(point_vehicle)

        for point_vehicle in parent2[crossover_point:]:
            point_idx, _ = point_vehicle
            if point_idx not in [p[0] for p in child1]:
                child1.append(point_vehicle)

        return child1, child2

    def mutation(self, individual):
        """变异操作"""
        if random.random() > self.mutation_rate or len(individual) < 2:
            return individual

        mutated = individual.copy()

        # 选择两个点进行交换
        idx1, idx2 = random.sample(range(len(mutated)), 2)
        mutated[idx1], mutated[idx2] = mutated[idx2], mutated[idx1]

        return mutated

    def optimize(self):
        """主优化函数"""
        # 初始化种群
        population = [self.create_individual() for _ in range(self.population_size)]
        best_individual = None
        best_fitness = float('inf')
        fitness_history = []

        for generation in range(self.generations):
            # 计算适应度
            fitnesses = [self.calculate_fitness(ind) for ind in population]

            # 更新最佳解
            current_best_fitness = min(fitnesses)
            current_best_idx = np.argmin(fitnesses)

            if current_best_fitness < best_fitness:
                best_fitness = current_best_fitness
                best_individual = population[current_best_idx].copy()

            fitness_history.append(best_fitness)

            # 选择
            selected = self.selection(population, fitnesses)

            # 交叉
            offspring = []
            for i in range(0, len(selected), 2):
                if i + 1 < len(selected):
                    child1, child2 = self.crossover(selected[i], selected[i+1])
                    offspring.extend([child1, child2])
                else:
                    offspring.append(selected[i])

            # 变异
            mutated_offspring = []
            for ind in offspring:
                # 确保个体长度正确
                if len(ind) == self.num_demands:
                    mutated_offspring.append(self.mutation(ind))
                else:
                    # 如果个体长度不正确，重新生成
                    mutated_offspring.append(self.create_individual())

            # 精英保留
            if self.elite_size > 0:
                # 选择精英个体
                elite_indices = np.argsort(fitnesses)[:self.elite_size]
                elites = [population[i] for i in elite_indices]

                # 用精英替换最差的个体
                mutated_offspring = elites + mutated_offspring[:-self.elite_size]

            population = mutated_offspring

            if generation % 50 == 0:
                print(f"Generation {generation}, Best Fitness: {best_fitness:.2f}")

        return best_individual, best_fitness, fitness_history

    def print_solution(self, best_individual):
        """打印最优解"""
        vehicle_routes = self.decode_individual(best_individual)
        total_cost = 0

        print("最优运输方案：")
        print("=" * 50)

        for i, route in enumerate(vehicle_routes):
            vehicle_id = f'V{i+1}'
            vehicle = self.data.vehicles[vehicle_id]

            if route:
                route_points = ['S0'] + [self.demand_points[idx] for idx in route] + ['S0']
                route_str = ' -> '.join(route_points)

                total_demand = self._calculate_route_demand(route)
                route_distance = self._calculate_route_distance(route)
                route_time = self._calculate_route_time(route, vehicle['speed'])
                route_cost = route_distance * vehicle['cost_per_km']
                total_cost += route_cost

                print(f"车辆 {vehicle_id}:")
                print(f"  路径: {route_str}")
                print(f"  总需求量: {total_demand}箱 (容量: {vehicle['capacity']}箱)")
                print(f"  行驶距离: {route_distance:.2f}km")
                print(f"  行驶时间: {route_time:.2f}h (限制: {vehicle['max_time']}h)")
                print(f"  运输成本: {route_cost:.2f}元")

                # 检查D1约束
                if 'D1' in route_points:
                    if route_time <= 4:
                        print(f"  ✓ D1在4小时内送达")
                    else:
                        print(f"  ✗ D1送达时间超过4小时限制")
                print()
            else:
                print(f"车辆 {vehicle_id}: 未使用")
                print()

        print(f"总运输成本: {total_cost:.2f}元")
        return total_cost

# 运行遗传算法
def main():
    # 设置随机种子以便复现结果
    random.seed(42)
    np.random.seed(42)

    # 创建问题实例
    problem_data = ProblemData()

    # 创建遗传算法求解器
    ga_solver = GeneticAlgorithmVRP(
        problem_data=problem_data,
        population_size=100,
        generations=500,
        crossover_rate=0.8,
        mutation_rate=0.1,
        elite_size=10
    )

    print("开始遗传算法优化...")
    best_individual, best_fitness, fitness_history = ga_solver.optimize()

    print("\n优化完成！")
    print("=" * 50)

    # 输出最优解
    total_cost = ga_solver.print_solution(best_individual)

    # 绘制收敛曲线
    plt.figure(figsize=(10, 6))
    plt.plot(fitness_history)
    plt.title('遗传算法收敛曲线')
    plt.xlabel('迭代次数')
    plt.ylabel('总运输成本 (元)')
    plt.grid(True)
    plt.savefig('convergence_curve.png')
    plt.show()

    return best_individual, total_cost

if __name__ == "__main__":
    best_solution, final_cost = main()
