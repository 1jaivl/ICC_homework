#coding:utf-8
import numpy as np
import matplotlib.pyplot as plt
import random
from math import sqrt
from pylab import mpl

mpl.rcParams["font.sans-serif"] = ["SimHei"]


class ProblemData:
    def __init__(self):
        self.coordinates = {
            'S0': (0, 0),
            'D1': (10, 8), 'D2': (15, 20), 'D3': (5, 12), 'D4': (20, 5),
            'D5': (8, 18), 'D6': (25, 15), 'D7': (12, 3), 'D8': (3, 22),
            'D9': (18, 25), 'D10': (7, 10)
        }
        self.demands = {
            'D1': 25, 'D2': 18, 'D3': 30, 'D4': 12, 'D5': 22,
            'D6': 16, 'D7': 28, 'D8': 14, 'D9': 20, 'D10': 15
        }
        self.vehicles = {
            'V1': {'capacity': 40, 'cost_per_km': 8, 'max_time': 6, 'speed': 60},
            'V2': {'capacity': 50, 'cost_per_km': 10, 'max_time': 7, 'speed': 60},
            'V3': {'capacity': 45, 'cost_per_km': 9, 'max_time': 6.5, 'speed': 60},
            'V4': {'capacity': 35, 'cost_per_km': 7, 'max_time': 5.5, 'speed': 60},
            'V5': {'capacity': 55, 'cost_per_km': 11, 'max_time': 7.5, 'speed': 60}
        }


class SimulatedAnnealingVRP:
    def __init__(self, problem_data,
                 max_iters=3000,
                 init_temp=1000.0,
                 final_temp=1e-3,
                 alpha=0.995,
                 neighborhood_size=60,
                 penalty_capacity=10000,
                 penalty_time=10000,
                 penalty_priority=10000):
        self.data = problem_data
        self.demand_points = ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10']
        self.num_demands = len(self.demand_points)
        self.num_vehicles = 5

        self.max_iters = max_iters
        self.init_temp = init_temp
        self.final_temp = final_temp
        self.alpha = alpha
        self.neighborhood_size = neighborhood_size

        self.penalty_capacity = penalty_capacity
        self.penalty_time = penalty_time
        self.penalty_priority = penalty_priority

        self.distance_matrix = self._calculate_distance_matrix()

    def _calculate_distance_matrix(self):
        nodes = ['S0'] + self.demand_points
        n = len(nodes)
        dist = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    x1, y1 = self.data.coordinates[nodes[i]]
                    x2, y2 = self.data.coordinates[nodes[j]]
                    dist[i, j] = sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        return dist

    def _route_distance(self, route):
        if not route:
            return 0.0
        total = 0.0
        current = 0
        for idx in route:
            nxt = idx + 1
            total += self.distance_matrix[current, nxt]
            current = nxt
        total += self.distance_matrix[current, 0]
        return total

    def _route_time(self, route, speed):
        return self._route_distance(route) / speed

    def _route_demand(self, route):
        total = 0
        for idx in route:
            total += self.data.demands[self.demand_points[idx]]
        return total

    def _evaluate(self, solution):
        total_cost = 0.0
        penalty = 0.0
        assigned = set()

        for i, route in enumerate(solution):
            vid = f'V{i + 1}'
            v = self.data.vehicles[vid]
            load = self._route_demand(route)
            t = self._route_time(route, v['speed']) if route else 0.0

            if load > v['capacity']:
                penalty += self.penalty_capacity * (load - v['capacity'])
            if t > v['max_time']:
                penalty += self.penalty_time * (t - v['max_time'])
            if 0 in route and t > 4:
                penalty += self.penalty_priority * (t - 4)

            dist = self._route_distance(route)
            total_cost += dist * v['cost_per_km']

            for c in route:
                if c in assigned:
                    penalty += self.penalty_capacity
                assigned.add(c)

        if len(assigned) != self.num_demands:
            missing = self.num_demands - len(assigned)
            if missing > 0:
                penalty += self.penalty_capacity * missing

        return total_cost + penalty

    def _initial_solution(self):
        customers = list(range(self.num_demands))
        random.shuffle(customers)
        routes = [[] for _ in range(self.num_vehicles)]
        loads = [0] * self.num_vehicles

        for c in customers:
            demand_c = self.data.demands[self.demand_points[c]]
            best_v = None
            best_increase = float('inf')
            for v_idx in range(self.num_vehicles):
                vid = f'V{v_idx + 1}'
                trial_route = routes[v_idx] + [c]
                load = loads[v_idx] + demand_c
                dist_before = self._route_distance(routes[v_idx])
                dist_after = self._route_distance(trial_route)
                increase = (dist_after - dist_before) * self.data.vehicles[vid]['cost_per_km']
                if load <= self.data.vehicles[vid]['capacity'] * 1.2:
                    if increase < best_increase:
                        best_increase = increase
                        best_v = v_idx
            if best_v is None:
                best_v = random.randint(0, self.num_vehicles - 1)
            routes[best_v].append(c)
            loads[best_v] += demand_c

        return routes

    def _generate_neighbor(self, solution):
        new_sol = [r.copy() for r in solution]
        move_type = random.choice(['swap', 'relocate'])
        n_routes = len(new_sol)
        if move_type == 'swap':
            r1, r2 = random.sample(range(n_routes), 2)
            if not new_sol[r1] or not new_sol[r2]:
                return new_sol
            i = random.randrange(len(new_sol[r1]))
            j = random.randrange(len(new_sol[r2]))
            new_sol[r1][i], new_sol[r2][j] = new_sol[r2][j], new_sol[r1][i]
        else:
            r1 = random.randrange(n_routes)
            if not new_sol[r1]:
                return new_sol
            r2 = random.randrange(n_routes)
            if r1 == r2 and len(new_sol[r1]) == 1:
                return new_sol
            i = random.randrange(len(new_sol[r1]))
            customer = new_sol[r1].pop(i)
            insert_pos = random.randrange(len(new_sol[r2]) + 1)
            new_sol[r2].insert(insert_pos, customer)
        return new_sol

    def optimize(self):
        current = self._initial_solution()
        current_cost = self._evaluate(current)
        best = [r.copy() for r in current]
        best_cost = current_cost
        history = []

        T = self.init_temp
        it = 0

        while T > self.final_temp and it < self.max_iters:
            for _ in range(self.neighborhood_size):
                neighbor = self._generate_neighbor(current)
                cost_neighbor = self._evaluate(neighbor)
                delta = cost_neighbor - current_cost
                if delta < 0:
                    current = neighbor
                    current_cost = cost_neighbor
                else:
                    p = np.exp(-delta / T)
                    if random.random() < p:
                        current = neighbor
                        current_cost = cost_neighbor

                if current_cost < best_cost:
                    best = [r.copy() for r in current]
                    best_cost = current_cost

                history.append(best_cost)

            if it % 50 == 0:
                print(f"迭代 {it}, 当前最佳成本: {best_cost:.2f}, 温度: {T:.2f}")

            T *= self.alpha
            it += 1

        return best, best_cost, history

    def print_solution(self, best_solution):
        total_cost = 0.0
        print("最优运输方案（模拟退火）：")
        print("=" * 50)
        for i, route in enumerate(best_solution):
            vid = f'V{i + 1}'
            v = self.data.vehicles[vid]
            if route:
                route_points = ['S0'] + [self.demand_points[idx] for idx in route] + ['S0']
                route_str = ' -> '.join(route_points)
                load = self._route_demand(route)
                dist = self._route_distance(route)
                t = self._route_time(route, v['speed'])
                cost = dist * v['cost_per_km']
                total_cost += cost
                print(f"车辆 {vid}:")
                print(f"  路径: {route_str}")
                print(f"  总需求量: {load}箱 (容量: {v['capacity']}箱)")
                print(f"  行驶距离: {dist:.2f}km")
                print(f"  行驶时间: {t:.2f}h (限制: {v['max_time']}h)")
                print(f"  运输成本: {cost:.2f}元")
                if 'D1' in route_points:
                    if t <= 4:
                        print("  ✓ D1在4小时内送达")
                    else:
                        print("  ✗ D1送达时间超过4小时限制")
                print()
            else:
                print(f"车辆 {vid}: 未使用")
                print()
        print(f"总运输成本(不含惩罚): {total_cost:.2f}元")
        return total_cost

    def plot_convergence(self, history, filename="SA_convergence_curve.png"):
        plt.figure(figsize=(10, 6))
        plt.plot(history)
        plt.title('模拟退火算法收敛曲线')
        plt.xlabel('迭代步数')
        plt.ylabel('历史最佳成本 (含惩罚)')
        plt.grid(True)
        plt.savefig(filename)
        plt.show()

    def plot_routes(self, best_solution, filename="SA_VRP_Results.png"):
        plt.figure(figsize=(10, 6))
        x_s0, y_s0 = self.data.coordinates['S0']
        plt.scatter(x_s0, y_s0, c='red', marker='s', s=100, label='供应点 S0')
        for dp in self.demand_points:
            x, y = self.data.coordinates[dp]
            plt.scatter(x, y, c='blue')
            plt.text(x + 0.3, y + 0.3, dp)
        colors = ['orange', 'green', 'purple', 'brown', 'cyan']
        for i, route in enumerate(best_solution):
            if not route:
                continue
            color = colors[i % len(colors)]
            nodes = ['S0'] + [self.demand_points[idx] for idx in route] + ['S0']
            xs = [self.data.coordinates[n][0] for n in nodes]
            ys = [self.data.coordinates[n][1] for n in nodes]
            plt.plot(xs, ys, '-o', color=color, label=f'车辆 V{i + 1}')
        plt.title('模拟退火算法 - 车辆路径方案')
        plt.xlabel('X 坐标 (km)')
        plt.ylabel('Y 坐标 (km)')
        plt.legend()
        plt.grid(True)
        plt.savefig(filename)
        plt.show()


def main():
    random.seed(123)
    np.random.seed(123)
    problem_data = ProblemData()
    sa_solver = SimulatedAnnealingVRP(
        problem_data,
        max_iters=2000,
        init_temp=800.0,
        final_temp=1e-3,
        alpha=0.995,
        neighborhood_size=60
    )
    print("开始模拟退火优化...")
    best_solution, best_cost, history = sa_solver.optimize()
    print("\n优化完成！")
    print("=" * 50)
    sa_solver.print_solution(best_solution)
    sa_solver.plot_convergence(history)
    sa_solver.plot_routes(best_solution)
    return best_solution, best_cost


if __name__ == "__main__":
    best, cost = main()
