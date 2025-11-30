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


class ParticleSwarmVRP:
    def __init__(self, problem_data,
                 swarm_size=60,
                 max_iters=500,
                 w=0.7,
                 c1=1.5,
                 c2=1.5,
                 w_min=0.4,
                 w_max=0.9,
                 velocity_clamp=0.3,
                 stagnation_iters=40,
                 penalty_capacity=10000,
                 penalty_time=10000,
                 penalty_priority=10000):
        self.data = problem_data
        self.demand_points = ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10']
        self.num_demands = len(self.demand_points)
        self.num_vehicles = 5

        self.swarm_size = swarm_size
        self.max_iters = max_iters
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.w_min = w_min
        self.w_max = w_max
        self.velocity_clamp = velocity_clamp
        self.stagnation_iters = stagnation_iters

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

    def _evaluate_routes(self, routes):
        total_cost = 0.0
        penalty = 0.0
        assigned = set()

        for i, route in enumerate(routes):
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

    # 粒子编码：
    # position: 长度为 num_demands 的实数向量
    # 解码规则：根据 position 从小到大排序得到客户访问顺序，再按简单启发式切分到 5 辆车

    def _decode_position(self, position):
        idx_order = np.argsort(position)
        customers = list(idx_order)
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

    def optimize(self):
        dim = self.num_demands
        # 初始化粒子群
        positions = np.random.rand(self.swarm_size, dim)
        velocities = np.zeros((self.swarm_size, dim))

        pbest_pos = positions.copy()
        pbest_val = np.full(self.swarm_size, np.inf)
        stagnation_counter = np.zeros(self.swarm_size, dtype=int)

        gbest_pos = None
        gbest_val = np.inf

        history = []

        for it in range(self.max_iters):
            inertia = self.w if self.w_min == self.w_max else (
                self.w_max - (self.w_max - self.w_min) * (it / max(1, self.max_iters - 1))
            )
            for i in range(self.swarm_size):
                routes = self._decode_position(positions[i])
                val = self._evaluate_routes(routes)
                if val < pbest_val[i]:
                    pbest_val[i] = val
                    pbest_pos[i] = positions[i].copy()
                    stagnation_counter[i] = 0
                if val < gbest_val:
                    gbest_val = val
                    gbest_pos = positions[i].copy()
                if val >= pbest_val[i]:
                    stagnation_counter[i] += 1

            history.append(gbest_val)

            if it % 50 == 0:
                print(f"迭代 {it}, 当前全局最佳成本: {gbest_val:.2f}")

            # 更新速度和位置
            r1 = np.random.rand(self.swarm_size, dim)
            r2 = np.random.rand(self.swarm_size, dim)
            velocities = (inertia * velocities +
                          self.c1 * r1 * (pbest_pos - positions) +
                          self.c2 * r2 * (gbest_pos - positions))
            if self.velocity_clamp is not None:
                velocities = np.clip(velocities, -self.velocity_clamp, self.velocity_clamp)
            positions = positions + velocities
            # 可选：限制位置范围在 [0,1]
            positions = np.clip(positions, 0.0, 1.0)

            if self.stagnation_iters and gbest_pos is not None:
                stuck_mask = stagnation_counter >= self.stagnation_iters
                if np.any(stuck_mask):
                    positions[stuck_mask] = np.random.rand(np.sum(stuck_mask), dim)
                    velocities[stuck_mask] = 0.0
                    stagnation_counter[stuck_mask] = 0

        best_routes = self._decode_position(gbest_pos)
        return best_routes, gbest_val, history

    def print_solution(self, best_solution):
        total_cost = 0.0
        print("最优运输方案（粒子群算法）：")
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

    def plot_convergence(self, history, filename="PSO_convergence_curve.png"):
        plt.figure(figsize=(10, 6))
        plt.plot(history)
        plt.title('粒子群算法收敛曲线')
        plt.xlabel('迭代次数')
        plt.ylabel('全局最佳成本 (含惩罚)')
        plt.grid(True)
        plt.savefig(filename)
        plt.show()

    def plot_routes(self, best_solution, filename="PSO_VRP_Results.png"):
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
        plt.title('粒子群算法 - 车辆路径方案')
        plt.xlabel('X 坐标 (km)')
        plt.ylabel('Y 坐标 (km)')
        plt.legend()
        plt.grid(True)
        plt.savefig(filename)
        plt.show()


def main():
    random.seed(101)
    np.random.seed(101)
    problem_data = ProblemData()
    pso_solver = ParticleSwarmVRP(
        problem_data,
        swarm_size=60,
        max_iters=400,
        w=0.7,
        c1=1.5,
        c2=1.5,
        w_min=0.4,
        w_max=0.9,
        velocity_clamp=0.3,
        stagnation_iters=50
    )
    print("开始粒子群算法优化...")
    best_solution, best_cost, history = pso_solver.optimize()
    print("\n优化完成！")
    print("=" * 50)
    pso_solver.print_solution(best_solution)
    pso_solver.plot_convergence(history)
    pso_solver.plot_routes(best_solution)
    return best_solution, best_cost


if __name__ == "__main__":
    best, cost = main()
