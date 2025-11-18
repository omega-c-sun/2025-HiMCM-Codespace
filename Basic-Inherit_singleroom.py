import heapq
import math
from collections import defaultdict
from itertools import combinations
import time
import random
import numpy as np

def generate_grid(rows=12, cols=16, fire_ratio=0.5, people_ratio=0.1):

    total_cells = rows * cols
    
    fire_count = int(total_cells * fire_ratio)
    people_count = int(total_cells * people_ratio)
    
    fire_count = min(fire_count, total_cells - 1) 
    
    all_positions = []
    for i in range(rows):
        for j in range(cols):
            if i == 0 and j == 0:
                continue 
            all_positions.append((i, j))
    
    fire_positions = random.sample(all_positions, fire_count)
    
    remaining_positions = [pos for pos in all_positions if pos not in fire_positions]
    people_positions = random.sample(remaining_positions, min(people_count, len(remaining_positions)))
    
    grid = [[0 for _ in range(cols)] for _ in range(rows)]
    
    grid[0][0] = 0
    
    for i, j in fire_positions:
        grid[i][j] = -1
    
    for i, j in people_positions:
        grid[i][j] = 1
    
    return grid

def print_grid(grid):
    symbols = {-1: '🔥', 0: '·', 1: '👤'}
    
    print("网格布局:")
    print("🔥 = 火, · = 空地, 👤 = 人")
    print("-" * (len(grid[0]) * 3 + 1))
    
    for row in grid:
        print("|", end="")
        for cell in row:
            print(f" {symbols[cell]}", end="")
        print(" |")
    
    print("-" * (len(grid[0]) * 3 + 1))
    
    fire_count = sum(row.count(-1) for row in grid)
    people_count = sum(row.count(1) for row in grid)
    empty_count = sum(row.count(0) for row in grid)
    
    print(f"统计: 火={fire_count}, 人={people_count}, 空地={empty_count}")
    print(f"比例: 火={fire_count/(len(grid)*len(grid[0])):.1%}, 人={people_count/(len(grid)*len(grid[0])):.1%}")

def generate_multiple_grids(num_grids=5, rows=12, cols=16):
    """生成多个网格供选择"""
    grids = []
    for i in range(num_grids):
        print(f"\n=== 网格 {i+1} ===")
        grid = generate_grid(rows, cols)
        print_grid(grid)
        grids.append(grid)
    
    return grids

class OptimizedPathFinder:
    def __init__(self, grid):
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])
        self.movement_speed = 1/6
        self.fire_extinguish_time = 5
        
        self.people_positions = []
        for i in range(self.rows):
            for j in range(self.cols):
                if self.grid[i][j] == 1:
                    self.people_positions.append((i, j))
        
        self.start = (0, 0)
        self.all_points = [self.start] + self.people_positions
        
    def solve_with_dp(self):
        if not self.people_positions:
            return [self.start], 0
        
        n = len(self.all_points)
        
        print("预计算点对距离...")
        dist_matrix = self._precompute_distances()
        
        dp = [[float('inf')] * n for _ in range(1 << n)]
        parent = [[-1] * n for _ in range(1 << n)]
        
        dp[1][0] = 0
        
        print("运行动态规划...")
        for mask in range(1 << n):
            for last in range(n):
                if dp[mask][last] == float('inf'):
                    continue
                
                for next_node in range(n):
                    if mask & (1 << next_node):
                        continue
                    
                    new_mask = mask | (1 << next_node)
                    new_time = dp[mask][last] + dist_matrix[last][next_node]
                    
                    if new_time < dp[new_mask][next_node]:
                        dp[new_mask][next_node] = new_time
                        parent[new_mask][next_node] = last
        
        full_mask = (1 << n) - 1
        min_time = float('inf')
        best_last = -1
        
        for last in range(n):
            total_time = dp[full_mask][last] + dist_matrix[last][0]
            if total_time < min_time:
                min_time = total_time
                best_last = last
        
        path = self._reconstruct_path(parent, full_mask, best_last)
        return path, min_time
    
    def _precompute_distances(self):
        n = len(self.all_points)
        dist_matrix = [[0] * n for _ in range(n)]
        
        for i in range(n):
            for j in range(i + 1, n):
                dist = self._bidirectional_search(self.all_points[i], self.all_points[j])
                dist_matrix[i][j] = dist
                dist_matrix[j][i] = dist
        
        return dist_matrix
    
    def _bidirectional_search(self, start, end):
        if start == end:
            return 0
        
        forward_pq = [(0, start, False)]
        forward_visited = {}
        
        backward_pq = [(0, end, False)]
        backward_visited = {}
        
        best_time = float('inf')
        
        while forward_pq and backward_pq:
            if forward_pq:
                f_time, f_pos, f_has_people = heapq.heappop(forward_pq)
                
                if (f_pos, f_has_people) in forward_visited:
                    continue
                forward_visited[(f_pos, f_has_people)] = f_time
                
                if (f_pos, f_has_people) in backward_visited:
                    total_time = f_time + backward_visited[(f_pos, f_has_people)]
                    best_time = min(best_time, total_time)
                
                for neighbor in self._get_neighbors(f_pos, f_has_people):
                    nx, ny = neighbor
                    new_time = f_time + self._get_move_cost(f_pos, neighbor, f_has_people)
                    new_has_people = f_has_people or (self.grid[nx][ny] == 1)
                    
                    heapq.heappush(forward_pq, (new_time, neighbor, new_has_people))
            
            if backward_pq:
                b_time, b_pos, b_has_people = heapq.heappop(backward_pq)
                
                if (b_pos, b_has_people) in backward_visited:
                    continue
                backward_visited[(b_pos, b_has_people)] = b_time
                
                if (b_pos, b_has_people) in forward_visited:
                    total_time = b_time + forward_visited[(b_pos, b_has_people)]
                    best_time = min(best_time, total_time)
                
                for neighbor in self._get_neighbors(b_pos, b_has_people):
                    nx, ny = neighbor
                    new_time = b_time + self._get_move_cost(b_pos, neighbor, b_has_people)
                    new_has_people = b_has_people or (self.grid[nx][ny] == 1)
                    
                    heapq.heappush(backward_pq, (new_time, neighbor, new_has_people))
        
        return best_time
    
    def _get_neighbors(self, pos, has_people):
        x, y = pos
        neighbors = []
        
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.rows and 0 <= ny < self.cols:
                neighbors.append((nx, ny))
        
        return neighbors
    
    def _get_move_cost(self, from_pos, to_pos, has_people):
        x, y = to_pos
        move_time = self.movement_speed
        
        if self.grid[x][y] == -1 and has_people:
            move_time += self.fire_extinguish_time
        
        return move_time
    
    def _reconstruct_path(self, parent, mask, last):
        path = []
        current_mask = mask
        current_last = last
        
        while current_last != -1:
            path.append(self.all_points[current_last])
            prev_last = parent[current_mask][current_last]
            if prev_last == -1:
                break
            current_mask = current_mask ^ (1 << current_last)
            current_last = prev_last
        
        path.reverse()
        return path

class GeneticPathFinder:
    def __init__(self, grid, population_size=50, generations=1000):
        self.finder = OptimizedPathFinder(grid)
        self.population_size = population_size
        self.generations = generations
        
    def solve(self):
        if not self.finder.people_positions:
            return [self.finder.start], 0
        
        n = len(self.finder.all_points)
        dist_matrix = self.finder._precompute_distances()
        
        population = self._initialize_population(n)
        
        best_path = None
        best_time = float('inf')
        
        for generation in range(self.generations):
            fitness = [self._evaluate_fitness(ind, dist_matrix) for ind in population]
            
            min_fitness = min(fitness)
            if min_fitness < best_time:
                best_time = min_fitness
                best_idx = fitness.index(min_fitness)
                best_path = [self.finder.all_points[i] for i in population[best_idx]]
            
            new_population = []
            for _ in range(self.population_size // 2):
                parent1 = self._select_parent(population, fitness)
                parent2 = self._select_parent(population, fitness)
                child1, child2 = self._crossover(parent1, parent2)
                child1 = self._mutate(child1)
                child2 = self._mutate(child2)
                new_population.extend([child1, child2])
            
            population = new_population
        
        return best_path, best_time
    
    def _initialize_population(self, n):
        population = []
        for _ in range(self.population_size):
            individual = list(range(1, n))
            np.random.shuffle(individual)
            individual = [0] + individual
            population.append(individual)
        return population
    
    def _evaluate_fitness(self, individual, dist_matrix):

        total_time = 0
        for i in range(len(individual) - 1):
            total_time += dist_matrix[individual[i]][individual[i + 1]]
 
        total_time += dist_matrix[individual[-1]][individual[0]]
        return total_time
    
    def _select_parent(self, population, fitness):
    
        max_fitness = max(fitness)
        adjusted_fitness = [max_fitness - f + 1 for f in fitness]
        total = sum(adjusted_fitness)
        pick = random.uniform(0, total)
        current = 0
        
        for i, individual in enumerate(population):
            current += adjusted_fitness[i]
            if current > pick:
                return individual
        
        return population[0]
    
    def _crossover(self, parent1, parent2):
    
        size = len(parent1)
        start, end = sorted(random.sample(range(1, size), 2))
        
        child1 = [0] + [-1] * (size - 1)
        child2 = [0] + [-1] * (size - 1)
        
        child1[start:end] = parent1[start:end]
        child2[start:end] = parent2[start:end]
        
        self._fill_remaining(child1, parent2, start, end)
        self._fill_remaining(child2, parent1, start, end)
        
        return child1, child2
    
    def _fill_remaining(self, child, parent, start, end):
        size = len(child)
        current_pos = end % size
        
        for gene in parent:
            if gene not in child:
                while child[current_pos] != -1:
                    current_pos = (current_pos + 1) % size
                child[current_pos] = gene
    
    def _mutate(self, individual):
        if random.random() < 0.1:
            i, j = random.sample(range(1, len(individual)), 2)
            individual[i], individual[j] = individual[j], individual[i]
        return individual

def main():
    # grid=[[0,1,0,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    #  [1,0,1,1,-1,-1,-1,-1,-1,-1,-1,-1],
    #     [0,1,0,-1,1,1,-1,-1,-1,-1,-1,-1],
    #     [-1,1,-1,0,-1,1,-1,1,-1,-1,-1,-1],
    #     [-1,-1,1,-1,0,-1,-1,1,1,-1,-1,-1],
    #     [-1,-1,1,1,-1,0,-1,-1,-1,1,-1,-1],
    #     [-1,-1,-1,-1,-1,-1,0,1,-1,-1,1,1],
    #     [-1,-1,-1,1,1,-1,1,0,-1,-1,-1,1],
    #     [-1,-1,-1,-1,-1,-1,-1,-1,0,1,1,-1],
    #     [-1,-1,-1,-1,-1,1,-1,-1,1,0,-1,1],
    #     [-1,-1,-1,-1,-1,-1,1,-1,1,-1,0,1],
    #     [-1,-1,-1,-1,-1,-1,1,1,-1,1,1,0]]
    
    # print("=== 动态规划解法 ===")
    # start_time = time.time()
    # dp_finder = OptimizedPathFinder(grid)
    # dp_path, dp_time = dp_finder.solve_with_dp()
    # dp_duration = time.time() - start_time
    
    # print("路径:", dp_path)
    # print("时间:", round(dp_time, 2), "秒")
    # print("计算时间:", round(dp_duration, 2), "秒")
    print("=== 生成随机网格 ===")
    grid = generate_grid()
    print_grid(grid)
    print("\n=== 遗传算法解法 ===")
    start_time = time.time()
    ga_finder = GeneticPathFinder(grid)
    ga_path, ga_time = ga_finder.solve()
    ga_duration = time.time() - start_time
        
    print("路径:", ga_path)
    print("时间:", round(ga_time, 2), "秒")
    print("计算时间:", round(ga_duration, 2), "秒")

if __name__ == "__main__":
    import numpy as np
    import random
    main()

