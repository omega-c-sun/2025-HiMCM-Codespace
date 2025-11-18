import heapq
import math
import random
import numpy as np
from collections import defaultdict
import time

class PathFinder:
    def __init__(self, grid):
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])
        self.movement_speed = 1/6  # 6格/秒 = 1/6秒/格
        self.fire_extinguish_time = 5  # 5秒灭一格火
        
    def find_path(self, start, end, has_people=False):
        """使用A*算法找到两点间的最短路径"""
        def heuristic(a, b):
            return abs(a[0]-b[0]) + abs(a[1]-b[1])
        
        open_set = []
        heapq.heappush(open_set, (0, start, has_people))
        came_from = {}
        g_score = defaultdict(lambda: float('inf'))
        g_score[(start, has_people)] = 0
        f_score = defaultdict(lambda: float('inf'))
        f_score[(start, has_people)] = heuristic(start, end)
        
        while open_set:
            _, current, current_has_people = heapq.heappop(open_set)
            
            if current == end:
                # 重建路径
                path = [current]
                total_time = g_score[(current, current_has_people)]
                while current in came_from:
                    current, current_has_people = came_from[(current, current_has_people)]
                    path.append(current)
                path.reverse()
                return path, total_time
            
            for neighbor in self._get_neighbors(current):
                nx, ny = neighbor
                
                # 计算移动成本
                move_cost = self.movement_speed
                if self.grid[nx][ny] == -1 and current_has_people:
                    move_cost += self.fire_extinguish_time
                
                # 检查是否可以移动
                if self.grid[nx][ny] == -2:  # 墙
                    continue
                
                # 更新是否携带人员状态
                new_has_people = current_has_people
                if self.grid[nx][ny] == 1:
                    new_has_people = True
                
                # 计算新的g分数
                tentative_g_score = g_score[(current, current_has_people)] + move_cost
                
                if tentative_g_score < g_score[(neighbor, new_has_people)]:
                    came_from[(neighbor, new_has_people)] = (current, current_has_people)
                    g_score[(neighbor, new_has_people)] = tentative_g_score
                    f_score[(neighbor, new_has_people)] = tentative_g_score + heuristic(neighbor, end)
                    heapq.heappush(open_set, (f_score[(neighbor, new_has_people)], neighbor, new_has_people))
        
        return [], float('inf')
    
    def _get_neighbors(self, pos):
        """获取有效邻居"""
        x, y = pos
        neighbors = []
        
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.rows and 0 <= ny < self.cols:
                neighbors.append((nx, ny))
        
        return neighbors

class BuildingGenerator:
    """生成建筑布局"""
    
    def __init__(self, rows=15, cols=36):
        self.rows = rows
        self.cols = cols
    
    def generate_building(self):
        """生成建筑布局"""
        # 初始化建筑网格，全部为墙(-2)
        building = [[-2 for _ in range(self.cols)] for _ in range(self.rows)]
        
        # 定义走廊位置 (行索引6-8)
        hallway_start_row = 6
        hallway_end_row = 8
        
        # 创建走廊 (空地)
        for i in range(hallway_start_row, hallway_end_row + 1):
            for j in range(self.cols):
                building[i][j] = 0
        
        # 定义房间位置和门的位置
        room_doors = {
            1: (5, 6),   # 房间1的门位置 (行, 列)
            2: (5, 18),  # 房间2的门位置
            3: (5, 30),  # 房间3的门位置
            4: (9, 6),   # 房间4的门位置
            5: (9, 18),  # 房间5的门位置
            6: (9, 30)   # 房间6的门位置
        }
        
        # 定义每个房间的起始位置和大小
        rooms = {
            1: {"start_row": 0, "end_row": 5, "start_col": 0, "end_col": 11},
            2: {"start_row": 0, "end_row": 5, "start_col": 12, "end_col": 23},
            3: {"start_row": 0, "end_row": 5, "start_col": 24, "end_col": 35},
            4: {"start_row": 9, "end_row": 14, "start_col": 0, "end_col": 11},
            5: {"start_row": 9, "end_row": 14, "start_col": 12, "end_col": 23},
            6: {"start_row": 9, "end_row": 14, "start_col": 24, "end_col": 35}
        }
        
        # 为每个房间生成内部布局
        for room_id, room_info in rooms.items():
            room_rows = room_info["end_row"] - room_info["start_row"] + 1
            room_cols = room_info["end_col"] - room_info["start_col"] + 1
            
            # 生成房间内部布局
            room_grid = self._generate_room_layout(room_rows, room_cols)
            
            # 将房间布局复制到建筑中
            for i in range(room_rows):
                for j in range(room_cols):
                    building[room_info["start_row"] + i][room_info["start_col"] + j] = room_grid[i][j]
            
            # 设置门的位置 (连接房间和走廊)
            door_row, door_col = room_doors[room_id]
            building[door_row][door_col] = 0  # 门位置设为空地
        
        # 设置救援者起始位置
        building[7][0] = 2   # 左侧救援者 (标记为2)
        building[7][35] = 3  # 右侧救援者 (标记为3)
        
        return building, rooms, room_doors
    
    def _generate_room_layout(self, rows=6, cols=12, fire_ratio=0.5, people_ratio=0.1):
        """生成单个房间的内部布局"""
        total_cells = rows * cols
        
        # 计算火和人的数量
        fire_count = int(total_cells * fire_ratio)
        people_count = int(total_cells * people_ratio)
        
        # 创建所有可能位置的列表
        all_positions = [(i, j) for i in range(rows) for j in range(cols)]
        
        # 随机选择火的位置
        fire_positions = random.sample(all_positions, fire_count)
        
        # 从剩余位置中随机选择人的位置
        remaining_positions = [pos for pos in all_positions if pos not in fire_positions]
        people_positions = random.sample(remaining_positions, min(people_count, len(remaining_positions)))
        
        # 创建房间网格
        room = [[0 for _ in range(cols)] for _ in range(rows)]
        
        # 设置火的位置
        for i, j in fire_positions:
            room[i][j] = -1
        
        # 设置人的位置
        for i, j in people_positions:
            room[i][j] = 1
        
        return room

class RescuePlanner:
    """救援规划器"""
    
    def __init__(self, building, rooms, room_doors):
        self.building = building
        self.rooms = rooms
        self.room_doors = room_doors
        self.finder = PathFinder(building)
        
        # 救援者起始位置
        self.left_rescuer_start = (7, 0)
        self.right_rescuer_start = (7, 35)
        
        # 出口位置
        self.exits = [(7, 0), (7, 35)]
    
    def get_room_info(self):
        """获取每个房间的详细信息"""
        room_info = {}
        
        for room_id, room_data in self.rooms.items():
            start_row = room_data["start_row"]
            end_row = room_data["end_row"]
            start_col = room_data["start_col"]
            end_col = room_data["end_col"]
            
            # 提取房间区域
            room_area = []
            for i in range(start_row, end_row + 1):
                row = []
                for j in range(start_col, end_col + 1):
                    row.append(self.building[i][j])
                room_area.append(row)
            
            # 统计房间内的人员和火
            people_count = sum(row.count(1) for row in room_area)
            fire_count = sum(row.count(-1) for row in room_area)
            
            # 计算房间中心位置
            center_row = (start_row + end_row) // 2
            center_col = (start_col + end_col) // 2
            
            room_info[room_id] = {
                "people_count": people_count,
                "fire_count": fire_count,
                "area": (end_row - start_row + 1) * (end_col - start_col + 1),
                "door_position": self.room_doors[room_id],
                "center_position": (center_row, center_col),
                "importance": people_count * 10 + fire_count * 2  # 重要性计算公式
            }
        
        return room_info
    
    def calculate_rescue_time(self, start_pos, room_id, room_info):
        """计算从起始位置到房间救援的时间"""
        door_pos = room_info[room_id]["door_position"]
        
        # 计算到门的时间
        path_to_door, time_to_door = self.finder.find_path(start_pos, door_pos, has_people=False)
        
        # 计算房间内救援时间
        room_time = self._calculate_room_rescue_time(room_id, room_info)
        
        # 计算从门到最近出口的时间
        exit_times = []
        for exit_pos in self.exits:
            _, time_to_exit = self.finder.find_path(door_pos, exit_pos, has_people=True)
            exit_times.append(time_to_exit)
        
        min_exit_time = min(exit_times)
        
        total_time = time_to_door + room_time + min_exit_time
        return total_time
    
    def _calculate_room_rescue_time(self, room_id, room_info):
        """计算房间内救援时间"""
        room_data = self.rooms[room_id]
        door_pos = room_info[room_id]["door_position"]
        
        # 提取房间内的人员位置
        people_positions = []
        for i in range(room_data["start_row"], room_data["end_row"] + 1):
            for j in range(room_data["start_col"], room_data["end_col"] + 1):
                if self.building[i][j] == 1:
                    people_positions.append((i, j))
        
        # 如果没有人员，只需要进出房间的时间
        if not people_positions:
            return 2 * self.finder.movement_speed  # 进出房间的时间
        
        # 计算救援所有人员的时间
        # 这里使用简化模型：时间与人员数量和火的数量成正比
        base_time = len(people_positions) * 2 * self.finder.movement_speed
        fire_penalty = room_info[room_id]["fire_count"] * self.finder.fire_extinguish_time
        
        return base_time + fire_penalty
    
    def optimize_rescue_order(self):
        """优化救援顺序"""
        room_info = self.get_room_info()
        
        # 计算每个房间对两个救援者的救援时间
        left_rescue_times = {}
        right_rescue_times = {}
        
        for room_id in room_info.keys():
            left_rescue_times[room_id] = self.calculate_rescue_time(self.left_rescuer_start, room_id, room_info)
            right_rescue_times[room_id] = self.calculate_rescue_time(self.right_rescuer_start, room_id, room_info)
        
        # 分配房间给救援者
        left_rooms, right_rooms = self._assign_rooms(room_info, left_rescue_times, right_rescue_times)
        
        # 优化每个救援者的房间访问顺序
        left_order = self._optimize_sequence(left_rooms, room_info, left_rescue_times, self.left_rescuer_start)
        right_order = self._optimize_sequence(right_rooms, room_info, right_rescue_times, self.right_rescuer_start)
        
        return left_order, right_order, room_info
    
    def _assign_rooms(self, room_info, left_times, right_times):
        """分配房间给救援者"""
        # 计算每个房间对两个救援者的相对成本
        room_costs = {}
        for room_id in room_info.keys():
            # 考虑救援时间和重要性
            cost_diff = abs(left_times[room_id] - right_times[room_id])
            importance = room_info[room_id]["importance"]
            room_costs[room_id] = (cost_diff, importance, left_times[room_id], right_times[room_id])
        
        # 按成本和重要性排序
        sorted_rooms = sorted(room_costs.keys(), 
                             key=lambda x: (room_costs[x][0], -room_costs[x][1]), 
                             reverse=True)
        
        left_rooms = []
        right_rooms = []
        left_total_time = 0
        right_total_time = 0
        
        for room_id in sorted_rooms:
            left_time = left_times[room_id]
            right_time = right_times[room_id]
            
            # 平衡分配，考虑总时间
            if left_total_time + left_time <= right_total_time + right_time:
                left_rooms.append(room_id)
                left_total_time += left_time
            else:
                right_rooms.append(room_id)
                right_total_time += right_time
        
        return left_rooms, right_rooms
    
    def _optimize_sequence(self, rooms, room_info, rescue_times, start_pos):
        """优化房间访问顺序"""
        if not rooms:
            return []
        
        # 使用简单的贪心算法：优先访问重要性高且救援时间短的房间
        room_scores = {}
        for room_id in rooms:
            importance = room_info[room_id]["importance"]
            time_cost = rescue_times[room_id]
            # 得分 = 重要性 / 时间成本
            room_scores[room_id] = importance / time_cost if time_cost > 0 else float('inf')
        
        # 按得分排序
        sorted_rooms = sorted(rooms, key=lambda x: room_scores[x], reverse=True)
        
        return sorted_rooms

class GeneticRescuePlanner:
    """使用遗传算法优化救援计划"""
    
    def __init__(self, building, rooms, room_doors, population_size=50, generations=100):
        self.planner = RescuePlanner(building, rooms, room_doors)
        self.population_size = population_size
        self.generations = generations
        self.room_info = self.planner.get_room_info()
        
    def solve(self):
        """使用遗传算法求解"""
        room_ids = list(self.room_info.keys())
        
        if not room_ids:
            return [], [], self.room_info
        
        # 初始化种群
        population = self._initialize_population(room_ids)
        
        best_left = None
        best_right = None
        best_time = float('inf')
        
        for generation in range(self.generations):
            # 评估适应度
            fitness = [self._evaluate_fitness(ind) for ind in population]
            
            # 找到最优解
            min_fitness = min(fitness)
            if min_fitness < best_time:
                best_time = min_fitness
                best_idx = fitness.index(min_fitness)
                best_left, best_right = population[best_idx]
            
            # 选择、交叉、变异
            new_population = []
            for _ in range(self.population_size // 2):
                parent1 = self._select_parent(population, fitness)
                parent2 = self._select_parent(population, fitness)
                child1, child2 = self._crossover(parent1, parent2, room_ids)
                child1 = self._mutate(child1, room_ids)
                child2 = self._mutate(child2, room_ids)
                new_population.extend([child1, child2])
            
            population = new_population
        
        return best_left, best_right, self.room_info
    
    def _initialize_population(self, room_ids):
        """初始化种群"""
        population = []
        for _ in range(self.population_size):
            # 随机分配房间给两个救援者
            left_rooms = random.sample(room_ids, random.randint(0, len(room_ids)))
            right_rooms = [r for r in room_ids if r not in left_rooms]
            
            # 随机排序
            random.shuffle(left_rooms)
            random.shuffle(right_rooms)
            
            population.append((left_rooms, right_rooms))
        return population
    
    def _evaluate_fitness(self, individual):
        """评估个体适应度（总时间）"""
        left_rooms, right_rooms = individual
        
        # 计算左侧救援者的总时间
        left_time = 0
        current_pos = self.planner.left_rescuer_start
        for room_id in left_rooms:
            left_time += self.planner.calculate_rescue_time(current_pos, room_id, self.room_info)
            # 更新位置为房间门
            current_pos = self.room_info[room_id]["door_position"]
        
        # 计算右侧救援者的总时间
        right_time = 0
        current_pos = self.planner.right_rescuer_start
        for room_id in right_rooms:
            right_time += self.planner.calculate_rescue_time(current_pos, room_id, self.room_info)
            # 更新位置为房间门
            current_pos = self.room_info[room_id]["door_position"]
        
        # 总时间是两个救援者中较长的那个
        return max(left_time, right_time)
    
    def _select_parent(self, population, fitness):
        """轮盘赌选择父代"""
        # 转换为适应度（数值越小越好）
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
    
    def _crossover(self, parent1, parent2, room_ids):
        """交叉操作"""
        left1, right1 = parent1
        left2, right2 = parent2
        
        # 合并所有房间
        all_rooms = list(set(left1 + right1))
        
        # 随机选择一部分房间给子代1
        sample_size1 = random.randint(0, len(all_rooms))
        child1_left = random.sample(all_rooms, sample_size1) if all_rooms else []
        child1_right = [r for r in all_rooms if r not in child1_left]
        
        # 随机选择一部分房间给子代2
        sample_size2 = random.randint(0, len(all_rooms))
        child2_left = random.sample(all_rooms, sample_size2) if all_rooms else []
        child2_right = [r for r in all_rooms if r not in child2_left]
        
        return (child1_left, child1_right), (child2_left, child2_right)
    
    def _mutate(self, individual, room_ids):
        """变异操作"""
        left, right = individual
        
        if random.random() < 0.2:  # 20%变异概率
            # 交换一个房间
            if left and right:
                room_from_left = random.choice(left)
                room_from_right = random.choice(right)
                left.remove(room_from_left)
                right.remove(room_from_right)
                left.append(room_from_right)
                right.append(room_from_left)
            elif left and not right:
                # 如果只有左侧有房间，移动一个到右侧
                room_to_move = random.choice(left)
                left.remove(room_to_move)
                right.append(room_to_move)
            elif not left and right:
                # 如果只有右侧有房间，移动一个到左侧
                room_to_move = random.choice(right)
                right.remove(room_to_move)
                left.append(room_to_move)
        
        # 随机打乱顺序
        random.shuffle(left)
        random.shuffle(right)
        
        return (left, right)

class FireExtinguisher:
    """灭火规划器"""
    
    def __init__(self, building, rooms, room_doors):
        self.building = building
        self.rooms = rooms
        self.room_doors = room_doors
        self.finder = PathFinder(building)
        
        # 救援者起始位置
        self.left_rescuer_start = (7, 0)
        self.right_rescuer_start = (7, 35)
        
        # 出口位置
        self.exits = [(7, 0), (7, 35)]
    
    def get_fire_locations(self):
        """获取所有火的位置"""
        fire_locations = []
        for i in range(len(self.building)):
            for j in range(len(self.building[0])):
                if self.building[i][j] == -1:
                    fire_locations.append((i, j))
        return fire_locations
    
    def optimize_fire_extinguish_plan(self):
        """优化灭火计划"""
        fire_locations = self.get_fire_locations()
        
        if not fire_locations:
            return [], [], 0
        
        # 将火分配给两个救援者
        left_fires, right_fires = self._assign_fires(fire_locations)
        
        # 优化每个救援者的灭火顺序
        left_plan, left_time = self._optimize_fire_sequence(left_fires, self.left_rescuer_start)
        right_plan, right_time = self._optimize_fire_sequence(right_fires, self.right_rescuer_start)
        
        return left_plan, right_plan, max(left_time, right_time)
    
    def _assign_fires(self, fire_locations):
        """将火分配给两个救援者"""
        if not fire_locations:
            return [], []
        
        # 按距离左侧和右侧的距离分组
        left_fires = []
        right_fires = []
        
        for fire in fire_locations:
            # 计算到左侧和右侧的距离
            left_dist = abs(fire[0] - self.left_rescuer_start[0]) + abs(fire[1] - self.left_rescuer_start[1])
            right_dist = abs(fire[0] - self.right_rescuer_start[0]) + abs(fire[1] - self.right_rescuer_start[1])
            
            if left_dist <= right_dist:
                left_fires.append(fire)
            else:
                right_fires.append(fire)
        
        return left_fires, right_fires
    
    def _optimize_fire_sequence(self, fires, start_pos):
        """优化灭火顺序（使用最近邻算法）"""
        if not fires:
            return [], 0
        
        # 使用最近邻算法找到近似最优路径
        unvisited = fires.copy()
        current_pos = start_pos
        path = [current_pos]
        total_time = 0
        
        while unvisited:
            # 找到最近的未访问火点
            nearest_fire = None
            min_distance = float('inf')
            
            for fire in unvisited:
                # 使用曼哈顿距离作为启发式
                distance = abs(fire[0] - current_pos[0]) + abs(fire[1] - current_pos[1])
                if distance < min_distance:
                    min_distance = distance
                    nearest_fire = fire
            
            # 计算到最近火点的实际路径和时间
            path_to_fire, time_to_fire = self.finder.find_path(current_pos, nearest_fire, has_people=False)
            
            # 添加移动时间和灭火时间
            total_time += time_to_fire + self.finder.fire_extinguish_time
            
            # 更新当前位置和路径
            current_pos = nearest_fire
            path.extend(path_to_fire[1:])  # 跳过起点（已经在路径中）
            path.append(("EXTINGUISH", current_pos))  # 标记灭火点
            
            # 从未访问列表中移除
            unvisited.remove(nearest_fire)
        
        # 最后回到起始位置
        path_to_start, time_to_start = self.finder.find_path(current_pos, start_pos, has_people=False)
        total_time += time_to_start
        path.extend(path_to_start[1:])
        
        return path, total_time

def print_building(building):
    """打印建筑布局"""
    symbols = {-2: '▓', -1: '🔥', 0: '·', 1: '👤', 2: '🚒', 3: '🚑'}
    
    print("建筑布局:")
    print("▓ = 墙, 🔥 = 火, · = 空地, 👤 = 人, 🚒 = 左侧救援者, 🚑 = 右侧救援者")
    print("-" * (len(building[0]) * 3 + 1))
    
    for i, row in enumerate(building):
        print(f"{i:2d}|", end="")
        for cell in row:
            print(f" {symbols[cell]}", end="")
        print(" |")
    
    print("-" * (len(building[0]) * 3 + 1))

def print_room_info(room_info):
    """打印房间信息"""
    print("\n房间信息:")
    print("房间号 | 人员数 | 火数量 | 重要性 | 门位置")
    print("-" * 50)
    for room_id in sorted(room_info.keys()):
        info = room_info[room_id]
        print(f"  {room_id}    |   {info['people_count']}   |   {info['fire_count']}   |   {info['importance']}   | {info['door_position']}")

def print_rescue_plan(left_order, right_order, room_info):
    """打印救援计划"""
    print("\n=== 最优救援计划 ===")
    print("\n左侧救援者路线:")
    if left_order:
        for i, room_id in enumerate(left_order):
            info = room_info[room_id]
            print(f"  {i+1}. 房间 {room_id} (人员: {info['people_count']}, 火: {info['fire_count']}, 重要性: {info['importance']})")
    else:
        print("  无房间需要救援")
    
    print("\n右侧救援者路线:")
    if right_order:
        for i, room_id in enumerate(right_order):
            info = room_info[room_id]
            print(f"  {i+1}. 房间 {room_id} (人员: {info['people_count']}, 火: {info['fire_count']}, 重要性: {info['importance']})")
    else:
        print("  无房间需要救援")

def print_fire_extinguish_plan(left_plan, right_plan, fire_time):
    """打印灭火计划"""
    print("\n=== 灭火计划 ===")
    
    print(f"\n预计灭火时间: {fire_time:.2f} 秒")
    
    # 统计左侧救援者灭火点
    left_fires = [step[1] for step in left_plan if isinstance(step, tuple) and step[0] == "EXTINGUISH"]
    print(f"\n左侧救援者灭火点 ({len(left_fires)} 个):")
    for i, fire_pos in enumerate(left_fires):
        print(f"  {i+1}. 位置 {fire_pos}")
    
    # 统计右侧救援者灭火点
    right_fires = [step[1] for step in right_plan if isinstance(step, tuple) and step[0] == "EXTINGUISH"]
    print(f"\n右侧救援者灭火点 ({len(right_fires)} 个):")
    for i, fire_pos in enumerate(right_fires):
        print(f"  {i+1}. 位置 {fire_pos}")

def main():
    # 生成建筑布局
    print("生成建筑布局...")
    generator = BuildingGenerator()
    building, rooms, room_doors = generator.generate_building()
    
    # 打印建筑布局
    print_building(building)
    
    # 使用遗传算法优化救援计划
    print("\n使用遗传算法优化救援计划...")
    start_time = time.time()
    genetic_planner = GeneticRescuePlanner(building, rooms, room_doors)
    left_order, right_order, room_info = genetic_planner.solve()
    genetic_time = time.time() - start_time
    
    # 打印房间信息
    print_room_info(room_info)
    
    # 打印救援计划
    print_rescue_plan(left_order, right_order, room_info)
    
    print(f"\n计算时间: {genetic_time:.2f} 秒")
    
    # 计算总救援时间
    planner = RescuePlanner(building, rooms, room_doors)
    left_time = 0
    current_pos = planner.left_rescuer_start
    for room_id in left_order:
        left_time += planner.calculate_rescue_time(current_pos, room_id, room_info)
        current_pos = room_info[room_id]["door_position"]
    
    right_time = 0
    current_pos = planner.right_rescuer_start
    for room_id in right_order:
        right_time += planner.calculate_rescue_time(current_pos, room_id, room_info)
        current_pos = room_info[room_id]["door_position"]
    
    rescue_time = max(left_time, right_time)
    print(f"\n预计救援时间: {rescue_time:.2f} 秒")
    print(f"左侧救援者时间: {left_time:.2f} 秒")
    print(f"右侧救援者时间: {right_time:.2f} 秒")
    
    # 灭火计划
    print("\n\n规划灭火任务...")
    fire_planner = FireExtinguisher(building, rooms, room_doors)
    left_fire_plan, right_fire_plan, fire_time = fire_planner.optimize_fire_extinguish_plan()
    
    print_fire_extinguish_plan(left_fire_plan, right_fire_plan, fire_time)
    
    # 总时间（救援 + 灭火）
    total_time = rescue_time + fire_time
    print(f"\n=== 总任务时间 ===")
    print(f"救援时间: {rescue_time:.2f} 秒")
    print(f"灭火时间: {fire_time:.2f} 秒")
    print(f"总时间: {total_time:.2f} 秒")

if __name__ == "__main__":
    main()