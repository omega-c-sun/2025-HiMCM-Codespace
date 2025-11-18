import heapq
import math
import random
import numpy as np
from collections import defaultdict
import time

class PathFinder:
    def __init__(self, grid_3d):
        self.grid_3d = grid_3d
        self.floors = len(grid_3d)
        self.rows = len(grid_3d[0])
        self.cols = len(grid_3d[0][0])
        self.movement_speed = 1/6
        self.fire_extinguish_time = 5
        self.stair_time = 20
        
    def find_path(self, start, end, has_people=False):
        def heuristic(a, b):
            floor_dist = abs(a[0]-b[0]) * self.stair_time / self.movement_speed
            row_dist = abs(a[1]-b[1])
            col_dist = abs(a[2]-b[2])
            return floor_dist + row_dist + col_dist
        
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
                path = [current]
                total_time = g_score[(current, current_has_people)]
                while current in came_from:
                    current, current_has_people = came_from[(current, current_has_people)]
                    path.append(current)
                path.reverse()
                return path, total_time
            
            for neighbor in self._get_neighbors(current):
                nf, nx, ny = neighbor
                
                if self.grid_3d[nf][nx][ny] == -3:
                    continue
                
                move_cost = self.movement_speed
                
                if current[0] != nf:
                    move_cost += self.stair_time
                
                if self.grid_3d[nf][nx][ny] == -1 and current_has_people:
                    move_cost += self.fire_extinguish_time
                
                new_has_people = current_has_people
                if self.grid_3d[nf][nx][ny] == 1:
                    new_has_people = True
                
                tentative_g_score = g_score[(current, current_has_people)] + move_cost
                
                if tentative_g_score < g_score[(neighbor, new_has_people)]:
                    came_from[(neighbor, new_has_people)] = (current, current_has_people)
                    g_score[(neighbor, new_has_people)] = tentative_g_score
                    f_score[(neighbor, new_has_people)] = tentative_g_score + heuristic(neighbor, end)
                    heapq.heappush(open_set, (f_score[(neighbor, new_has_people)], neighbor, new_has_people))
        
        return [], float('inf')
    
    def _get_neighbors(self, pos):
        floor, x, y = pos
        neighbors = []
        
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.rows and 0 <= ny < self.cols:
                neighbors.append((floor, nx, ny))
        
        if self.grid_3d[floor][x][y] == 4:
            if floor > 0:
                neighbors.append((floor-1, x, y))
            if floor < self.floors - 1:
                neighbors.append((floor+1, x, y))
        
        return neighbors

class BuildingGenerator:
    def __init__(self):
        self.floors = random.randint(3, 6)
        self.rows = 15
        self.cols = 36
        self.rooms_per_side = random.randint(2, 5)
    
    def generate_building(self):
        building_3d = []
        room_layouts = []
        door_layouts = []
        
        room_width = self.cols // (self.rooms_per_side * 2)
        
        all_rooms = {}
        all_doors = {}
        
        for room_id in range(1, self.rooms_per_side * 2 + 1):
            if room_id <= self.rooms_per_side:
                start_row = 0
                end_row = 5
                room_col = (room_id - 1) * room_width
                door_row = 5
                door_col = room_col + room_width // 2
            else:
                start_row = 9
                end_row = 14
                room_col = (room_id - self.rooms_per_side - 1) * room_width
                door_row = 9
                door_col = room_col + room_width // 2
            
            all_rooms[room_id] = {
                "start_row": start_row, 
                "end_row": end_row, 
                "start_col": room_col, 
                "end_col": room_col + room_width - 1
            }
            all_doors[room_id] = (door_row, door_col)
        
        for floor in range(self.floors):
            building = [[-2 for _ in range(self.cols)] for _ in range(self.rows)]
            
            hallway_start_row = 6
            hallway_end_row = 8
            
            for i in range(hallway_start_row, hallway_end_row + 1):
                for j in range(self.cols):
                    building[i][j] = 0
            
            for room_id, room_info in all_rooms.items():
                room_rows = room_info["end_row"] - room_info["start_row"] + 1
                room_cols = room_info["end_col"] - room_info["start_col"] + 1
                
                room_grid = self._generate_room_layout(room_rows, room_cols)
                
                for i in range(room_rows):
                    for j in range(room_cols):
                        building[room_info["start_row"] + i][room_info["start_col"] + j] = room_grid[i][j]
                
                door_row, door_col = all_doors[room_id]
                building[door_row][door_col] = 0
            
            building[7][1] = 4
            building[7][34] = 4
            
            building = self._add_obstacles(building)
            
            building_3d.append(building)
            room_layouts.append(all_rooms.copy())
            door_layouts.append(all_doors.copy())
        
        building_3d[0][7][0] = 2
        building_3d[0][7][35] = 3
        
        return building_3d, room_layouts, door_layouts
    
    def _generate_room_layout(self, rows=6, cols=12, fire_ratio=0.5, people_ratio=0.1):
        total_cells = rows * cols
        
        fire_count = int(total_cells * fire_ratio)
        people_count = int(total_cells * people_ratio)
        
        room = [[0 for _ in range(cols)] for _ in range(rows)]
        
        all_positions = [(i, j) for i in range(rows) for j in range(cols)]
        
        if fire_count > 0:
            fire_positions = random.sample(all_positions, fire_count)
            for i, j in fire_positions:
                room[i][j] = -1
        
        non_fire_positions = [pos for pos in all_positions if room[pos[0]][pos[1]] != -1]
        if people_count > 0 and len(non_fire_positions) > 0:
            people_count = min(people_count, len(non_fire_positions))
            people_positions = random.sample(non_fire_positions, people_count)
            for i, j in people_positions:
                room[i][j] = 1
        
        return room
    
    def _add_obstacles(self, building, obstacle_ratio=0.05):
        rows = len(building)
        cols = len(building[0])
        
        obstacle_count = int(rows * cols * obstacle_ratio)
        
        available_positions = []
        for i in range(rows):
            for j in range(cols):
                if building[i][j] in [0, -1, 1]:
                    available_positions.append((i, j))
        
        if available_positions and obstacle_count > 0:
            obstacle_count = min(obstacle_count, len(available_positions))
            obstacle_positions = random.sample(available_positions, obstacle_count)
            
            for i, j in obstacle_positions:
                building[i][j] = -3
        
        return building

class RescuePlanner:
    def __init__(self, building_3d, rooms_3d, doors_3d):
        self.building_3d = building_3d
        self.rooms_3d = rooms_3d
        self.doors_3d = doors_3d
        self.finder = PathFinder(building_3d)
        
        self.left_rescuer_start = (0, 7, 0)
        self.right_rescuer_start = (0, 7, 35)
        
        self.exits = []
        for floor in range(len(building_3d)):
            self.exits.append((floor, 7, 0))
            self.exits.append((floor, 7, 35))
    
    def get_room_info(self):
        room_info = {}
        
        for floor in range(len(self.rooms_3d)):
            for room_id, room_data in self.rooms_3d[floor].items():
                start_row = room_data["start_row"]
                end_row = room_data["end_row"]
                start_col = room_data["start_col"]
                end_col = room_data["end_col"]
                
                room_area = []
                for i in range(start_row, end_row + 1):
                    row = []
                    for j in range(start_col, end_col + 1):
                        row.append(self.building_3d[floor][i][j])
                    room_area.append(row)
                
                people_count = sum(row.count(1) for row in room_area)
                fire_count = sum(row.count(-1) for row in room_area)
                
                center_row = (start_row + end_row) // 2
                center_col = (start_col + end_col) // 2
                
                room_key = f"{floor}_{room_id}"
                
                room_info[room_key] = {
                    "floor": floor,
                    "room_id": room_id,
                    "people_count": people_count,
                    "fire_count": fire_count,
                    "area": (end_row - start_row + 1) * (end_col - start_col + 1),
                    "door_position": (floor, self.doors_3d[floor][room_id][0], self.doors_3d[floor][room_id][1]),
                    "center_position": (floor, center_row, center_col),
                    "importance": people_count * 10 + fire_count * 2
                }
        
        return room_info
    
    def calculate_rescue_time(self, start_pos, room_key, room_info):
        room_data = room_info[room_key]
        door_pos = room_data["door_position"]
        
        path_to_door, time_to_door = self.finder.find_path(start_pos, door_pos, has_people=False)
        
        room_time = self._calculate_room_rescue_time(room_key, room_info)
        
        exit_times = []
        for exit_pos in self.exits:
            _, time_to_exit = self.finder.find_path(door_pos, exit_pos, has_people=True)
            exit_times.append(time_to_exit)
        
        min_exit_time = min(exit_times)
        
        total_time = time_to_door + room_time + min_exit_time
        return total_time
    
    def _calculate_room_rescue_time(self, room_key, room_info):
        room_data = room_info[room_key]
        floor = room_data["floor"]
        room_id = room_data["room_id"]
        door_pos = room_data["door_position"]
        
        room_layout = self.rooms_3d[floor][room_id]
        people_positions = []
        for i in range(room_layout["start_row"], room_layout["end_row"] + 1):
            for j in range(room_layout["start_col"], room_layout["end_col"] + 1):
                if self.building_3d[floor][i][j] == 1:
                    people_positions.append((floor, i, j))
        
        if not people_positions:
            return 2 * self.finder.movement_speed
        
        base_time = len(people_positions) * 2 * self.finder.movement_speed
        fire_penalty = room_data["fire_count"] * self.finder.fire_extinguish_time
        
        return base_time + fire_penalty
    
    def optimize_rescue_order(self):
        room_info = self.get_room_info()
        
        left_rescue_times = {}
        right_rescue_times = {}
        
        for room_key in room_info.keys():
            left_rescue_times[room_key] = self.calculate_rescue_time(self.left_rescuer_start, room_key, room_info)
            right_rescue_times[room_key] = self.calculate_rescue_time(self.right_rescuer_start, room_key, room_info)
        
        left_rooms, right_rooms = self._assign_rooms(room_info, left_rescue_times, right_rescue_times)
        
        left_order = self._optimize_sequence(left_rooms, room_info, left_rescue_times, self.left_rescuer_start)
        right_order = self._optimize_sequence(right_rooms, room_info, right_rescue_times, self.right_rescuer_start)
        
        return left_order, right_order, room_info
    
    def _assign_rooms(self, room_info, left_times, right_times):
        room_costs = {}
        for room_key in room_info.keys():
            cost_diff = abs(left_times[room_key] - right_times[room_key])
            importance = room_info[room_key]["importance"]
            room_costs[room_key] = (cost_diff, importance, left_times[room_key], right_times[room_key])
        
        sorted_rooms = sorted(room_costs.keys(), 
                             key=lambda x: (room_costs[x][0], -room_costs[x][1]), 
                             reverse=True)
        
        left_rooms = []
        right_rooms = []
        left_total_time = 0
        right_total_time = 0
        
        for room_key in sorted_rooms:
            left_time = left_times[room_key]
            right_time = right_times[room_key]
            
            if left_total_time + left_time <= right_total_time + right_time:
                left_rooms.append(room_key)
                left_total_time += left_time
            else:
                right_rooms.append(room_key)
                right_total_time += right_time
        
        return left_rooms, right_rooms
    
    def _optimize_sequence(self, rooms, room_info, rescue_times, start_pos):
        if not rooms:
            return []
        
        room_scores = {}
        for room_key in rooms:
            importance = room_info[room_key]["importance"]
            time_cost = rescue_times[room_key]
            room_scores[room_key] = importance / time_cost if time_cost > 0 else float('inf')
        
        sorted_rooms = sorted(rooms, key=lambda x: room_scores[x], reverse=True)
        
        return sorted_rooms

class GeneticRescuePlanner:
    def __init__(self, building_3d, rooms_3d, doors_3d, population_size=50, generations=100):
        self.planner = RescuePlanner(building_3d, rooms_3d, doors_3d)
        self.population_size = population_size
        self.generations = generations
        self.room_info = self.planner.get_room_info()
        
    def solve(self):
        room_keys = list(self.room_info.keys())
        
        if not room_keys:
            return [], [], self.room_info
        
        population = self._initialize_population(room_keys)
        
        best_left = None
        best_right = None
        best_time = float('inf')
        
        for generation in range(self.generations):
            fitness = [self._evaluate_fitness(ind) for ind in population]
            
            min_fitness = min(fitness)
            if min_fitness < best_time:
                best_time = min_fitness
                best_idx = fitness.index(min_fitness)
                best_left, best_right = population[best_idx]
            
            new_population = []
            for _ in range(self.population_size // 2):
                parent1 = self._select_parent(population, fitness)
                parent2 = self._select_parent(population, fitness)
                child1, child2 = self._crossover(parent1, parent2, room_keys)
                child1 = self._mutate(child1, room_keys)
                child2 = self._mutate(child2, room_keys)
                new_population.extend([child1, child2])
            
            population = new_population
        
        return best_left, best_right, self.room_info
    
    def _initialize_population(self, room_keys):
        population = []
        for _ in range(self.population_size):
            left_rooms = random.sample(room_keys, random.randint(0, len(room_keys)))
            right_rooms = [r for r in room_keys if r not in left_rooms]
            
            random.shuffle(left_rooms)
            random.shuffle(right_rooms)
            
            population.append((left_rooms, right_rooms))
        return population
    
    def _evaluate_fitness(self, individual):
        left_rooms, right_rooms = individual
        
        left_time = 0
        current_pos = self.planner.left_rescuer_start
        for room_key in left_rooms:
            left_time += self.planner.calculate_rescue_time(current_pos, room_key, self.room_info)
            current_pos = self.room_info[room_key]["door_position"]
        
        right_time = 0
        current_pos = self.planner.right_rescuer_start
        for room_key in right_rooms:
            right_time += self.planner.calculate_rescue_time(current_pos, room_key, self.room_info)
            current_pos = self.room_info[room_key]["door_position"]
        
        return max(left_time, right_time)
    
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
    
    def _crossover(self, parent1, parent2, room_keys):
        left1, right1 = parent1
        left2, right2 = parent2
        
        all_rooms = list(set(left1 + right1))
        
        sample_size1 = random.randint(0, len(all_rooms))
        child1_left = random.sample(all_rooms, sample_size1) if all_rooms else []
        child1_right = [r for r in all_rooms if r not in child1_left]
        
        sample_size2 = random.randint(0, len(all_rooms))
        child2_left = random.sample(all_rooms, sample_size2) if all_rooms else []
        child2_right = [r for r in all_rooms if r not in child2_left]
        
        return (child1_left, child1_right), (child2_left, child2_right)
    
    def _mutate(self, individual, room_keys):
        left, right = individual
        
        if random.random() < 0.2:
            if left and right:
                room_from_left = random.choice(left)
                room_from_right = random.choice(right)
                left.remove(room_from_left)
                right.remove(room_from_right)
                left.append(room_from_right)
                right.append(room_from_left)
            elif left and not right:
                room_to_move = random.choice(left)
                left.remove(room_to_move)
                right.append(room_to_move)
            elif not left and right:
                room_to_move = random.choice(right)
                right.remove(room_to_move)
                left.append(room_to_move)
        
        random.shuffle(left)
        random.shuffle(right)
        
        return (left, right)

def print_building_floor(building, floor=0):
    symbols = {-3: '█', -2: '▓', -1: '🔥', 0: '·', 1: '👤', 2: '🚒', 3: '🚑', 4: '↕'}
    
    print(f"\n楼层 {floor+1} 布局:")
    print("█ = 障碍物, ▓ = 墙, 🔥 = 火, · = 空地, 👤 = 人, 🚒 = 左侧救援者, 🚑 = 右侧救援者, ↕ = 楼梯")
    print("-" * (len(building[0]) * 3 + 1))
    
    for i, row in enumerate(building[floor]):
        print(f"{i:2d}|", end="")
        for cell in row:
            print(f" {symbols[cell]}", end="")
        print(" |")
    
    print("-" * (len(building[0]) * 3 + 1))

def print_room_info(room_info, building_3d):
    print("\n房间信息:")
    print("楼层-房间号 | 人员数 | 火数量 | 障碍物 | 重要性 | 门位置")
    print("-" * 70)
    for room_key in sorted(room_info.keys()):
        info = room_info[room_key]
        floor = info["floor"]
        room_id = info["room_id"]
        
        obstacle_count = 0
        door_pos = info["door_position"]
        door_floor, door_row, door_col = door_pos
        
        for i in range(max(0, door_row-1), min(len(building_3d[door_floor]), door_row+2)):
            for j in range(max(0, door_col-1), min(len(building_3d[door_floor][0]), door_col+2)):
                if building_3d[door_floor][i][j] == -3:
                    obstacle_count += 1
        
        print(f"  {floor}-{room_id}    |   {info['people_count']}   |   {info['fire_count']}   |   {obstacle_count}   |   {info['importance']}   | {info['door_position']}")

def print_rescue_plan(left_order, right_order, room_info):
    print("\n=== 最优救援计划 ===")
    print("\n左侧救援者路线:")
    if left_order:
        for i, room_key in enumerate(left_order):
            info = room_info[room_key]
            floor = info["floor"]
            room_id = info["room_id"]
            print(f"  {i+1}. 楼层{floor}房间{room_id} (人员: {info['people_count']}, 火: {info['fire_count']}, 重要性: {info['importance']})")
    else:
        print("  无房间需要救援")
    
    print("\n右侧救援者路线:")
    if right_order:
        for i, room_key in enumerate(right_order):
            info = room_info[room_key]
            floor = info["floor"]
            room_id = info["room_id"]
            print(f"  {i+1}. 楼层{floor}房间{room_id} (人员: {info['people_count']}, 火: {info['fire_count']}, 重要性: {info['importance']})")
    else:
        print("  无房间需要救援")

def main():
    print("生成多层建筑布局...")
    generator = BuildingGenerator()
    building_3d, rooms_3d, doors_3d = generator.generate_building()
    
    print(f"建筑信息: {generator.floors}层, 每侧{generator.rooms_per_side}个房间")
    
    for floor in range(len(building_3d)):
        print_building_floor(building_3d, floor)
    
    print("\n使用遗传算法优化救援计划...")
    start_time = time.time()
    genetic_planner = GeneticRescuePlanner(building_3d, rooms_3d, doors_3d)
    left_order, right_order, room_info = genetic_planner.solve()
    genetic_time = time.time() - start_time
    
    print_room_info(room_info, building_3d)
    
    print_rescue_plan(left_order, right_order, room_info)
    
    print(f"\n计算时间: {genetic_time:.2f} 秒")
    
    planner = RescuePlanner(building_3d, rooms_3d, doors_3d)
    left_time = 0
    current_pos = planner.left_rescuer_start
    for room_key in left_order:
        left_time += planner.calculate_rescue_time(current_pos, room_key, room_info)
        current_pos = room_info[room_key]["door_position"]
    
    right_time = 0
    current_pos = planner.right_rescuer_start
    for room_key in right_order:
        right_time += planner.calculate_rescue_time(current_pos, room_key, room_info)
        current_pos = room_info[room_key]["door_position"]
    
    total_time = max(left_time, right_time)
    print(f"\n预计总救援时间: {total_time:.2f} 秒")
    print(f"左侧救援者时间: {left_time:.2f} 秒")
    print(f"右侧救援者时间: {right_time:.2f} 秒")

if __name__ == "__main__":
    main()