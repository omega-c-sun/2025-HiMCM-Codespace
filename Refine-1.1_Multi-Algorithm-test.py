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
        
        # 计算房间宽度，确保所有房间都能适应建筑宽度
        total_rooms = self.rooms_per_side * 2
        room_width = self.cols // total_rooms
        
        all_rooms = {}
        all_doors = {}
        
        # 生成房间布局
        for room_id in range(1, total_rooms + 1):
            if room_id <= self.rooms_per_side:
                # 上层房间 (0-4行)
                start_row = 0
                end_row = 4
                start_col = (room_id - 1) * room_width
                end_col = start_col + room_width - 1
                # 门的位置在房间底部中间
                door_row = end_row
                door_col = start_col + room_width // 2
            else:
                # 下层房间 (10-14行)
                start_row = 10
                end_row = 14
                lower_room_id = room_id - self.rooms_per_side
                start_col = (lower_room_id - 1) * room_width
                end_col = start_col + room_width - 1
                # 门的位置在房间顶部中间
                door_row = start_row
                door_col = start_col + room_width // 2
            
            # 确保边界在范围内，特别是最后一个房间
            if room_id == self.rooms_per_side or room_id == total_rooms:
                end_col = self.cols - 1
            
            all_rooms[room_id] = {
                "start_row": start_row, 
                "end_row": end_row, 
                "start_col": start_col, 
                "end_col": end_col
            }
            all_doors[room_id] = (door_row, door_col)
        
        for floor in range(self.floors):
            # 初始化建筑为空地
            building = [[0 for _ in range(self.cols)] for _ in range(self.rows)]
            
            # 添加房间
            for room_id, room_info in all_rooms.items():
                start_row = room_info["start_row"]
                end_row = room_info["end_row"]
                start_col = room_info["start_col"]
                end_col = room_info["end_col"]
                
                # 生成房间内容
                room_rows = end_row - start_row + 1
                room_cols = end_col - start_col + 1
                
                if room_rows > 0 and room_cols > 0:
                    # 直接生成房间内容到建筑中
                    self._generate_room_content(building, start_row, end_row, start_col, end_col)
                
                # 添加门（确保门的位置是空地）
                door_row, door_col = all_doors[room_id]
                if 0 <= door_row < self.rows and 0 <= door_col < self.cols:
                    building[door_row][door_col] = 0
            
            # 走廊区域 (5-9行) - 确保是空地
            for i in range(5, 10):
                for j in range(self.cols):
                    building[i][j] = 0
            
            # 楼梯位置
            building[7][1] = 4  # 左侧楼梯
            building[7][self.cols - 2] = 4  # 右侧楼梯
            
            # 添加障碍物
            building = self._add_obstacles(building)
            
            building_3d.append(building)
            room_layouts.append(all_rooms.copy())
            door_layouts.append(all_doors.copy())
        
        # 出口位置
        if len(building_3d) > 0:
            building_3d[0][7][0] = 2  # 左侧出口
            building_3d[0][7][self.cols - 1] = 3  # 右侧出口
        
        return building_3d, room_layouts, door_layouts
    
    def _generate_room_content(self, building, start_row, end_row, start_col, end_col):
        """直接在建筑中生成房间内容"""
        rows = end_row - start_row + 1
        cols = end_col - start_col + 1
        
        if rows <= 0 or cols <= 0:
            return
        
        # 计算火和人的数量 - 确保至少有一些
        fire_count = max(1, int(rows * cols * 0.3))  # 至少1个火
        people_count = max(1, int(rows * cols * 0.1))  # 至少1个人
        
        # 生成所有可能的位置
        all_positions = []
        for i in range(rows):
            for j in range(cols):
                # 计算实际在建筑中的位置
                actual_row = start_row + i
                actual_col = start_col + j
                
                # 确保位置在建筑范围内
                if 0 <= actual_row < self.rows and 0 <= actual_col < self.cols:
                    # 避开门的附近位置
                    is_near_door = False
                    # 上层房间门在底部中间
                    if start_row == 0 and i == rows-1 and j == cols//2:
                        is_near_door = True
                    # 下层房间门在顶部中间
                    if start_row == 10 and i == 0 and j == cols//2:
                        is_near_door = True
                    
                    if not is_near_door:
                        all_positions.append((actual_row, actual_col))
        
        if not all_positions:
            return
        
        # 添加火源
        if len(all_positions) >= fire_count:
            fire_positions = random.sample(all_positions, fire_count)
            for row, col in fire_positions:
                building[row][col] = -1
                # 从可用位置中移除，避免人和火在同一位置
                if (row, col) in all_positions:
                    all_positions.remove((row, col))
        
        # 添加人员（在剩余位置中）
        if all_positions and len(all_positions) >= people_count:
            people_positions = random.sample(all_positions, people_count)
            for row, col in people_positions:
                building[row][col] = 1
    
    def _add_obstacles(self, building, obstacle_ratio=0.03):
        rows = len(building)
        cols = len(building[0])
        
        obstacle_count = int(rows * cols * obstacle_ratio)
        
        available_positions = []
        for i in range(rows):
            for j in range(cols):
                # 只在空地和有内容的区域添加障碍物，避开门、楼梯、出口
                if building[i][j] in [0, -1, 1] and (i, j) not in [(7, 0), (7, cols-1), (7, 1), (7, cols-2)]:
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
        
        # 动态计算右侧救援者起始位置（基于建筑宽度）
        if len(building_3d) > 0 and len(building_3d[0]) > 0:
            right_start_col = len(building_3d[0][0]) - 1
            self.right_rescuer_start = (0, 7, right_start_col)
        else:
            self.right_rescuer_start = (0, 7, 35)
        
        # 动态计算出口位置
        self.exits = []
        for floor in range(len(building_3d)):
            self.exits.append((floor, 7, 0))  # 左侧出口
            if len(building_3d[floor]) > 0:
                right_exit_col = len(building_3d[floor][0]) - 1
                self.exits.append((floor, 7, right_exit_col))  # 右侧出口
    
    def get_room_info(self):
        room_info = {}
        
        for floor in range(len(self.rooms_3d)):
            for room_id, room_data in self.rooms_3d[floor].items():
                start_row = room_data["start_row"]
                end_row = room_data["end_row"]
                start_col = room_data["start_col"]
                end_col = room_data["end_col"]
                
                # 确保房间在建筑范围内
                if (start_row < 0 or end_row >= len(self.building_3d[floor]) or 
                    start_col < 0 or end_col >= len(self.building_3d[floor][0])):
                    continue
                
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
        if room_key not in room_info:
            return float('inf')
            
        room_data = room_info[room_key]
        door_pos = room_data["door_position"]
        
        path_to_door, time_to_door = self.finder.find_path(start_pos, door_pos, has_people=False)
        
        room_time = self._calculate_room_rescue_time(room_key, room_info)
        
        exit_times = []
        for exit_pos in self.exits:
            _, time_to_exit = self.finder.find_path(door_pos, exit_pos, has_people=True)
            exit_times.append(time_to_exit)
        
        min_exit_time = min(exit_times) if exit_times else float('inf')
        
        total_time = time_to_door + room_time + min_exit_time
        return total_time
    
    def _calculate_room_rescue_time(self, room_key, room_info):
        if room_key not in room_info:
            return 0
            
        room_data = room_info[room_key]
        floor = room_data["floor"]
        room_id = room_data["room_id"]
        door_pos = room_data["door_position"]
        
        if floor >= len(self.rooms_3d) or room_id not in self.rooms_3d[floor]:
            return 0
            
        room_layout = self.rooms_3d[floor][room_id]
        people_positions = []
        
        # 确保房间在建筑范围内
        if (room_layout["start_row"] < 0 or room_layout["end_row"] >= len(self.building_3d[floor]) or
            room_layout["start_col"] < 0 or room_layout["end_col"] >= len(self.building_3d[floor][0])):
            return 0
            
        for i in range(room_layout["start_row"], room_layout["end_row"] + 1):
            for j in range(room_layout["start_col"], room_layout["end_col"] + 1):
                if self.building_3d[floor][i][j] == 1:
                    people_positions.append((floor, i, j))
        
        if not people_positions:
            return 2 * self.finder.movement_speed
        
        base_time = len(people_positions) * 2 * self.finder.movement_speed
        fire_penalty = room_data["fire_count"] * self.finder.fire_extinguish_time
        
        return base_time + fire_penalty

class MultiAStarRescuePlanner:
    def __init__(self, building_3d, rooms_3d, doors_3d):
        self.planner = RescuePlanner(building_3d, rooms_3d, doors_3d)
        self.room_info = self.planner.get_room_info()
    
    def solve(self):
        """使用多起点A* + 贪心策略"""
        room_keys = list(self.room_info.keys())
        
        if not room_keys:
            return [], [], self.room_info
        
        # 计算每个房间到两个起点的代价
        room_costs = {}
        for room_key in room_keys:
            left_cost = self.planner.calculate_rescue_time(
                self.planner.left_rescuer_start, room_key, self.room_info)
            right_cost = self.planner.calculate_rescue_time(
                self.planner.right_rescuer_start, room_key, self.room_info)
            room_costs[room_key] = {
                'left': left_cost,
                'right': right_cost,
                'importance': self.room_info[room_key]['importance'],
                'cost_diff': abs(left_cost - right_cost)
            }
        
        # 贪心分配：优先处理重要性高、代价差异大的房间
        sorted_rooms = sorted(room_keys, 
                            key=lambda x: (-room_costs[x]['importance'], 
                                         -room_costs[x]['cost_diff']))
        
        left_rooms = []
        right_rooms = []
        left_total_time = 0
        right_total_time = 0
        
        for room_key in sorted_rooms:
            left_time = room_costs[room_key]['left']
            right_time = room_costs[room_key]['right']
            
            # 选择使得总时间增长较小的救援者
            if left_total_time + left_time <= right_total_time + right_time:
                left_rooms.append(room_key)
                left_total_time += left_time
            else:
                right_rooms.append(room_key)
                right_total_time += right_time
        
        # 对每个救援者的房间序列进行局部优化
        left_rooms = self._optimize_sequence(left_rooms, self.planner.left_rescuer_start)
        right_rooms = self._optimize_sequence(right_rooms, self.planner.right_rescuer_start)
        
        return left_rooms, right_rooms, self.room_info
    
    def _optimize_sequence(self, rooms, start_pos):
        """使用最近邻算法优化序列"""
        if len(rooms) <= 1:
            return rooms
        
        current_pos = start_pos
        unvisited = set(rooms)
        optimized_sequence = []
        
        while unvisited:
            # 找到距离当前位置最近的房间
            nearest_room = min(unvisited, 
                             key=lambda x: self._estimate_distance(current_pos, x))
            optimized_sequence.append(nearest_room)
            unvisited.remove(nearest_room)
            current_pos = self.room_info[nearest_room]['door_position']
        
        return optimized_sequence
    
    def _estimate_distance(self, pos1, room_key):
        """快速估算距离（避免完整路径计算）"""
        pos2 = self.room_info[room_key]['door_position']
        floor_dist = abs(pos1[0] - pos2[0]) * 20  # 楼梯代价
        manhattan_dist = abs(pos1[1] - pos2[1]) + abs(pos1[2] - pos2[2])
        return floor_dist + manhattan_dist

class AlgorithmComparator:
    def __init__(self, building_3d, rooms_3d, doors_3d):
        self.building_3d = building_3d
        self.rooms_3d = rooms_3d
        self.doors_3d = doors_3d
        
    def compare_algorithms(self):
        """比较所有算法性能"""
        algorithms = {
            "MultiAStar": MultiAStarRescuePlanner,
        }
        
        results = {}
        
        for name, planner_class in algorithms.items():
            try:
                print(f"正在测试 {name} 算法...")
                planner = planner_class(self.building_3d, self.rooms_3d, self.doors_3d)
                
                start_time = time.time()
                left_order, right_order, room_info = planner.solve()
                computation_time = time.time() - start_time
                
                # 计算救援时间
                rescue_time = self._calculate_rescue_time(left_order, right_order, room_info)
                
                results[name] = {
                    'computation_time': computation_time,
                    'rescue_time': rescue_time,
                    'solution_quality': len(left_order) + len(right_order),
                    'left_rooms': left_order,
                    'right_rooms': right_order,
                    'room_info': room_info
                }
                
                print(f"  {name}: 计算时间={computation_time:.4f}s, 救援时间={rescue_time:.2f}s")
                
            except Exception as e:
                print(f"  {name} 算法出错: {e}")
                results[name] = None
        
        return results
    
    def _calculate_rescue_time(self, left_rooms, right_rooms, room_info):
        """计算救援时间"""
        planner = RescuePlanner(self.building_3d, self.rooms_3d, self.doors_3d)
        
        left_time = 0
        current_pos = planner.left_rescuer_start
        for room_key in left_rooms:
            # 添加错误处理，防止路径查找失败
            try:
                room_time = planner.calculate_rescue_time(current_pos, room_key, room_info)
                if room_time == float('inf'):
                    print(f"警告: 无法找到从 {current_pos} 到房间 {room_key} 的路径")
                    room_time = 1000  # 设置一个很大的惩罚值
                left_time += room_time
                current_pos = room_info[room_key]["door_position"]
            except Exception as e:
                print(f"计算左侧房间 {room_key} 救援时间时出错: {e}")
                left_time += 1000  # 设置一个很大的惩罚值
        
        right_time = 0
        current_pos = planner.right_rescuer_start
        for room_key in right_rooms:
            # 添加错误处理，防止路径查找失败
            try:
                room_time = planner.calculate_rescue_time(current_pos, room_key, room_info)
                if room_time == float('inf'):
                    print(f"警告: 无法找到从 {current_pos} 到房间 {room_key} 的路径")
                    room_time = 1000  # 设置一个很大的惩罚值
                right_time += room_time
                current_pos = room_info[room_key]["door_position"]
            except Exception as e:
                print(f"计算右侧房间 {room_key} 救援时间时出错: {e}")
                right_time += 1000  # 设置一个很大的惩罚值
        
        return max(left_time, right_time)
    
    def select_best_algorithm(self, results):
        """选择最优算法（时间为最高优先级）"""
        valid_results = {k: v for k, v in results.items() if v is not None}
        
        if not valid_results:
            return None, None
        
        # 时间为最高优先级，选择计算时间最短的算法
        best_algorithm = min(valid_results.keys(), 
                           key=lambda x: valid_results[x]['computation_time'])
        
        return best_algorithm, valid_results[best_algorithm]

class FireExtinguisher3D:
    """多层建筑灭火规划器"""
    
    def __init__(self, building_3d, rooms_3d, doors_3d):
        self.building_3d = building_3d
        self.rooms_3d = rooms_3d
        self.doors_3d = doors_3d
        self.finder = PathFinder(building_3d)
        
        # 动态计算救援者起始位置
        self.left_rescuer_start = (0, 7, 0)
        if len(building_3d) > 0 and len(building_3d[0]) > 0:
            right_start_col = len(building_3d[0][0]) - 1
            self.right_rescuer_start = (0, 7, right_start_col)
        else:
            self.right_rescuer_start = (0, 7, 35)
        
        # 动态计算出口位置
        self.exits = []
        for floor in range(len(building_3d)):
            self.exits.append((floor, 7, 0))
            if len(building_3d[floor]) > 0:
                right_exit_col = len(building_3d[floor][0]) - 1
                self.exits.append((floor, 7, right_exit_col))
    
    def get_fire_locations(self):
        """获取所有楼层火的位置"""
        fire_locations = []
        for floor in range(len(self.building_3d)):
            for i in range(len(self.building_3d[floor])):
                for j in range(len(self.building_3d[floor][0])):
                    if self.building_3d[floor][i][j] == -1:
                        fire_locations.append((floor, i, j))
        return fire_locations
    
    def optimize_fire_extinguish_plan(self):
        """优化多层建筑灭火计划"""
        fire_locations = self.get_fire_locations()
        
        if not fire_locations:
            return [], [], 0
        
        # 将火分配给两个救援者
        left_fires, right_fires = self._assign_fires(fire_locations)
        
        # 优化每个救援者的灭火顺序
        left_plan, left_time = self._optimize_fire_sequence_3d(left_fires, self.left_rescuer_start)
        right_plan, right_time = self._optimize_fire_sequence_3d(right_fires, self.right_rescuer_start)
        
        return left_plan, right_plan, max(left_time, right_time)
    
    def _assign_fires(self, fire_locations):
        """将火分配给两个救援者（考虑楼层）"""
        if not fire_locations:
            return [], []
        
        left_fires = []
        right_fires = []
        
        for fire in fire_locations:
            # 计算到左侧和右侧的距离（考虑楼层）
            left_dist = self._calculate_3d_distance(fire, self.left_rescuer_start)
            right_dist = self._calculate_3d_distance(fire, self.right_rescuer_start)
            
            if left_dist <= right_dist:
                left_fires.append(fire)
            else:
                right_fires.append(fire)
        
        return left_fires, right_fires
    
    def _calculate_3d_distance(self, pos1, pos2):
        """计算3D距离（考虑楼层移动成本）"""
        floor_diff = abs(pos1[0] - pos2[0])
        row_diff = abs(pos1[1] - pos2[1])
        col_diff = abs(pos1[2] - pos2[2])
        
        # 楼层移动有额外成本
        return floor_diff * 10 + row_diff + col_diff
    
    def _optimize_fire_sequence_3d(self, fires, start_pos):
        """优化多层灭火顺序（使用最近邻算法）"""
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
            min_heuristic_distance = float('inf')
            
            for fire in unvisited:
                # 使用3D启发式距离
                heuristic_distance = self._calculate_3d_distance(current_pos, fire)
                if heuristic_distance < min_heuristic_distance:
                    min_heuristic_distance = heuristic_distance
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
        
        if door_floor < len(building_3d) and door_row < len(building_3d[door_floor]) and door_col < len(building_3d[door_floor][0]):
            for i in range(max(0, door_row-1), min(len(building_3d[door_floor]), door_row+2)):
                for j in range(max(0, door_col-1), min(len(building_3d[door_floor][0]), door_col+2)):
                    if building_3d[door_floor][i][j] == -3:
                        obstacle_count += 1
        
        print(f"  {floor}-{room_id}    |   {info['people_count']}   |   {info['fire_count']}   |   {obstacle_count}   |   {info['importance']}   | {info['door_position']}")

def print_rescue_plan(left_order, right_order, room_info, algorithm_name):
    print(f"\n=== 最优救援计划 (使用{algorithm_name}算法) ===")
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

def print_fire_extinguish_plan_3d(left_plan, right_plan, fire_time, building_3d):
    """打印多层灭火计划"""
    print("\n=== 灭火计划 ===")
    
    print(f"\n预计灭火时间: {fire_time:.2f} 秒")
    
    # 统计左侧救援者灭火点（按楼层分组）
    left_fires = [step[1] for step in left_plan if isinstance(step, tuple) and step[0] == "EXTINGUISH"]
    left_fires_by_floor = {}
    for fire in left_fires:
        floor = fire[0]
        if floor not in left_fires_by_floor:
            left_fires_by_floor[floor] = []
        left_fires_by_floor[floor].append(fire)
    
    print(f"\n左侧救援者灭火点 ({len(left_fires)} 个):")
    for floor in sorted(left_fires_by_floor.keys()):
        print(f"  楼层 {floor+1}:")
        for i, fire_pos in enumerate(left_fires_by_floor[floor]):
            print(f"    {i+1}. 位置 ({fire_pos[1]}, {fire_pos[2]})")
    
    # 统计右侧救援者灭火点（按楼层分组）
    right_fires = [step[1] for step in right_plan if isinstance(step, tuple) and step[0] == "EXTINGUISH"]
    right_fires_by_floor = {}
    for fire in right_fires:
        floor = fire[0]
        if floor not in right_fires_by_floor:
            right_fires_by_floor[floor] = []
        right_fires_by_floor[floor].append(fire)
    
    print(f"\n右侧救援者灭火点 ({len(right_fires)} 个):")
    for floor in sorted(right_fires_by_floor.keys()):
        print(f"  楼层 {floor+1}:")
        for i, fire_pos in enumerate(right_fires_by_floor[floor]):
            print(f"    {i+1}. 位置 ({fire_pos[1]}, {fire_pos[2]})")

def main():
    ans =0
    fire_ans =0
    for _ in range(100):
        generator = BuildingGenerator()
        building_3d, rooms_3d, doors_3d = generator.generate_building()
        comparator = AlgorithmComparator(building_3d, rooms_3d, doors_3d)
        results = comparator.compare_algorithms()
        best_algorithm, best_result = comparator.select_best_algorithm(results)
        left_order = best_result['left_rooms']
        right_order = best_result['right_rooms']
        room_info = best_result['room_info']
        fire_planner = FireExtinguisher3D(building_3d, rooms_3d, doors_3d)
        left_fire_plan, right_fire_plan, fire_time = fire_planner.optimize_fire_extinguish_plan()
        total_time = best_result['rescue_time'] + fire_time
        ans += total_time
        fire_ans += fire_time
    print(f"平均救援时间: {ans/100:.2f} 秒")
    print(f"平均灭火时间: {fire_ans/100:.2f} 秒")

if __name__ == "__main__":
    main()