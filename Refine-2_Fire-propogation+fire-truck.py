import heapq
import math
import random
import numpy as np
from collections import defaultdict, deque
import time
import copy

class PathFinder:
    def __init__(self, grid_3d):
        self.grid_3d = grid_3d
        self.floors = len(grid_3d)
        self.rows = len(grid_3d[0])
        self.cols = len(grid_3d[0][0])
        self.normal_speed = 6.0
        self.smoke_speed = 3.0
        self.movement_speed = 1/self.normal_speed
        self.fire_extinguish_time = 5  # 5秒灭一格火
        self.stair_time = 10
        
    def find_path(self, start, end, has_people=False, smoke_grid=None):
        def get_smoke_penalty(smoke_level):
            if smoke_level < 0.1: return 1.0
            elif smoke_level < 0.3: return 1.5
            elif smoke_level < 0.6: return 2.0
            else: return 3.0
            
        def heuristic(a, b):
            floor_time = abs(a[0]-b[0]) * self.stair_time
            horizontal_dist = abs(a[1]-b[1]) + abs(a[2]-b[2])
            horizontal_time = horizontal_dist * self.movement_speed
            return floor_time + horizontal_time
        
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
                
                # 计算移动代价
                if smoke_grid and smoke_grid[nf][nx][ny] > 0.1:
                    smoke_level = smoke_grid[nf][nx][ny]
                    penalty = get_smoke_penalty(smoke_level)
                    move_cost = (1/self.smoke_speed) * penalty
                else:
                    move_cost = self.movement_speed
                
                # 如果目标位置有火，增加灭火时间
                if self.grid_3d[nf][nx][ny] == -1:
                    move_cost += self.fire_extinguish_time
                
                if current[0] != nf:
                    move_cost += self.stair_time
                
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

class ImprovedIntelligentPeopleMovementModel:
    def __init__(self, building_3d, rooms_3d, exits):
        self.building_3d = building_3d
        self.rooms_3d = rooms_3d
        self.exits = exits
        self.floors = len(building_3d)
        self.rows = len(building_3d[0])
        self.cols = len(building_3d[0][0])
        
        self.move_probability = 0.8
        self.panic_level = 0.2
        self.intelligence_level = 0.8
        self.time_passed = 0
        
        self.people_states = {}
        
        self.distance_field = self._compute_distance_field()
        
    def _compute_distance_field(self):
        distance_field = []
        
        for floor in range(self.floors):
            floor_distances = [[float('inf') for _ in range(self.cols)] for _ in range(self.rows)]
            
            queue = deque()
            for exit_pos in self.exits:
                if exit_pos[0] == floor:
                    queue.append((exit_pos[1], exit_pos[2], 0))
                    floor_distances[exit_pos[1]][exit_pos[2]] = 0
            
            while queue:
                i, j, dist = queue.popleft()
                
                for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    ni, nj = i + di, j + dj
                    
                    if (0 <= ni < self.rows and 0 <= nj < self.cols and
                        self.building_3d[floor][ni][nj] != -3 and
                        floor_distances[ni][nj] > dist + 1):
                        
                        floor_distances[ni][nj] = dist + 1
                        queue.append((ni, nj, dist + 1))
            
            distance_field.append(floor_distances)
        
        return distance_field
    
    def update_people_movement(self, elapsed_time, smoke_grid=None):
        self.time_passed = elapsed_time
        
        self._initialize_people_states()
        
        time_factor = min(1.0, elapsed_time / 60)
        current_move_prob = self.move_probability * (1 + time_factor * 0.5)
        
        self_rescue_count = 0
        people_moved = 0
        
        # 获取所有人员位置 - 确保只获取当前位置的人员
        people_positions = []
        for floor in range(self.floors):
            for i in range(self.rows):
                for j in range(self.cols):
                    if self.building_3d[floor][i][j] == 1:
                        people_positions.append((floor, i, j))
        
        # 随机打乱人员顺序，避免顺序偏差
        random.shuffle(people_positions)
        
        for pos in people_positions:
            floor, i, j = pos
            # 检查当前位置是否还有人员（可能已被其他移动改变）
            if self.building_3d[floor][i][j] != 1:
                continue
                
            if random.random() < current_move_prob:
                result = self._move_person_intelligently(floor, i, j, smoke_grid)
                if result == "self_rescue":
                    self_rescue_count += 1
                elif result:
                    people_moved += 1
        
        return self_rescue_count, people_moved
    
    def _initialize_people_states(self):
        # 清除已不存在的人员状态
        current_people = set()
        for floor in range(self.floors):
            for i in range(self.rows):
                for j in range(self.cols):
                    if self.building_3d[floor][i][j] == 1:
                        current_people.add((floor, i, j))
        
        # 移除不再存在的人员状态
        for pos in list(self.people_states.keys()):
            if pos not in current_people:
                del self.people_states[pos]
        
        # 为新人员添加状态
        for pos in current_people:
            if pos not in self.people_states:
                self.people_states[pos] = {
                    'total_fire_exposure': 0,
                    'consecutive_fire_exposure': 0,
                    'last_update_time': 0
                }
    
    def _move_person_intelligently(self, floor, i, j, smoke_grid=None):
        # 检查当前位置是否是出口
        if self.building_3d[floor][i][j] in [2, 3]:
            self.building_3d[floor][i][j] = 0
            return "self_rescue"
        
        possible_moves = []
        move_scores = []
        
        current_safe = self._is_position_safe(floor, i, j)
        
        for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            ni, nj = i + di, j + dj
            
            if (0 <= ni < self.rows and 0 <= nj < self.cols and
                self.building_3d[floor][ni][nj] in [0, 2, 3, 4]):
                
                is_safe = self._is_position_safe(floor, ni, nj)
                
                # 计算移动得分
                current_dist = self.distance_field[floor][i][j]
                new_dist = self.distance_field[floor][ni][nj]
                distance_score = current_dist - new_dist
                
                smoke_penalty = 0
                if smoke_grid and smoke_grid[floor][ni][nj] > 0.3:
                    smoke_penalty = smoke_grid[floor][ni][nj] * 2
                
                time_factor = min(1.0, self.time_passed / 60)
                panic_factor = random.uniform(-self.panic_level * (1 + time_factor), 
                                            self.panic_level * (1 + time_factor))
                
                # 安全优先：如果当前位置不安全，优先移动到安全位置
                safety_bonus = 0
                if not current_safe and is_safe:
                    safety_bonus = 10
                elif current_safe and not is_safe:
                    safety_bonus = -10
                
                total_score = (distance_score * self.intelligence_level - 
                             smoke_penalty + panic_factor + safety_bonus)
                
                possible_moves.append((ni, nj))
                move_scores.append(total_score)
        
        if possible_moves:
            best_move_index = np.argmax(move_scores)
            best_i, best_j = possible_moves[best_move_index]
            
            old_pos = (floor, i, j)
            new_pos = (floor, best_i, best_j)
            
            # 更新人员状态
            if old_pos in self.people_states:
                self.people_states[new_pos] = self.people_states[old_pos]
                del self.people_states[old_pos]
            
            # 移动到出口则自救成功
            if self.building_3d[floor][best_i][best_j] in [2, 3]:
                self.building_3d[floor][i][j] = 0
                self.building_3d[floor][best_i][best_j] = 0
                return "self_rescue"
            else:
                # 确保目标位置是空的
                if self.building_3d[floor][best_i][best_j] == 0:
                    self.building_3d[floor][i][j] = 0
                    self.building_3d[floor][best_i][best_j] = 1
                    return True
        
        return False
    
    def _is_position_safe(self, floor, i, j):
        # 检查当前位置是否有火
        if self.building_3d[floor][i][j] == -1:
            return False
        
        # 检查周围8个方向是否有火
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0:
                    continue
                ni, nj = i + di, j + dj
                if (0 <= ni < self.rows and 0 <= nj < self.cols and
                    self.building_3d[floor][ni][nj] == -1):
                    return False
                
        return True

class SmokeSpreadModel:
    def __init__(self, building_3d):
        self.building_3d = building_3d
        self.floors = len(building_3d)
        self.rows = len(building_3d[0])
        self.cols = len(building_3d[0][0])
        
        self.office_smoke_rate = 0.0067
        self.corridor_smoke_rate = 0.003
        self.smoke_decay = 0.067
        
        self.smoke_grid_3d = []
        for floor in range(self.floors):
            self.smoke_grid_3d.append([[0.0 for _ in range(self.cols)] for _ in range(self.rows)])
        
        self.fire_sources = self._identify_fire_sources()
        
    def _identify_fire_sources(self):
        fire_sources = []
        for floor in range(self.floors):
            for i in range(self.rows):
                for j in range(self.cols):
                    if self.building_3d[floor][i][j] == -1:
                        fire_sources.append((floor, i, j))
        return fire_sources
    
    def update_smoke_spread(self, elapsed_time):
        self._update_office_smoke(elapsed_time)
        self._update_corridor_smoke(elapsed_time)
    
    def _update_office_smoke(self, elapsed_time):
        for floor in range(self.floors):
            for i in range(self.rows):
                for j in range(self.cols):
                    if self.building_3d[floor][i][j] == -1:
                        self.smoke_grid_3d[floor][i][j] = 1.0
                    elif self._is_room_area(floor, i, j):
                        min_fire_dist = self._distance_to_nearest_fire(floor, i, j)
                        smoke_increase = self.office_smoke_rate * elapsed_time
                        distance_factor = max(0, 1 - min_fire_dist / 20)
                        self.smoke_grid_3d[floor][i][j] = min(1.0, 
                            self.smoke_grid_3d[floor][i][j] + smoke_increase * distance_factor)
    
    def _update_corridor_smoke(self, elapsed_time):
        for floor in range(self.floors):
            for i in range(self.rows):
                for j in range(self.cols):
                    if self._is_corridor_area(floor, i, j):
                        corridor_distance = self._distance_along_corridor(floor, i, j)
                        if corridor_distance < float('inf'):
                            smoke_increase = (self.corridor_smoke_rate * corridor_distance * 
                                            math.exp(-self.smoke_decay)) * elapsed_time
                            self.smoke_grid_3d[floor][i][j] = min(1.0, 
                                self.smoke_grid_3d[floor][i][j] + smoke_increase)
    
    def _is_room_area(self, floor, i, j):
        return not (12 <= i <= 14)
    
    def _is_corridor_area(self, floor, i, j):
        return 12 <= i <= 14
    
    def _distance_to_nearest_fire(self, floor, i, j):
        min_dist = float('inf')
        for fire_floor, fire_i, fire_j in self.fire_sources:
            if fire_floor == floor:
                dist = math.sqrt((i - fire_i)**2 + (j - fire_j)**2)
                min_dist = min(min_dist, dist)
        return min_dist if min_dist < float('inf') else 20
    
    def _distance_along_corridor(self, floor, i, j):
        if not self._is_corridor_area(floor, i, j):
            return float('inf')
        
        nearest_fire = None
        min_dist = float('inf')
        
        for fire_floor, fire_i, fire_j in self.fire_sources:
            if fire_floor == floor and self._is_corridor_area(fire_floor, fire_i, fire_j):
                dist = abs(i - fire_i) + abs(j - fire_j)
                if dist < min_dist:
                    min_dist = dist
                    nearest_fire = (fire_i, fire_j)
        
        if nearest_fire is None:
            return float('inf')
        
        visited = set()
        queue = deque([(i, j, 0)])
        visited.add((i, j))
        
        while queue:
            curr_i, curr_j, dist = queue.popleft()
            if (curr_i, curr_j) == nearest_fire:
                return dist
            for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                ni, nj = curr_i + di, curr_j + dj
                if (0 <= ni < self.rows and 0 <= nj < self.cols and 
                    (ni, nj) not in visited and self._is_corridor_area(floor, ni, nj)):
                    visited.add((ni, nj))
                    queue.append((ni, nj, dist + 1))
        
        return float('inf')

class FireSpreadModel:
    def __init__(self, building_3d):
        self.building_3d = building_3d
        self.floors = len(building_3d)
        self.rows = len(building_3d[0])
        self.cols = len(building_3d[0][0])
        self.k = 0.01
        self.fire_areas = []
        
    def update_fire_spread(self, elapsed_time):
        self._identify_fire_areas()
        for fire_area in self.fire_areas:
            self._spread_fire_area(fire_area, elapsed_time)
    
    def _identify_fire_areas(self):
        visited = set()
        self.fire_areas = []
        
        for floor in range(self.floors):
            for i in range(self.rows):
                for j in range(self.cols):
                    if self.building_3d[floor][i][j] == -1 and (floor, i, j) not in visited:
                        fire_area = self._bfs_fire_area(floor, i, j, visited)
                        self.fire_areas.append(fire_area)
    
    def _bfs_fire_area(self, start_floor, start_i, start_j, visited):
        fire_area = []
        queue = deque([(start_floor, start_i, start_j)])
        visited.add((start_floor, start_i, start_j))
        
        while queue:
            floor, i, j = queue.popleft()
            fire_area.append((floor, i, j))
            
            for df in [0]:
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        if di == 0 and dj == 0:
                            continue
                        nf, ni, nj = floor + df, i + di, j + dj
                        if (0 <= nf < self.floors and 
                            0 <= ni < self.rows and 
                            0 <= nj < self.cols and
                            self.building_3d[nf][ni][nj] == -1 and
                            (nf, ni, nj) not in visited):
                            visited.add((nf, ni, nj))
                            queue.append((nf, ni, nj))
        
        return fire_area
    
    def _spread_fire_area(self, fire_area, elapsed_time):
        if not fire_area:
            return
        
        current_area = len(fire_area)
        target_area = int(current_area * math.exp(self.k * elapsed_time))
        
        if target_area > current_area:
            new_fire_cells = target_area - current_area
            spread_candidates = self._get_spread_candidates(fire_area)
            
            if spread_candidates:
                num_to_spread = min(new_fire_cells, len(spread_candidates))
                spread_positions = random.sample(spread_candidates, num_to_spread)
                
                for floor, i, j in spread_positions:
                    if self.building_3d[floor][i][j] == 0:
                        self.building_3d[floor][i][j] = -1
    
    def _get_spread_candidates(self, fire_area):
        candidates = set()
        
        for floor, i, j in fire_area:
            for df in [0]:
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        if di == 0 and dj == 0:
                            continue
                        nf, ni, nj = floor + df, i + di, j + dj
                        if (0 <= nf < self.floors and 
                            0 <= ni < self.rows and 
                            0 <= nj < self.cols and
                            self.building_3d[nf][ni][nj] == 0):
                            candidates.add((nf, ni, nj))
        
        return list(candidates)

class BuildingGenerator:
    def __init__(self):
        self.floors = random.randint(3, 6)
        self.rooms_per_side = random.randint(2, 5)
        self.rows = 27
        self.cols = self.rooms_per_side * 10
        
        self.room_height = 12
        self.room_width = 10
        self.corridor_width = 3
        
        self.total_people = 0
        self.total_fires = 0
    
    def generate_building(self):
        building_3d = []
        room_layouts = []
        door_layouts = []
        
        all_rooms = {}
        all_doors = {}
        
        upper_start_row = 0
        upper_end_row = self.room_height - 1
        
        corridor_start_row = self.room_height
        corridor_end_row = corridor_start_row + self.corridor_width - 1
        
        lower_start_row = corridor_end_row + 1
        lower_end_row = lower_start_row + self.room_height - 1
        
        for room_id in range(1, self.rooms_per_side * 2 + 1):
            if room_id <= self.rooms_per_side:
                start_row = upper_start_row
                end_row = upper_end_row
                start_col = (room_id - 1) * self.room_width
                end_col = start_col + self.room_width - 1
                door_row = end_row
                door_col = start_col + self.room_width // 2
            else:
                start_row = lower_start_row
                end_row = lower_end_row
                lower_room_id = room_id - self.rooms_per_side
                start_col = (lower_room_id - 1) * self.room_width
                end_col = start_col + self.room_width - 1
                door_row = start_row
                door_col = start_col + self.room_width // 2
            
            if room_id == self.rooms_per_side or room_id == self.rooms_per_side * 2:
                end_col = self.cols - 1
            
            all_rooms[room_id] = {
                "start_row": start_row, 
                "end_row": end_row, 
                "start_col": start_col, 
                "end_col": end_col
            }
            all_doors[room_id] = (door_row, door_col)
        
        self.total_people = 0
        self.total_fires = 0
        
        for floor in range(self.floors):
            building = [[0 for _ in range(self.cols)] for _ in range(self.rows)]
            
            for room_id, room_info in all_rooms.items():
                start_row = room_info["start_row"]
                end_row = room_info["end_row"]
                start_col = room_info["start_col"]
                end_col = room_info["end_col"]
                
                room_rows = end_row - start_row + 1
                room_cols = end_col - start_col + 1
                
                if room_rows > 0 and room_cols > 0:
                    self._generate_room_content(building, start_row, end_row, start_col, end_col, floor)
                
                door_row, door_col = all_doors[room_id]
                if 0 <= door_row < self.rows and 0 <= door_col < self.cols:
                    building[door_row][door_col] = 0
            
            for i in range(corridor_start_row, corridor_end_row + 1):
                for j in range(self.cols):
                    building[i][j] = 0
            
            corridor_mid_row = (corridor_start_row + corridor_end_row) // 2
            building[corridor_mid_row][1] = 4
            building[corridor_mid_row][self.cols - 2] = 4
            
            building = self._add_obstacles(building)
            
            floor_people = sum(row.count(1) for row in building)
            floor_fires = sum(row.count(-1) for row in building)
            self.total_people += floor_people
            self.total_fires += floor_fires
            
            building_3d.append(building)
            room_layouts.append(all_rooms.copy())
            door_layouts.append(all_doors.copy())
        
        corridor_mid_row = (corridor_start_row + corridor_end_row) // 2
        if len(building_3d) > 0:
            building_3d[0][corridor_mid_row][0] = 2
            building_3d[0][corridor_mid_row][self.cols - 1] = 3
        
        return building_3d, room_layouts, door_layouts
    
    def _generate_room_content(self, building, start_row, end_row, start_col, end_col, floor):
        rows = end_row - start_row + 1
        cols = end_col - start_col + 1
        
        if rows <= 0 or cols <= 0:
            return
        
        # 调整火生成概率为15%，人员生成概率为10%
        fire_count = max(1, int(rows * cols * 0.15))
        people_count = max(1, int(rows * cols * 0.1))
        
        all_positions = []
        for i in range(rows):
            for j in range(cols):
                actual_row = start_row + i
                actual_col = start_col + j
                
                if 0 <= actual_row < self.rows and 0 <= actual_col < self.cols:
                    is_near_door = False
                    if start_row == 0 and i == rows-1 and j == cols//2:
                        is_near_door = True
                    if start_row == 15 and i == 0 and j == cols//2:
                        is_near_door = True
                    
                    if not is_near_door:
                        all_positions.append((actual_row, actual_col))
        
        if not all_positions:
            return
        
        if all_positions and len(all_positions) >= people_count:
            people_positions = random.sample(all_positions, people_count)
            for row, col in people_positions:
                building[row][col] = 1
                if (row, col) in all_positions:
                    all_positions.remove((row, col))
        
        if all_positions and len(all_positions) >= fire_count:
            fire_positions = random.sample(all_positions, fire_count)
            for row, col in fire_positions:
                building[row][col] = -1
    
    def _add_obstacles(self, building, obstacle_ratio=0.03):
        rows = len(building)
        cols = len(building[0])
        
        obstacle_count = int(rows * cols * obstacle_ratio)
        
        available_positions = []
        for i in range(rows):
            for j in range(cols):
                corridor_mid_row = 13
                if building[i][j] in [0, -1, 1] and (i, j) not in [(corridor_mid_row, 0), (corridor_mid_row, cols-1), (corridor_mid_row, 1), (corridor_mid_row, cols-2)]:
                    available_positions.append((i, j))
        
        if available_positions and obstacle_count > 0:
            obstacle_count = min(obstacle_count, len(available_positions))
            obstacle_positions = random.sample(available_positions, obstacle_count)
            
            for i, j in obstacle_positions:
                building[i][j] = -3
        
        return building

class ImprovedRescuePlanner:
    def __init__(self, building_3d, rooms_3d, doors_3d, initial_people_count):
        self.original_building = copy.deepcopy(building_3d)
        self.building_3d = building_3d
        self.rooms_3d = rooms_3d
        self.doors_3d = doors_3d
        self.finder = PathFinder(building_3d)
        self.fire_model = FireSpreadModel(building_3d)
        self.smoke_model = SmokeSpreadModel(building_3d)
        
        self.floors = len(building_3d)
        self.rows = len(building_3d[0]) if building_3d else 0
        self.cols = len(building_3d[0][0]) if building_3d and building_3d[0] else 0
        
        corridor_mid_row = 13
        
        self.left_rescuer_start = (0, corridor_mid_row, 0)
        
        if len(building_3d) > 0 and len(building_3d[0]) > 0:
            right_start_col = len(building_3d[0][0]) - 1
            self.right_rescuer_start = (0, corridor_mid_row, right_start_col)
        else:
            self.right_rescuer_start = (0, corridor_mid_row, 35)
        
        self.exits = []
        for floor in range(len(building_3d)):
            self.exits.append((floor, corridor_mid_row, 0))
            if len(building_3d[floor]) > 0:
                right_exit_col = len(building_3d[floor][0]) - 1
                self.exits.append((floor, corridor_mid_row, right_exit_col))
        
        self.people_model = ImprovedIntelligentPeopleMovementModel(building_3d, rooms_3d, self.exits)
        
        self.fire_truck_available = True
        self.floor_rescue_status = [False] * len(building_3d)
        
        self.initial_people_count = initial_people_count
        print(f"从生成器获取的初始人员数量: {self.initial_people_count}")
        
        # 添加准确的人员跟踪
        self.initial_people_positions = self._get_all_people_positions()
        print(f"初始人员位置数量: {len(self.initial_people_positions)}")
        
        # 重置统计
        self.people_saved = 0
        self.people_killed = 0
        self.people_self_rescued = 0
        self.rescue_timeline = []
        self.people_movement_count = 0
        
        # 添加救援状态跟踪
        self.rescue_operations = []
        
        # 添加当前人员跟踪
        self.current_people_count = initial_people_count
        
        # 添加救援成功跟踪
        self.successful_rescues = 0
        self.failed_rescues = 0
        
        # 添加时间同步跟踪
        self.current_simulation_time = 0
        self.last_update_time = 0
        self.update_interval = 5  # 每5秒更新一次环境状态
        
        # 灭火统计
        self.fires_extinguished = 0
        
    def _get_all_people_positions(self):
        """获取建筑中所有人员的初始位置"""
        people_positions = []
        for floor in range(len(self.building_3d)):
            for i in range(len(self.building_3d[floor])):
                for j in range(len(self.building_3d[floor][0])):
                    if self.building_3d[floor][i][j] == 1:
                        people_positions.append((floor, i, j))
        return people_positions
    
    def get_room_info(self):
        room_info = {}
        
        for floor in range(len(self.rooms_3d)):
            for room_id, room_data in self.rooms_3d[floor].items():
                start_row = room_data["start_row"]
                end_row = room_data["end_row"]
                start_col = room_data["start_col"]
                end_col = room_data["end_col"]
                
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
    
    def calculate_rescue_time(self, start_pos, room_key, room_info, elapsed_time=0):
        if room_key not in room_info:
            return float('inf')
            
        room_data = room_info[room_key]
        door_pos = room_data["door_position"]
        
        update_interval = 3
        if elapsed_time % update_interval == 0:
            self.fire_model.update_fire_spread(update_interval)
            self.smoke_model.update_smoke_spread(update_interval)
            
            # 更新人员移动并检查自救
            self_rescue_count, people_moved = self.people_model.update_people_movement(elapsed_time, self.smoke_model.smoke_grid_3d)
            if self_rescue_count > 0:
                self.people_self_rescued += self_rescue_count
                self.current_people_count -= self_rescue_count
                for _ in range(self_rescue_count):
                    self.rescue_timeline.append(("SELF_RESCUE", door_pos, elapsed_time))
                    print(f"自救成功: 在时间 {elapsed_time:.2f}s")
            
            if people_moved > 0:
                self.people_movement_count += people_moved
            
            self._check_people_death(elapsed_time)
        
        path_to_door, time_to_door = self.finder.find_path(
            start_pos, door_pos, has_people=False, 
            smoke_grid=self.smoke_model.smoke_grid_3d)
        
        room_time = self._calculate_room_rescue_time(room_key, room_info, elapsed_time + time_to_door)
        
        exit_times = []
        for exit_pos in self.exits:
            _, time_to_exit = self.finder.find_path(
                door_pos, exit_pos, has_people=True,
                smoke_grid=self.smoke_model.smoke_grid_3d)
            exit_times.append(time_to_exit)
        
        min_exit_time = min(exit_times) if exit_times else float('inf')
        
        total_time = time_to_door + room_time + min_exit_time
        return total_time
    
    def _check_people_death(self, current_time):
        """改进的死亡检测逻辑 - 使用累计20次或连续10次"""
        people_to_remove = []
        
        for floor in range(len(self.building_3d)):
            for i in range(len(self.building_3d[floor])):
                for j in range(len(self.building_3d[floor][0])):
                    if self.building_3d[floor][i][j] == 1:
                        pos_key = (floor, i, j)
                        
                        # 检查是否在火中或紧邻火源
                        is_exposed = False
                        
                        # 当前位置有火
                        if self.building_3d[floor][i][j] == -1:
                            is_exposed = True
                        
                        # 更新人员状态
                        if pos_key not in self.people_model.people_states:
                            self.people_model.people_states[pos_key] = {
                                'total_fire_exposure': 0,
                                'consecutive_fire_exposure': 0,
                                'last_update_time': current_time
                            }
                        
                        if is_exposed:
                            self.people_model.people_states[pos_key]['total_fire_exposure'] += 1
                            self.people_model.people_states[pos_key]['consecutive_fire_exposure'] += 1
                            self.people_model.people_states[pos_key]['last_update_time'] = current_time
                        else:
                            self.people_model.people_states[pos_key]['consecutive_fire_exposure'] = 0
                        
                        state = self.people_model.people_states[pos_key]
                        
                        # 死亡条件：累计暴露20次或连续暴露10次
                        if (state['total_fire_exposure'] >= 20 or 
                            state['consecutive_fire_exposure'] >= 10):
                            
                            people_to_remove.append((floor, i, j, state))
        
        # 移除死亡人员
        for floor, i, j, state in people_to_remove:
            # 确保该位置确实有人
            if self.building_3d[floor][i][j] == 1:
                self.building_3d[floor][i][j] = 0
                self.people_killed += 1
                self.current_people_count -= 1
                
                # 记录死亡事件
                death_reason = "累计暴露" if state['total_fire_exposure'] >= 20 else "连续暴露"
                exposure_count = state['total_fire_exposure'] if death_reason == "累计暴露" else state['consecutive_fire_exposure']
                
                self.rescue_timeline.append((
                    "DEATH", 
                    (floor, i, j), 
                    current_time,
                    f"{death_reason}({exposure_count}次)"
                ))
                
                print(f"人员死亡: 位置({floor},{i},{j}) - {death_reason} {exposure_count}次")
                
                # 从状态字典中移除
                pos_key = (floor, i, j)
                if pos_key in self.people_model.people_states:
                    del self.people_model.people_states[pos_key]
    
    def _calculate_room_rescue_time(self, room_key, room_info, current_time):
        if room_key not in room_info:
            return 0
            
        room_data = room_info[room_key]
        floor = room_data["floor"]
        room_id = room_data["room_id"]
        
        if floor >= len(self.rooms_3d) or room_id not in self.rooms_3d[floor]:
            return 0
            
        room_layout = self.rooms_3d[floor][room_id]
        people_positions = []
        
        if (room_layout["start_row"] < 0 or room_layout["end_row"] >= len(self.building_3d[floor]) or
            room_layout["start_col"] < 0 or room_layout["end_col"] >= len(self.building_3d[floor][0])):
            return 0
            
        current_fire_count = 0
        for i in range(room_layout["start_row"], room_layout["end_row"] + 1):
            for j in range(room_layout["start_col"], room_layout["end_col"] + 1):
                if self.building_3d[floor][i][j] == 1:
                    people_positions.append((floor, i, j))
                elif self.building_3d[floor][i][j] == -1:
                    current_fire_count += 1
        
        base_time = len(people_positions) * 2 * self.finder.movement_speed
        
        smoke_penalty = 0
        for floor, i, j in people_positions:
            if (floor < len(self.smoke_model.smoke_grid_3d) and 
                i < len(self.smoke_model.smoke_grid_3d[floor]) and 
                j < len(self.smoke_model.smoke_grid_3d[floor][0])):
                smoke_level = self.smoke_model.smoke_grid_3d[floor][i][j]
                if smoke_level > 0.1:
                    smoke_penalty += smoke_level * 2
        
        return base_time + smoke_penalty
    
    def execute_rescue_operation(self, room_key, room_info, rescue_time):
        """执行救援操作并正确统计人数"""
        if room_key not in room_info:
            return 0
            
        room_data = room_info[room_key]
        floor = room_data["floor"]
        room_id = room_data["room_id"]
        
        if floor >= len(self.rooms_3d) or room_id not in self.rooms_3d[floor]:
            return 0
            
        room_layout = self.rooms_3d[floor][room_id]
        rescued_count = 0
        
        if (room_layout["start_row"] < 0 or room_layout["end_row"] >= len(self.building_3d[floor]) or
            room_layout["start_col"] < 0 or room_layout["end_col"] >= len(self.building_3d[floor][0])):
            return 0
            
        # 救出房间内所有人员
        for i in range(room_layout["start_row"], room_layout["end_row"] + 1):
            for j in range(room_layout["start_col"], room_layout["end_col"] + 1):
                if self.building_3d[floor][i][j] == 1:
                    self.building_3d[floor][i][j] = 0
                    rescued_count += 1
                    self.people_saved += 1
                    self.current_people_count -= 1
                    self.rescue_timeline.append(("RESCUE", (floor, i, j), rescue_time))
                    print(f"救援执行: 在时间 {rescue_time:.2f}s 救出人员 ({floor}, {i}, {j}) - 房间 {room_key}")
        
        # 记录救援操作
        self.rescue_operations.append({
            'room': room_key,
            'time': rescue_time,
            'rescued': rescued_count,
            'position': room_data['door_position']
        })
        
        if rescued_count > 0:
            self.successful_rescues += 1
        else:
            self.failed_rescues += 1
        
        return rescued_count
    
    def execute_complete_rescue_plan(self, left_rooms, right_rooms, room_info):
        """执行完整的救援计划，包含时间同步"""
        print(f"开始执行完整救援计划: 左侧{len(left_rooms)}个房间, 右侧{len(right_rooms)}个房间")
        
        # 重置模拟时间
        self.current_simulation_time = 0
        self.last_update_time = 0
        
        # 执行左侧救援路线
        left_rescuer_time = self._execute_rescue_route(
            left_rooms, room_info, self.left_rescuer_start, "左侧")
        
        # 执行右侧救援路线  
        right_rescuer_time = self._execute_rescue_route(
            right_rooms, room_info, self.right_rescuer_start, "右侧")
        
        total_time = max(left_rescuer_time, right_rescuer_time)
        
        # 救援结束后继续模拟一段时间，确保所有事件完成
        final_time = self._simulate_after_rescue(total_time)
        
        return final_time
    
    def _execute_rescue_route(self, room_sequence, room_info, start_pos, route_name):
        """执行单个救援路线"""
        current_time = 0
        current_pos = start_pos
        
        print(f"{route_name}救援开始于位置 {start_pos}")
        
        for room_key in room_sequence:
            if room_key not in room_info:
                print(f"警告: 房间 {room_key} 不存在于房间信息中")
                continue
                
            # 更新环境状态到当前时间
            self._update_environment_to_time(current_time)
            
            # 计算到房间的路径和时间
            path, travel_time = self.finder.find_path(
                current_pos, room_info[room_key]["door_position"], 
                has_people=False, smoke_grid=self.smoke_model.smoke_grid_3d
            )
            
            if travel_time == float('inf'):
                print(f"警告: {route_name}无法到达房间 {room_key}，跳过")
                continue
                
            # 移动救援人员并灭火
            current_time += travel_time
            current_pos = room_info[room_key]["door_position"]
            
            # 更新环境状态
            self._update_environment_to_time(current_time)
            
            # 执行救援
            rescued_count = self.execute_rescue_operation(room_key, room_info, current_time)
            print(f"{route_name}救援: 在时间 {current_time:.2f}s 救出房间 {room_key} 的 {rescued_count} 人")
            
            # 短暂停留处理救援
            rescue_processing_time = max(1, rescued_count * 0.5)
            current_time += rescue_processing_time
            
        return current_time
    
    def _update_environment_to_time(self, target_time):
        """更新环境状态到指定时间"""
        while self.current_simulation_time < target_time:
            # 计算时间增量
            time_step = min(self.update_interval, target_time - self.current_simulation_time)
            
            if time_step <= 0:
                break
                
            # 更新火灾蔓延
            self.fire_model.update_fire_spread(time_step)
            
            # 更新烟雾扩散
            self.smoke_model.update_smoke_spread(time_step)
            
            # 更新人员移动和自救
            self_rescue_count, people_moved = self.people_model.update_people_movement(
                self.current_simulation_time, self.smoke_model.smoke_grid_3d
            )
            
            # 记录自救事件
            if self_rescue_count > 0:
                self.people_self_rescued += self_rescue_count
                self.current_people_count -= self_rescue_count
                for _ in range(self_rescue_count):
                    self.rescue_timeline.append(("SELF_RESCUE", "出口", self.current_simulation_time))
                print(f"自救成功: 在时间 {self.current_simulation_time:.2f}s, {self_rescue_count} 人自救")
            
            # 更新人员移动计数
            if people_moved > 0:
                self.people_movement_count += people_moved
            
            # 检测人员死亡
            self._check_people_death(self.current_simulation_time)
            
            # 更新时间
            self.current_simulation_time += time_step
    
    def _simulate_after_rescue(self, rescue_end_time):
        """救援结束后继续模拟，确保所有事件完成"""
        final_time = rescue_end_time
        max_additional_time = 300  # 最多额外模拟5分钟
        
        print("救援结束，开始后续模拟...")
        
        for additional_time in range(0, max_additional_time, self.update_interval):
            current_time = rescue_end_time + additional_time
            
            # 更新环境
            self.fire_model.update_fire_spread(self.update_interval)
            self.smoke_model.update_smoke_spread(self.update_interval)
            
            # 更新人员移动
            self_rescue_count, people_moved = self.people_model.update_people_movement(
                current_time, self.smoke_model.smoke_grid_3d
            )
            
            # 记录自救
            if self_rescue_count > 0:
                self.people_self_rescued += self_rescue_count
                self.current_people_count -= self_rescue_count
                for _ in range(self_rescue_count):
                    self.rescue_timeline.append(("SELF_RESCUE", "出口", current_time))
                print(f"后续自救: 在时间 {current_time:.2f}s, {self_rescue_count} 人自救")
            
            # 更新移动计数
            if people_moved > 0:
                self.people_movement_count += people_moved
            
            # 检测死亡
            self._check_people_death(current_time)
            
            final_time = current_time
            
            # 如果没有人员剩余，提前结束
            if self.current_people_count <= 0:
                print("所有人员已处理完毕，结束模拟")
                break
        
        return final_time
    
    def use_fire_truck(self, floor):
        if not self.fire_truck_available:
            return 0
            
        fire_count = 0
        for i in range(len(self.building_3d[floor])):
            for j in range(len(self.building_3d[floor][0])):
                if self.building_3d[floor][i][j] == -1:
                    fire_count += 1
                    self.building_3d[floor][i][j] = 0
        
        fire_truck_time = fire_count * 0.2
        self.floor_rescue_status[floor] = True
        
        return fire_truck_time
    
    def calculate_rescue_score(self):
        total_rescued = self.people_saved + self.people_self_rescued
        total_lost = self.people_killed + self.current_people_count
        
        # 验证人员平衡
        if total_rescued + total_lost != self.initial_people_count:
            print(f"警告: 人员统计不平衡! 初始: {self.initial_people_count}, 统计: {total_rescued + total_lost}")
            # 强制平衡
            self.current_people_count = max(0, self.initial_people_count - total_rescued - self.people_killed)
            total_lost = self.people_killed + self.current_people_count
        
        if self.initial_people_count == 0:
            rescue_rate = 1.0
        else:
            rescue_rate = total_rescued / self.initial_people_count
        
        base_score = 100
        score = base_score * rescue_rate
        
        if self.people_killed > 0:
            death_penalty = (self.people_killed / self.initial_people_count) * 50
            score -= death_penalty
        
        total_rescue_time = self._get_total_rescue_time()
        if total_rescue_time > 300:
            time_penalty = (total_rescue_time - 300) / 10
            score -= time_penalty
        
        if self.people_self_rescued > 0:
            self_rescue_bonus = (self.people_self_rescued / self.initial_people_count) * 10
            score += self_rescue_bonus
        
        # 灭火奖励
        if self.fires_extinguished > 0:
            fire_bonus = min(10, self.fires_extinguished * 0.5)
            score += fire_bonus
        
        score = max(0, score)
        
        return {
            "score": score,
            "rescue_rate": rescue_rate,
            "initial_people": self.initial_people_count,
            "people_saved": self.people_saved,
            "people_self_rescued": self.people_self_rescued,
            "people_killed": self.people_killed,
            "total_rescue_time": total_rescue_time,
            "people_movement_count": self.people_movement_count,
            "rescue_operations": len(self.rescue_operations),
            "current_people": self.current_people_count,
            "successful_rescues": self.successful_rescues,
            "failed_rescues": self.failed_rescues,
            "total_processed": total_rescued + total_lost,
            "fires_extinguished": self.fires_extinguished
        }
    
    def _get_total_rescue_time(self):
        if not self.rescue_timeline:
            return 0
        
        last_time = 0
        for event in self.rescue_timeline:
            if len(event) >= 3 and event[2] > last_time:
                last_time = event[2]
        
        return last_time

class MultiAStarRescuePlanner:
    def __init__(self, building_3d, rooms_3d, doors_3d, initial_people_count):
        self.building_3d = building_3d
        self.rooms_3d = rooms_3d
        self.doors_3d = doors_3d
        self.initial_people_count = initial_people_count
        
    def solve(self):
        # 创建救援计划器
        rescue_planner = ImprovedRescuePlanner(
            self.building_3d, self.rooms_3d, self.doors_3d, self.initial_people_count
        )
        room_info = rescue_planner.get_room_info()
        
        room_keys = list(room_info.keys())
        
        if not room_keys:
            return [], [], room_info
        
        room_costs = {}
        for room_key in room_keys:
            left_cost = rescue_planner.calculate_rescue_time(
                rescue_planner.left_rescuer_start, room_key, room_info)
            right_cost = rescue_planner.calculate_rescue_time(
                rescue_planner.right_rescuer_start, room_key, room_info)
            room_costs[room_key] = {
                'left': left_cost,
                'right': right_cost,
                'importance': room_info[room_key]['importance'],
                'cost_diff': abs(left_cost - right_cost)
            }
        
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
            
            if left_total_time + left_time <= right_total_time + right_time:
                left_rooms.append(room_key)
                left_total_time += left_time
            else:
                right_rooms.append(room_key)
                right_total_time += right_time
        
        left_rooms = self._optimize_sequence(left_rooms, rescue_planner.left_rescuer_start, room_info)
        right_rooms = self._optimize_sequence(right_rooms, rescue_planner.right_rescuer_start, room_info)
        
        return left_rooms, right_rooms, room_info
    
    def _optimize_sequence(self, rooms, start_pos, room_info):
        if len(rooms) <= 1:
            return rooms
        
        current_pos = start_pos
        unvisited = set(rooms)
        optimized_sequence = []
        
        while unvisited:
            nearest_room = min(unvisited, 
                             key=lambda x: self._estimate_distance(current_pos, x, room_info))
            optimized_sequence.append(nearest_room)
            unvisited.remove(nearest_room)
            current_pos = room_info[nearest_room]['door_position']
        
        return optimized_sequence
    
    def _estimate_distance(self, pos1, room_key, room_info):
        pos2 = room_info[room_key]['door_position']
        floor_dist = abs(pos1[0] - pos2[0]) * 20
        manhattan_dist = abs(pos1[1] - pos2[1]) + abs(pos1[2] - pos2[2])
        return floor_dist + manhattan_dist

class AlgorithmComparator:
    def __init__(self, building_3d, rooms_3d, doors_3d, initial_people_count):
        self.building_3d = building_3d
        self.rooms_3d = rooms_3d
        self.doors_3d = doors_3d
        self.initial_people_count = initial_people_count
        
    def compare_algorithms(self):
        algorithms = {
            "MultiAStar": MultiAStarRescuePlanner,
        }
        
        results = {}
        
        for name, planner_class in algorithms.items():
            try:
                print(f"正在测试 {name} 算法...")
                planner = planner_class(self.building_3d, self.rooms_3d, self.doors_3d, self.initial_people_count)
                
                start_time = time.time()
                left_order, right_order, room_info = planner.solve()
                computation_time = time.time() - start_time
                
                rescue_time, rescue_score, rescue_executor = self._calculate_rescue_time_and_score(left_order, right_order, room_info)
                
                results[name] = {
                    'computation_time': computation_time,
                    'rescue_time': rescue_time,
                    'rescue_score': rescue_score,
                    'solution_quality': len(left_order) + len(right_order),
                    'left_rooms': left_order,
                    'right_rooms': right_order,
                    'room_info': room_info,
                    'rescue_executor': rescue_executor
                }
                
                print(f"  {name}: 计算时间={computation_time:.4f}s, 救援时间={rescue_time:.2f}s, 救援评分={rescue_score['score']:.2f}")
                
            except Exception as e:
                print(f"  {name} 算法出错: {e}")
                import traceback
                traceback.print_exc()
                results[name] = None
        
        return results
    
    def _calculate_rescue_time_and_score(self, left_rooms, right_rooms, room_info):
        # 创建救援计划执行器 - 使用原始建筑的深拷贝
        rescue_executor = ImprovedRescuePlanner(
            copy.deepcopy(self.building_3d), 
            self.rooms_3d, 
            self.doors_3d, 
            self.initial_people_count
        )
        
        # 使用新的执行方法
        total_time = rescue_executor.execute_complete_rescue_plan(left_rooms, right_rooms, room_info)
        
        rescue_score = rescue_executor.calculate_rescue_score()
        
        # 打印详细统计
        print(f"\n=== 最终救援统计 ===")
        print(f"初始人员: {rescue_score['initial_people']}")
        print(f"救出人员: {rescue_score['people_saved']}")
        print(f"自救人员: {rescue_score['people_self_rescued']}") 
        print(f"死亡人员: {rescue_score['people_killed']}")
        print(f"当前剩余: {rescue_score['current_people']}")
        print(f"灭火数量: {rescue_score['fires_extinguished']}")
        
        total_processed = rescue_score['total_processed']
        
        print(f"人员平衡: {rescue_score['initial_people']} = {total_processed} (救出{rescue_score['people_saved']} + 自救{rescue_score['people_self_rescued']} + 死亡{rescue_score['people_killed']} + 剩余{rescue_score['current_people']})")
        
        # 如果还有剩余人员，检查他们的状态
        if rescue_score['current_people'] > 0:
            print(f"\n=== 剩余人员分析 ===")
            remaining_people = rescue_executor._get_all_people_positions()
            print(f"剩余人员位置: {len(remaining_people)}个位置")
            
            for pos in remaining_people:
                floor, i, j = pos
                # 检查是否在火中
                if rescue_executor.building_3d[floor][i][j] == -1:
                    print(f"  位置 {pos}: 在火中")
                else:
                    # 检查周围是否有火
                    has_fire_nearby = False
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            ni, nj = i + di, j + dj
                            if (0 <= ni < len(rescue_executor.building_3d[floor]) and 
                                0 <= nj < len(rescue_executor.building_3d[floor][0]) and
                                rescue_executor.building_3d[floor][ni][nj] == -1):
                                has_fire_nearby = True
                                break
                        if has_fire_nearby:
                            break
                    
                    if has_fire_nearby:
                        print(f"  位置 {pos}: 火源附近")
                    else:
                        print(f"  位置 {pos}: 安全区域")
        
        return total_time, rescue_score, rescue_executor
    
    def select_best_algorithm(self, results):
        valid_results = {k: v for k, v in results.items() if v is not None}
        
        if not valid_results:
            return None, None
        
        best_algorithm = max(valid_results.keys(), 
                           key=lambda x: valid_results[x]['rescue_score']['score'])
        
        return best_algorithm, valid_results[best_algorithm]

class FireExtinguisher3D:
    def __init__(self, building_3d, rooms_3d, doors_3d):
        self.building_3d = building_3d
        self.rooms_3d = rooms_3d
        self.doors_3d = doors_3d
        self.finder = PathFinder(building_3d)
        
        corridor_mid_row = 13
        
        self.left_rescuer_start = (0, corridor_mid_row, 0)
        if len(building_3d) > 0 and len(building_3d[0]) > 0:
            right_start_col = len(building_3d[0][0]) - 1
            self.right_rescuer_start = (0, corridor_mid_row, right_start_col)
        else:
            self.right_rescuer_start = (0, corridor_mid_row, 35)
        
        self.exits = []
        for floor in range(len(building_3d)):
            self.exits.append((floor, corridor_mid_row, 0))
            if len(building_3d[floor]) > 0:
                right_exit_col = len(building_3d[floor][0]) - 1
                self.exits.append((floor, corridor_mid_row, right_exit_col))
    
    def get_fire_locations(self):
        fire_locations = []
        for floor in range(len(self.building_3d)):
            for i in range(len(self.building_3d[floor])):
                for j in range(len(self.building_3d[floor][0])):
                    if self.building_3d[floor][i][j] == -1:
                        fire_locations.append((floor, i, j))
        return fire_locations
    
    def optimize_fire_extinguish_plan(self):
        fire_locations = self.get_fire_locations()
        
        if not fire_locations:
            return [], [], 0
        
        left_fires, right_fires = self._assign_fires(fire_locations)
        
        left_plan, left_time = self._optimize_fire_sequence_3d(left_fires, self.left_rescuer_start)
        right_plan, right_time = self._optimize_fire_sequence_3d(right_fires, self.right_rescuer_start)
        
        return left_plan, right_plan, max(left_time, right_time)
    
    def _assign_fires(self, fire_locations):
        if not fire_locations:
            return [], []
        
        left_fires = []
        right_fires = []
        
        for fire in fire_locations:
            left_dist = self._calculate_3d_distance(fire, self.left_rescuer_start)
            right_dist = self._calculate_3d_distance(fire, self.right_rescuer_start)
            
            if left_dist <= right_dist:
                left_fires.append(fire)
            else:
                right_fires.append(fire)
        
        return left_fires, right_fires
    
    def _calculate_3d_distance(self, pos1, pos2):
        floor_diff = abs(pos1[0] - pos2[0])
        row_diff = abs(pos1[1] - pos2[1])
        col_diff = abs(pos1[2] - pos2[2])
        return floor_diff * 10 + row_diff + col_diff
    
    def _optimize_fire_sequence_3d(self, fires, start_pos):
        if not fires:
            return [], 0
        
        unvisited = fires.copy()
        current_pos = start_pos
        path = [current_pos]
        total_time = 0
        
        while unvisited:
            nearest_fire = None
            min_heuristic_distance = float('inf')
            
            for fire in unvisited:
                heuristic_distance = self._calculate_3d_distance(current_pos, fire)
                if heuristic_distance < min_heuristic_distance:
                    min_heuristic_distance = heuristic_distance
                    nearest_fire = fire
            
            path_to_fire, time_to_fire = self.finder.find_path(current_pos, nearest_fire, has_people=False)
            
            total_time += time_to_fire
            
            current_pos = nearest_fire
            path.extend(path_to_fire[1:])
            path.append(("EXTINGUISH", current_pos))
            
            unvisited.remove(nearest_fire)
        
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

def print_rescue_timeline(rescue_planner):
    print("\n=== 救援时间线 ===")
    if not rescue_planner.rescue_timeline:
        print("无救援事件")
        return
    
    sorted_timeline = sorted(rescue_planner.rescue_timeline, key=lambda x: x[2])
    
    rescue_count = 0
    death_count = 0
    self_rescue_count = 0
    
    for event in sorted_timeline:
        event_type = event[0]
        position = event[1]
        time_val = event[2]
        
        if event_type == "RESCUE":
            rescue_count += 1
            if len(position) >= 3:
                floor, i, j = position
                print(f"时间 {time_val:.2f}s: 救出人员 - 楼层{floor+1}位置({i},{j})")
            else:
                print(f"时间 {time_val:.2f}s: 救出人员 - 位置{position}")
        elif event_type == "DEATH":
            death_count += 1
            if len(position) >= 3:
                floor, i, j = position
                reason = event[3] if len(event) > 3 else "未知原因"
                print(f"时间 {time_val:.2f}s: 人员死亡 - 楼层{floor+1}位置({i},{j}) - {reason}")
            else:
                print(f"时间 {time_val:.2f}s: 人员死亡 - 位置{position}")
        elif event_type == "SELF_RESCUE":
            self_rescue_count += 1
            print(f"时间 {time_val:.2f}s: 人员自救 - 位置{position}")
    
    print(f"\n总计: 救出 {rescue_count} 人, 自救 {self_rescue_count} 人, 死亡 {death_count} 人")
    print(f"人员移动次数: {rescue_planner.people_movement_count}")
    print(f"救援操作次数: {len(rescue_planner.rescue_operations)}")
    print(f"成功救援: {rescue_planner.successful_rescues}")
    print(f"失败救援: {rescue_planner.failed_rescues}")
    print(f"灭火数量: {rescue_planner.fires_extinguished}")

def print_rescue_score(rescue_score):
    print("\n=== 救援评分 ===")
    print(f"初始人员数量: {rescue_score['initial_people']}")
    print(f"成功救出人员: {rescue_score['people_saved']}")
    print(f"自救成功人员: {rescue_score['people_self_rescued']}")
    print(f"死亡人员: {rescue_score['people_killed']}")
    print(f"当前剩余人员: {rescue_score['current_people']}")
    print(f"灭火数量: {rescue_score['fires_extinguished']}")
    print(f"救援成功率: {rescue_score['rescue_rate']:.2%}")
    print(f"总救援时间: {rescue_score['total_rescue_time']:.2f}秒")
    print(f"人员移动次数: {rescue_score['people_movement_count']}")
    print(f"救援操作次数: {rescue_score['rescue_operations']}")
    print(f"成功救援: {rescue_score['successful_rescues']}")
    print(f"失败救援: {rescue_score['failed_rescues']}")
    print(f"最终评分: {rescue_score['score']:.2f}/100")
    
    # 计算人员平衡
    total_rescued = rescue_score['people_saved'] + rescue_score['people_self_rescued']
    total_lost = rescue_score['people_killed'] + rescue_score['current_people']
    balance_check = total_rescued + total_lost
    
    print(f"\n人员平衡检查:")
    print(f"  初始人员: {rescue_score['initial_people']}")
    print(f"  总计救出: {total_rescued}")
    print(f"  总计损失: {total_lost}")
    print(f"  平衡结果: {rescue_score['initial_people']} = {total_rescued} + {total_lost} = {balance_check}")
    
    if rescue_score['score'] >= 90:
        evaluation = "优秀"
    elif rescue_score['score'] >= 80:
        evaluation = "良好"
    elif rescue_score['score'] >= 70:
        evaluation = "中等"
    elif rescue_score['score'] >= 60:
        evaluation = "及格"
    else:
        evaluation = "不及格"
    
    print(f"评价: {evaluation}")

def main():
    print("生成多层建筑布局...")
    generator = BuildingGenerator()
    building_3d, rooms_3d, doors_3d = generator.generate_building()
    
    print(f"建筑信息: {generator.floors}层, 每侧{generator.rooms_per_side}个房间")
    print(f"建筑尺寸: {generator.rows}行 x {generator.cols}列")
    print(f"房间尺寸: {generator.room_height}行 x {generator.room_width}列")
    print(f"走廊宽度: {generator.corridor_width}行")
    
    print(f"直接从生成器获取的总人数: {generator.total_people}")
    print(f"直接从生成器获取的总火源数: {generator.total_fires}")
    
    for floor in range(len(building_3d)):
        print_building_floor(building_3d, floor)
    
    print("\n正在比较算法性能...")
    comparator = AlgorithmComparator(building_3d, rooms_3d, doors_3d, generator.total_people)
    results = comparator.compare_algorithms()
    
    best_algorithm, best_result = comparator.select_best_algorithm(results)
    
    if best_algorithm is None:
        print("错误: 没有可用的算法")
        return
    
    print(f"\n最优算法: {best_algorithm}")
    print(f"计算时间: {best_result['computation_time']:.4f}秒")
    print(f"预计救援时间: {best_result['rescue_time']:.2f}秒")
    
    left_order = best_result['left_rooms']
    right_order = best_result['right_rooms']
    room_info = best_result['room_info']
    rescue_executor = best_result['rescue_executor']
    
    print_room_info(room_info, building_3d)
    print_rescue_plan(left_order, right_order, room_info, best_algorithm)
    
    print("\n=== 烟雾扩散情况 ===")
    rescue_planner = ImprovedRescuePlanner(building_3d, rooms_3d, doors_3d, generator.total_people)
    rescue_planner.smoke_model.update_smoke_spread(60)
    
    print("\n\n规划灭火任务...")
    fire_planner = FireExtinguisher3D(building_3d, rooms_3d, doors_3d)
    left_fire_plan, right_fire_plan, fire_time = fire_planner.optimize_fire_extinguish_plan()
    
    total_time = best_result['rescue_time'] + fire_time
    print(f"\n=== 总任务时间 ===")
    print(f"救援时间: {best_result['rescue_time']:.2f} 秒")
    print(f"灭火时间: {fire_time:.2f} 秒")
    print(f"总时间: {total_time:.2f} 秒")
    
    print_rescue_timeline(rescue_executor)
    print_rescue_score(best_result['rescue_score'])
    
    print("\n=== 算法比较结果 ===")
    print("算法名称 | 计算时间(秒) | 救援时间(秒) | 救援评分 | 解决方案质量")
    print("-" * 80)
    for name, result in results.items():
        if result:
            print(f"{name:8} | {result['computation_time']:11.4f} | {result['rescue_time']:12.2f} | {result['rescue_score']['score']:8.2f} | {result['solution_quality']:14}")

if __name__ == "__main__":
    main()