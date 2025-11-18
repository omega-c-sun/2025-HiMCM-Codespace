import heapq
import math
import random
import numpy as np
from collections import defaultdict, deque
import time
import copy
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap

# 设置matplotlib使用支持中文的字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

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
        
        # 不同年龄段的速度
        self.age_speeds = {
            'young': 6.0,    # 年轻人正常速度
            'adult': 6.0,    # 成年人正常速度  
            'elder': 4.0     # 老年人较慢速度
        }
        self.age_smoke_speeds = {
            'young': 3.0,    # 年轻人在烟雾中速度
            'adult': 3.0,    # 成年人在烟雾中速度
            'elder': 2.0     # 老年人在烟雾中更慢速度
        }
        
    def get_speed_for_person(self, person_type, in_smoke=False):
        """根据人员类型获取移动速度"""
        if in_smoke:
            return self.age_smoke_speeds.get(person_type, self.smoke_speed)
        else:
            return self.age_speeds.get(person_type, self.normal_speed)
        
    def find_path(self, start, end, has_people=False, smoke_grid=None, person_type='adult'):
        def get_smoke_penalty(smoke_level):
            if smoke_level < 0.1: return 1.0
            elif smoke_level < 0.3: return 1.5
            elif smoke_level < 0.6: return 2.0
            else: return 3.0
            
        def heuristic(a, b):
            floor_time = abs(a[0]-b[0]) * self.stair_time
            horizontal_dist = abs(a[1]-b[1]) + abs(a[2]-b[2])
            
            # 根据人员类型调整速度
            current_speed = self.get_speed_for_person(person_type, False)
            horizontal_time = horizontal_dist * (1/current_speed)
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
                
                # 计算移动代价 - 根据人员类型和烟雾情况
                if smoke_grid and smoke_grid[nf][nx][ny] > 0.1:
                    smoke_level = smoke_grid[nf][nx][ny]
                    penalty = get_smoke_penalty(smoke_level)
                    current_speed = self.get_speed_for_person(person_type, True)
                    move_cost = (1/current_speed) * penalty
                else:
                    current_speed = self.get_speed_for_person(person_type, False)
                    move_cost = 1/current_speed
                
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
        
        # 楼梯可以上下楼
        if self.grid_3d[floor][x][y] == 4:
            if floor > 0:
                neighbors.append((floor-1, x, y))
            if floor < self.floors - 1:
                neighbors.append((floor+1, x, y))
        
        return neighbors

class SprinklerSystem:
    def __init__(self, building_3d, rooms_3d):
        self.building_3d = building_3d
        self.rooms_3d = rooms_3d
        self.floors = len(building_3d)
        self.rows = len(building_3d[0])
        self.cols = len(building_3d[0][0])
        
        # 洒水器属性
        self.efficiency = 0.6  # 灭火效率
        self.radius = 10        # 影响半径
        self.duration = 30     # 持续时间（秒）
        self.trigger_threshold = 0.3  # 烟雾触发阈值
        
        # 初始化洒水器位置（每个房间中心）
        self.sprinkler_positions = self._initialize_sprinklers()
        self.active_sprinklers = {}  # 激活的洒水器 {位置: 剩余时间}
        self.used_sprinklers = set()  # 已使用的洒水器（一次性）
        
    def _initialize_sprinklers(self):
        """在每个房间中心放置洒水器"""
        sprinklers = []
        for floor in range(self.floors):
            for room_id, room_info in self.rooms_3d[floor].items():
                center_row = (room_info["start_row"] + room_info["end_row"]) // 2
                center_col = (room_info["start_col"] + room_info["end_col"]) // 2
                sprinklers.append((floor, center_row, center_col))
        return sprinklers
    
    def update(self, smoke_grid, elapsed_time):
        """更新洒水器状态"""
        # 检查是否需要触发洒水器（只检查未使用过的）
        available_sprinklers = [pos for pos in self.sprinkler_positions 
                              if pos not in self.used_sprinklers]
        
        for sprinkler_pos in available_sprinklers:
            floor, i, j = sprinkler_pos
            
            # 检查烟雾浓度是否达到触发阈值
            if (floor < len(smoke_grid) and i < len(smoke_grid[floor]) and 
                j < len(smoke_grid[floor][0]) and 
                smoke_grid[floor][i][j] >= self.trigger_threshold):
                
                print(f"Sprinkler activated at {sprinkler_pos}, smoke level: {smoke_grid[floor][i][j]:.2f}")
                self.active_sprinklers[sprinkler_pos] = self.duration
                self.used_sprinklers.add(sprinkler_pos)  # 标记为已使用
        
        # 更新激活的洒水器
        sprinklers_to_remove = []
        for sprinkler_pos, remaining_time in self.active_sprinklers.items():
            self.active_sprinklers[sprinkler_pos] -= 1
            
            # 洒水器工作期间灭火
            fires_extinguished = self._extinguish_fire_around(sprinkler_pos)
            
            if self.active_sprinklers[sprinkler_pos] <= 0:
                sprinklers_to_remove.append(sprinkler_pos)
                print(f"Sprinkler at {sprinkler_pos} finished working, extinguished {fires_extinguished} fires in total")
        
        for sprinkler_pos in sprinklers_to_remove:
            del self.active_sprinklers[sprinkler_pos]
    
    def _extinguish_fire_around(self, sprinkler_pos):
        """在洒水器周围灭火"""
        floor, center_i, center_j = sprinkler_pos
        
        fires_extinguished = 0
        for i in range(max(0, center_i - self.radius), min(self.rows, center_i + self.radius + 1)):
            for j in range(max(0, center_j - self.radius), min(self.cols, center_j + self.radius + 1)):
                # 计算距离
                distance = math.sqrt((i - center_i)**2 + (j - center_j)**2)
                if distance <= self.radius and self.building_3d[floor][i][j] == -1:
                    # 根据距离和效率计算灭火概率
                    extinguish_prob = self.efficiency * (1 - distance / self.radius)
                    if random.random() < extinguish_prob:
                        self.building_3d[floor][i][j] = 0
                        fires_extinguished += 1
        
        return fires_extinguished

    def get_sprinkler_status(self):
        """获取洒水器状态用于显示"""
        status = {
            'active': list(self.active_sprinklers.keys()),
            'used': list(self.used_sprinklers),
            'available': [pos for pos in self.sprinkler_positions 
                         if pos not in self.used_sprinklers and 
                         pos not in self.active_sprinklers]
        }
        return status

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
        self.people_types = {}  # 记录人员类型
        
        self.distance_field = self._compute_distance_field()
        
        # 创建路径查找器用于人员移动
        self.path_finder = PathFinder(building_3d)
        
        # 增强楼梯使用激励
        self.stair_usage_bonus = 35  # 进一步增加楼梯使用奖励
        self.floor_descent_bonus = 40  # 增加下楼奖励
        self.exit_attraction = 60  # 出口吸引力
        
        # 添加楼层感知
        self.floor_awareness = True
        
        # 增加求生欲望
        self.survival_desire = 0.95  # 求生欲望强度
        self.fire_avoidance = 0.6   # 进一步降低火焰回避程度
        
        # 添加秩序撤离参数
        self.orderly_evacuation = True
        self.stair_order_factor = 0.2  # 楼梯秩序因子
        
        # 添加高层逃生激励
        self.high_floor_urgency = 1.2  # 高层紧迫感因子
        
        # 添加楼梯下楼时间消耗和逃生时间
        self.stair_descent_time = 5  # 通过楼梯下楼需要的时间
        self.stair_escape_time = 3   # 通过1楼楼梯逃生需要的时间
        
        # 添加楼梯使用统计
        self.stair_escape_count = 0
        self.stair_descent_count = 0
        
        # 不同年龄段的移动概率调整
        self.age_move_adjustments = {
            'young': 1.2,   # 年轻人移动更积极
            'adult': 1.0,   # 成年人正常移动
            'elder': 0.7    # 老年人移动较慢
        }
        
    def _compute_distance_field(self):
        """计算到出口的距离场，考虑通过楼梯的路径"""
        distance_field = []
        
        for floor in range(self.floors):
            floor_distances = [[float('inf') for _ in range(self.cols)] for _ in range(self.rows)]
            
            queue = deque()
            
            # 对于1楼，出口就是实际出口
            if floor == 0:
                for exit_pos in self.exits:
                    if exit_pos[0] == floor:
                        queue.append((exit_pos[1], exit_pos[2], 0))
                        floor_distances[exit_pos[1]][exit_pos[2]] = 0
            else:
                # 对于高层，楼梯是通往出口的关键点
                # 找到所有楼梯位置
                for i in range(self.rows):
                    for j in range(self.cols):
                        if self.building_3d[floor][i][j] == 4:
                            # 楼梯到1楼出口的距离估算
                            stair_to_exit_distance = floor * 20  # 每层楼额外距离
                            queue.append((i, j, stair_to_exit_distance))
                            floor_distances[i][j] = stair_to_exit_distance
            
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
    
    def set_person_type(self, position, person_type):
        """设置人员类型"""
        self.people_types[position] = person_type
    
    def get_person_type(self, position):
        """获取人员类型，默认为成年人"""
        return self.people_types.get(position, 'adult')
    
    def update_people_movement(self, elapsed_time, smoke_grid=None):
        self.time_passed = elapsed_time
        
        self._initialize_people_states()
        
        # 随时间增加移动概率
        time_factor = min(1.0, elapsed_time / 60)
        base_move_prob = self.move_probability * (1 + time_factor * 0.8)  # 增加移动概率增长
        
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
                
            # 检查是否被火焰完全包围（超过5格）
            if self._is_surrounded_by_fire(floor, i, j):
                print(f"Person at ({floor},{i},{j}) is surrounded by fire, cannot move")
                continue
            
            # 根据人员类型调整移动概率
            person_type = self.get_person_type(pos)
            age_adjustment = self.age_move_adjustments.get(person_type, 1.0)
            current_move_prob = base_move_prob * age_adjustment
                
            if random.random() < current_move_prob:
                result = self._move_person_intelligently(floor, i, j, smoke_grid, person_type)
                if result == "self_rescue":
                    self_rescue_count += 1
                elif result == "stair_escape":
                    self_rescue_count += 1
                    self.stair_escape_count += 1
                elif result == "stair_descent":
                    people_moved += 1
                    self.stair_descent_count += 1
                elif result:
                    people_moved += 1
        
        return self_rescue_count, people_moved
    
    def _is_surrounded_by_fire(self, floor, i, j):
        """检查是否被火焰完全包围（超过5格）"""
        fire_count = 0
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0:
                    continue
                ni, nj = i + di, j + dj
                if (0 <= ni < self.rows and 0 <= nj < self.cols and
                    self.building_3d[floor][ni][nj] == -1):
                    fire_count += 1
        
        return fire_count >= 5
    
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
                if pos in self.people_types:
                    del self.people_types[pos]
        
        # 为新人员添加状态
        for pos in current_people:
            if pos not in self.people_states:
                self.people_states[pos] = {
                    'total_fire_exposure': 0,
                    'consecutive_fire_exposure': 0,
                    'last_update_time': 0,
                    'in_fire_count': 0,
                    'stair_usage_count': 0,
                    'last_stair_usage': 0
                }
    
    def _move_person_intelligently(self, floor, i, j, smoke_grid=None, person_type='adult'):
        # 检查当前位置是否是出口
        if self.building_3d[floor][i][j] in [2, 3]:
            return self._perform_self_rescue(floor, i, j)
        
        # 新增：检查当前位置是否是楼梯且在一楼（可以直接逃生）
        if self.building_3d[floor][i][j] == 4 and floor == 0:
            return self._perform_stair_escape(floor, i, j)
        
        # 新增：检查当前位置是否是楼梯且不在1楼（可以下楼）
        if self.building_3d[floor][i][j] == 4 and floor > 0:
            return self._perform_stair_descent(floor, i, j)
        
        # 增强的楼梯使用逻辑 - 有秩序撤离
        if self._should_use_stairs(floor, i, j):
            stair_move = self._move_to_stairs_orderly(floor, i, j, smoke_grid, person_type)
            if stair_move:
                return stair_move
        
        # 如果是走廊人员，使用增强的走廊移动逻辑
        if self._is_in_corridor(floor, i, j):
            corridor_move = self._enhanced_corridor_move(floor, i, j, smoke_grid, person_type)
            if corridor_move:
                return corridor_move
        
        # 原有移动逻辑，但增强楼梯使用
        return self._enhanced_room_person_move(floor, i, j, smoke_grid, person_type)

    def _perform_stair_descent(self, floor, i, j):
        """通过楼梯下楼"""
        if self.building_3d[floor][i][j] == 1:
            # 找到下一层的楼梯位置（假设楼梯在每层的相同位置）
            next_floor = floor - 1  # 下楼
            if next_floor >= 0:  # 确保不会下到负楼层
                # 检查下一层楼梯位置是否可用
                if self.building_3d[next_floor][i][j] in [0, 4]:
                    # 从当前位置移除人员
                    self.building_3d[floor][i][j] = 0
                    # 在下一层楼梯位置放置人员
                    self.building_3d[next_floor][i][j] = 1
                    
                    # 更新人员状态和类型
                    old_pos = (floor, i, j)
                    new_pos = (next_floor, i, j)
                    if old_pos in self.people_states:
                        self.people_states[new_pos] = self.people_states[old_pos]
                        del self.people_states[old_pos]
                    if old_pos in self.people_types:
                        self.people_types[new_pos] = self.people_types[old_pos]
                        del self.people_types[old_pos]
                    
                    # 记录楼梯使用
                    if new_pos in self.people_states:
                        self.people_states[new_pos]['stair_usage_count'] += 1
                        self.people_states[new_pos]['last_stair_usage'] = self.time_passed
                    
                    print(f"Person descended from floor {floor} to floor {next_floor} at position ({i},{j})")
                    return "stair_descent"
        
        return False

    def _perform_stair_escape(self, floor, i, j):
        """通过楼梯逃生（仅限1楼）"""
        if self.building_3d[floor][i][j] == 1:
            self.building_3d[floor][i][j] = 0
            
            # 更新人员状态
            if (floor, i, j) in self.people_states:
                del self.people_states[(floor, i, j)]
            if (floor, i, j) in self.people_types:
                del self.people_types[(floor, i, j)]
            
            print(f"Person escaped via stairs at floor {floor}, position ({i},{j})")
            return "stair_escape"
        return False

    def _enhanced_room_person_move(self, floor, i, j, smoke_grid, person_type='adult'):
        """增强的房间人员移动逻辑 - 特别优化高层人员行为"""
        neighbors = self.path_finder._get_neighbors((floor, i, j))
        
        # 优先检查是否可以移动到出口
        for neighbor in neighbors:
            nf, ni, nj = neighbor
            if self.building_3d[nf][ni][nj] in [2, 3]:
                return self._perform_self_rescue(floor, i, j, nf, ni, nj)
            
            # 新增：检查是否可以移动到1楼的楼梯（可以直接逃生）
            if self.building_3d[nf][ni][nj] == 4 and nf == 0:
                # 直接逃生，不需要先移动到楼梯位置
                return self._perform_stair_escape(floor, i, j)
            
            # 新增：检查是否可以移动到楼梯（可以下楼）
            if self.building_3d[nf][ni][nj] == 4 and nf == floor:
                # 直接下楼，不需要先移动到楼梯位置
                return self._perform_stair_descent(floor, i, j)
        
        possible_moves = []
        move_scores = []
        
        current_safe = self._is_position_safe(floor, i, j)
        
        # 高层紧迫感 - 楼层越高，逃生紧迫性越强
        floor_urgency = 1.0 + (floor / self.floors) * self.high_floor_urgency
        
        # 年龄紧迫感 - 老人和小孩更紧迫
        age_urgency = 1.0
        if person_type in ['young', 'elder']:
            age_urgency = 1.3  # 年轻人和老人紧迫感更强
        
        for neighbor in neighbors:
            nf, ni, nj = neighbor
            
            if self.building_3d[nf][ni][nj] == -3:
                continue
                
            # 允许通过火焰区域
            if self.building_3d[nf][ni][nj] in [0, 1, 2, 3, 4, -1]:
                is_safe = self._is_position_safe(nf, ni, nj)
                
                # 基础距离得分 - 增强求生欲望和楼层紧迫感
                current_dist = self.distance_field[floor][i][j]
                new_dist = self.distance_field[nf][ni][nj]
                distance_score = (current_dist - new_dist) * self.survival_desire * floor_urgency * age_urgency
                
                # 烟雾惩罚 - 进一步降低惩罚
                smoke_penalty = 0
                if smoke_grid and smoke_grid[nf][ni][nj] > 0.3:
                    smoke_penalty = smoke_grid[nf][ni][nj] * 1.2  # 进一步降低惩罚系数
                
                # 火焰惩罚 - 降低惩罚，允许通过火焰
                fire_penalty = 0
                if self.building_3d[nf][ni][nj] == -1:
                    fire_penalty = 1.5 * self.fire_avoidance  # 进一步降低火焰惩罚
                
                # 恐慌因素
                time_factor = min(1.0, self.time_passed / 60)
                panic_factor = random.uniform(-self.panic_level * (1 + time_factor), 
                                            self.panic_level * (1 + time_factor))
                
                # 安全奖励
                safety_bonus = 0
                if not current_safe and is_safe:
                    safety_bonus = 25  # 增加安全奖励
                elif current_safe and not is_safe:
                    safety_bonus = -20
                
                # 出口和楼梯奖励 - 大幅增加，特别是对高层人员
                exit_bonus = 0
                if self.building_3d[nf][ni][nj] in [2, 3]:
                    exit_bonus = self.exit_attraction * floor_urgency * age_urgency
                elif self.building_3d[nf][ni][nj] == 4:
                    exit_bonus = self.stair_usage_bonus * floor_urgency * age_urgency
                    # 如果是下楼，额外奖励 - 高层人员更鼓励下楼
                    if nf < floor:
                        exit_bonus += self.floor_descent_bonus * (floor - nf) * floor_urgency * age_urgency
                    # 新增：如果是1楼楼梯，额外逃生奖励
                    if nf == 0:
                        exit_bonus += self.exit_attraction * 0.8  # 1楼楼梯相当于出口的80%吸引力
                    # 新增：如果是同层楼梯，额外下楼奖励
                    elif nf == floor:
                        exit_bonus += self.stair_usage_bonus * 0.5  # 同层楼梯也有奖励
                
                # 楼层奖励 - 强烈鼓励向低楼层移动
                floor_bonus = 0
                if self.floor_awareness and nf != floor:
                    if nf < floor:  # 下楼
                        floor_bonus = (floor - nf) * 25 * floor_urgency * age_urgency  # 大幅增加下楼奖励
                    else:  # 上楼 - 通常不鼓励，除非必要
                        floor_bonus = -15
                
                # 求生奖励 - 新增
                survival_bonus = 0
                if self.building_3d[floor][i][j] == -1:  # 如果当前位置有火
                    survival_bonus = 30 * floor_urgency * age_urgency  # 强烈求生欲望
                
                # 高层特殊奖励 - 鼓励尽快下楼
                if floor > 1 and nf < floor:
                    high_floor_bonus = floor * 8  # 楼层越高，下楼奖励越大
                else:
                    high_floor_bonus = 0
                
                # 年龄特殊奖励 - 老人和小孩有额外逃生奖励
                age_bonus = 0
                if person_type in ['young', 'elder']:
                    age_bonus = 15  # 年轻人和老人额外奖励
                
                total_score = (distance_score * self.intelligence_level - 
                             smoke_penalty - fire_penalty + panic_factor + 
                             safety_bonus + exit_bonus + floor_bonus + 
                             survival_bonus + high_floor_bonus + age_bonus)
                
                possible_moves.append((nf, ni, nj))
                move_scores.append(total_score)
        
        if possible_moves:
            best_move_index = np.argmax(move_scores)
            best_floor, best_i, best_j = possible_moves[best_move_index]
            
            if self.building_3d[best_floor][best_i][best_j] in [2, 3]:
                return self._perform_self_rescue(floor, i, j, best_floor, best_i, best_j)
            elif self.building_3d[best_floor][best_i][best_j] == 4 and best_floor == 0:
                # 直接逃生
                return self._perform_stair_escape(floor, i, j)
            elif self.building_3d[best_floor][best_i][best_j] == 4 and best_floor == floor:
                # 直接下楼
                return self._perform_stair_descent(floor, i, j)
            else:
                return self._perform_move(floor, i, j, best_floor, best_i, best_j)
        
        return False

    def _enhanced_corridor_move(self, floor, i, j, smoke_grid, person_type='adult'):
        """增强的走廊人员移动逻辑"""
        # 如果在1楼走廊，直接寻找出口或楼梯
        if floor == 0:
            # 先检查当前位置是否是出口或楼梯
            if self.building_3d[floor][i][j] in [2, 3]:
                return self._perform_self_rescue(floor, i, j)
            elif self.building_3d[floor][i][j] == 4:
                return self._perform_stair_escape(floor, i, j)
            
            return self._move_to_exit_from_corridor(floor, i, j, smoke_grid, person_type)
        else:
            # 在楼上走廊，优先下楼
            # 先检查当前位置是否是楼梯
            if self.building_3d[floor][i][j] == 4:
                return self._perform_stair_descent(floor, i, j)
            
            stair_move = self._move_to_stairs_orderly(floor, i, j, smoke_grid, person_type)
            if stair_move:
                return stair_move
            # 如果无法下楼，寻找本层出口路径
            return self._move_to_exit_from_corridor(floor, i, j, smoke_grid, person_type)

    def _move_to_stairs_orderly(self, floor, i, j, smoke_grid, person_type='adult'):
        """有秩序地移动到楼梯"""
        neighbors = self.path_finder._get_neighbors((floor, i, j))
        
        # 优先选择楼梯
        stair_moves = []
        for neighbor in neighbors:
            nf, ni, nj = neighbor
            # 如果是楼梯
            if self.building_3d[nf][ni][nj] == 4:
                stair_moves.append((nf, ni, nj))
        
        if stair_moves:
            # 选择最近的楼梯，但考虑秩序因素
            best_stair = self._select_best_stair_orderly(floor, i, j, stair_moves)
            # 移动到楼梯位置
            if self._perform_move(floor, i, j, *best_stair):
                # 然后立即下楼（如果是同层楼梯）
                sf, si, sj = best_stair
                if sf == floor:  # 同层楼梯
                    return self._perform_stair_descent(sf, si, sj)
                return True
        
        return False

    def _should_use_stairs(self, floor, i, j):
        """判断是否应该使用楼梯 - 优化高层人员逻辑"""
        # 如果不在1楼，强烈建议使用楼梯
        if floor > 0:
            return True
            
        # 如果距离出口很远，也考虑使用楼梯（虽然在一楼，但可能有其他下楼路径）
        current_exit_dist = self.distance_field[floor][i][j]
        if current_exit_dist > 20:  # 增加距离阈值
            return True
        
        return False

    def _select_best_stair_orderly(self, floor, i, j, stair_positions):
        """有秩序地选择最佳楼梯"""
        # 计算每个楼梯的秩序得分
        stair_scores = []
        for stair_pos in stair_positions:
            sf, si, sj = stair_pos
            
            # 基础距离得分
            distance_score = 1.0 / (abs(si-i) + abs(sj-j) + 1)
            
            # 楼梯拥挤度（模拟秩序）
            stair_crowding = self._calculate_stair_crowding(sf, si, sj)
            crowding_penalty = stair_crowding * self.stair_order_factor
            
            # 总得分
            total_score = distance_score - crowding_penalty
            
            stair_scores.append(total_score)
        
        # 选择得分最高的楼梯
        best_index = np.argmax(stair_scores)
        return stair_positions[best_index]

    def _calculate_stair_crowding(self, floor, i, j):
        """计算楼梯拥挤度"""
        crowding = 0
        # 检查楼梯周围的人员数量
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0:
                    continue
                ni, nj = i + di, j + dj
                if (0 <= ni < self.rows and 0 <= nj < self.cols and
                    self.building_3d[floor][ni][nj] == 1):
                    crowding += 1
        
        return crowding / 8.0  # 归一化到0-1

    def _find_nearest_stairs(self, floor, i, j):
        """找到最近的楼梯位置"""
        stair_positions = []
        for stair_i in range(self.rows):
            for stair_j in range(self.cols):
                if self.building_3d[floor][stair_i][stair_j] == 4:
                    stair_positions.append((floor, stair_i, stair_j))
        return stair_positions

    def _move_towards_direction(self, floor, i, j, target_pos):
        """向目标方向移动一步"""
        _, target_i, target_j = target_pos
        
        # 计算方向
        di = 0
        if target_i > i:
            di = 1
        elif target_i < i:
            di = -1
            
        dj = 0  
        if target_j > j:
            dj = 1
        elif target_j < j:
            dj = -1
        
        # 尝试主要方向移动
        if di != 0 and 0 <= i+di < self.rows and self.building_3d[floor][i+di][j] in [0, 4]:
            return self._perform_move(floor, i, j, floor, i+di, j)
        elif dj != 0 and 0 <= j+dj < self.cols and self.building_3d[floor][i][j+dj] in [0, 4]:
            return self._perform_move(floor, i, j, floor, i, j+dj)
        
        return False

    def _move_to_exit_from_corridor(self, floor, i, j, smoke_grid, person_type='adult'):
        """从走廊移动到出口"""
        # 寻找最近出口
        nearest_exit, exit_dist = self._find_nearest_exit(floor, i, j)
        
        if nearest_exit:
            # 使用A*路径规划
            path_to_exit, _ = self.path_finder.find_path(
                (floor, i, j), nearest_exit, 
                has_people=False, smoke_grid=smoke_grid,
                person_type=person_type
            )
            
            if path_to_exit and len(path_to_exit) > 1:
                next_pos = path_to_exit[1]
                nf, ni, nj = next_pos
                
                if self.building_3d[nf][ni][nj] in [0, 2, 3, 4]:
                    if self.building_3d[nf][ni][nj] in [2, 3]:
                        return self._perform_self_rescue(floor, i, j, nf, ni, nj)
                    else:
                        return self._perform_move(floor, i, j, nf, ni, nj)
        
        return False

    def _perform_self_rescue(self, floor, i, j, exit_floor=None, exit_i=None, exit_j=None):
        """执行自救"""
        if exit_floor is not None and exit_i is not None and exit_j is not None:
            # 直接移动到指定出口位置
            self.building_3d[floor][i][j] = 0
            # 出口位置保持为出口，不设置为人员
            return "self_rescue"
        else:
            # 当前位置就是出口
            self.building_3d[floor][i][j] = 0
            return "self_rescue"

    def _perform_move(self, old_floor, old_i, old_j, new_floor, new_i, new_j):
        """执行移动"""
        # 确保目标位置是空的、楼梯或允许通过火焰
        if self.building_3d[new_floor][new_i][new_j] in [0, 4, -1]:
            # 保存人员类型
            person_type = self.get_person_type((old_floor, old_i, old_j))
            
            self.building_3d[old_floor][old_i][old_j] = 0
            self.building_3d[new_floor][new_i][new_j] = 1
            
            # 更新人员状态和类型
            old_pos = (old_floor, old_i, old_j)
            new_pos = (new_floor, new_i, new_j)
            if old_pos in self.people_states:
                self.people_states[new_pos] = self.people_states[old_pos]
                del self.people_states[old_pos]
            if old_pos in self.people_types:
                self.people_types[new_pos] = self.people_types[old_pos]
                del self.people_types[old_pos]
            
            # 记录楼梯使用
            if self.building_3d[new_floor][new_i][new_j] == 4:
                if new_pos in self.people_states:
                    self.people_states[new_pos]['stair_usage_count'] += 1
                    self.people_states[new_pos]['last_stair_usage'] = self.time_passed
            
            return True
        return False

    def _is_in_corridor(self, floor, i, j):
        """检查是否在走廊区域"""
        corridor_rows = [12, 13, 14]
        return i in corridor_rows

    def _find_nearest_exit(self, floor, i, j):
        """找到最近的出口"""
        min_dist = float('inf')
        nearest_exit = None
        
        for exit_pos in self.exits:
            if exit_pos[0] == floor:  # 同楼层出口
                dist = abs(i - exit_pos[1]) + abs(j - exit_pos[2])
                if dist < min_dist:
                    min_dist = dist
                    nearest_exit = exit_pos
        
        return nearest_exit, min_dist

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

    def update_people_death_detection(self, elapsed_time):
        """更合理的死亡检测 - 只考虑直接火中"""
        people_to_remove = []
        
        for floor in range(len(self.building_3d)):
            for i in range(len(self.building_3d[floor])):
                for j in range(len(self.building_3d[floor][0])):
                    if self.building_3d[floor][i][j] == 1:
                        pos_key = (floor, i, j)
                        
                        # 检查是否在火中
                        is_in_fire = self.building_3d[floor][i][j] == -1
                        
                        # 初始化人员状态
                        if pos_key not in self.people_states:
                            self.people_states[pos_key] = {
                                'total_fire_exposure': 0,
                                'consecutive_fire_exposure': 0,
                                'last_update_time': elapsed_time,
                                'in_fire_count': 0,
                                'stair_usage_count': 0,
                                'last_stair_usage': 0
                            }
                        
                        state = self.people_states[pos_key]
                        
                        # 更新暴露计数：只更新直接火中的情况
                        if is_in_fire:
                            # 直接在火中
                            state['in_fire_count'] += 1
                            state['consecutive_fire_exposure'] += 1
                        else:
                            # 不在火中，重置连续暴露计数
                            state['consecutive_fire_exposure'] = 0
                        
                        state['total_fire_exposure'] = state['in_fire_count']
                        state['last_update_time'] = elapsed_time
                        
                        # 死亡条件调整：直接火中累计20次或连续10次
                        if (state['in_fire_count'] >= 20 or 
                            state['consecutive_fire_exposure'] >= 10):
                            
                            people_to_remove.append((floor, i, j, state))
        
        return people_to_remove

class SmokeSpreadModel:
    def __init__(self, building_3d):
        self.building_3d = building_3d
        self.floors = len(building_3d)
        self.rows = len(building_3d[0])
        self.cols = len(building_3d[0][0])
        
        self.office_smoke_rate = 0.003  # 降低烟雾扩散速度
        self.corridor_smoke_rate = 0.001  # 降低烟雾扩散速度
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
                        distance_factor = max(0, 1 - min_fire_dist / 30)  # 增加影响距离
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
        return min_dist if min_dist < float('inf') else 30  # 增加影响距离
    
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
        self.k = 0.05  # 增加火焰蔓延速度
        self.fire_areas = []
        
    def update_fire_spread(self, elapsed_time):
        # 更新火源列表
        self.fire_sources = self._identify_fire_sources()
        self._identify_fire_areas()
        
        for fire_area in self.fire_areas:
            self._spread_fire_area(fire_area, elapsed_time)
    
    def _identify_fire_sources(self):
        fire_sources = []
        for floor in range(self.floors):
            for i in range(self.rows):
                for j in range(self.cols):
                    if self.building_3d[floor][i][j] == -1:
                        fire_sources.append((floor, i, j))
        return fire_sources
    
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
        # 使用指数增长模拟火势蔓延
        growth_rate = self.k * elapsed_time
        target_area = int(current_area * (1 + growth_rate))
        
        if target_area > current_area:
            new_fire_cells = target_area - current_area
            spread_candidates = self._get_spread_candidates(fire_area)
            
            if spread_candidates:
                num_to_spread = min(new_fire_cells, len(spread_candidates))
                
                # 按距离火源的远近加权选择
                weighted_candidates = []
                for candidate in spread_candidates:
                    min_dist = float('inf')
                    for fire_pos in fire_area:
                        dist = abs(candidate[1]-fire_pos[1]) + abs(candidate[2]-fire_pos[2])
                        if dist < min_dist:
                            min_dist = dist
                    # 距离越近，权重越高
                    weight = 1.0 / (min_dist + 1)
                    weighted_candidates.append((candidate, weight))
                
                # 按权重选择
                candidates, weights = zip(*weighted_candidates)
                total_weight = sum(weights)
                if total_weight > 0:
                    probabilities = [w / total_weight for w in weights]
                    spread_positions = np.random.choice(
                        len(candidates), 
                        size=min(num_to_spread, len(candidates)), 
                        p=probabilities, 
                        replace=False
                    )
                    
                    for idx in spread_positions:
                        floor, i, j = candidates[idx]
                        # 确保目标位置不是墙、出口或楼梯
                        if self.building_3d[floor][i][j] in [0, 1]:  # 空地或人员位置
                            self.building_3d[floor][i][j] = -1
                            print(f"Fire spread to ({floor},{i},{j})")
    
    def _get_spread_candidates(self, fire_area):
        candidates = set()
        
        for floor, i, j in fire_area:
            for df in [0]:  # 同层蔓延
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        if di == 0 and dj == 0:
                            continue
                        nf, ni, nj = floor + df, i + di, j + dj
                        if (0 <= nf < self.floors and 
                            0 <= ni < self.rows and 
                            0 <= nj < self.cols and
                            self.building_3d[nf][ni][nj] in [0, 1]):  # 空地或人员位置
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
        
        # 人员类型分布
        self.age_distribution = {
            'young': 0.2,   # 20% 年轻人
            'adult': 0.6,   # 60% 成年人
            'elder': 0.2    # 20% 老年人
        }
    
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
        
        # 人员类型映射
        people_type_mapping = {}
        
        for floor in range(self.floors):
            building = [[0 for _ in range(self.cols)] for _ in range(self.rows)]
            
            # 生成房间
            for room_id, room_info in all_rooms.items():
                start_row = room_info["start_row"]
                end_row = room_info["end_row"]
                start_col = room_info["start_col"]
                end_col = room_info["end_col"]
                
                room_rows = end_row - start_row + 1
                room_cols = end_col - start_col + 1
                
                if room_rows > 0 and room_cols > 0:
                    floor_people_type_mapping = self._generate_room_content(
                        building, start_row, end_row, start_col, end_col, floor)
                    people_type_mapping.update(floor_people_type_mapping)
                
                door_row, door_col = all_doors[room_id]
                if 0 <= door_row < self.rows and 0 <= door_col < self.cols:
                    building[door_row][door_col] = 0
            
            # 设置走廊
            for i in range(corridor_start_row, corridor_end_row + 1):
                for j in range(self.cols):
                    building[i][j] = 0
            
            # 设置楼梯 - 每层都有楼梯
            corridor_mid_row = (corridor_start_row + corridor_end_row) // 2
            building[corridor_mid_row][1] = 4  # 左侧楼梯
            building[corridor_mid_row][self.cols - 2] = 4  # 右侧楼梯
            
            building = self._add_obstacles(building)
            
            # 生成火焰（新的火种方式）
            if floor == 0:  # 只在第一层生成火焰
                self._generate_fire_seeds(building, floor)
            
            floor_people = sum(row.count(1) for row in building)
            floor_fires = sum(row.count(-1) for row in building)
            self.total_people += floor_people
            self.total_fires += floor_fires
            
            building_3d.append(building)
            room_layouts.append(all_rooms.copy())
            door_layouts.append(all_doors.copy())
        
        # 设置出口 - 只在第一层设置出口，其他楼层通过楼梯连接到1楼出口
        corridor_mid_row = (corridor_start_row + corridor_end_row) // 2
        if len(building_3d) > 0:
            building_3d[0][corridor_mid_row][0] = 2  # 左侧出口
            building_3d[0][corridor_mid_row][self.cols - 1] = 3  # 右侧出口
        
        return building_3d, room_layouts, door_layouts, people_type_mapping
    
    def _generate_room_content(self, building, start_row, end_row, start_col, end_col, floor):
        rows = end_row - start_row + 1
        cols = end_col - start_col + 1
        
        if rows <= 0 or cols <= 0:
            return {}
        
        # 调整人员生成概率为15%
        people_count = max(1, int(rows * cols * 0.15))  # 增加人员生成概率
        
        people_type_mapping = {}
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
            return {}
        
        if all_positions and len(all_positions) >= people_count:
            people_positions = random.sample(all_positions, people_count)
            for row, col in people_positions:
                building[row][col] = 1
                
                # 随机分配人员类型
                person_type = self._get_random_person_type()
                people_type_mapping[(floor, row, col)] = person_type
                
                if (row, col) in all_positions:
                    all_positions.remove((row, col))
        
        return people_type_mapping
    
    def _get_random_person_type(self):
        """随机生成人员类型"""
        rand = random.random()
        if rand < self.age_distribution['young']:
            return 'young'
        elif rand < self.age_distribution['young'] + self.age_distribution['adult']:
            return 'adult'
        else:
            return 'elder'
    
    def _generate_fire_seeds(self, building, floor):
        """生成火种，使初始火焰总面积约等于20%"""
        total_cells = self.rows * self.cols
        target_fire_cells = int(total_cells * 0.2)  # 目标火焰面积20%
        
        # 生成2-5个火种
        num_seeds = random.randint(2, 5)
        seeds = []
        
        # 选择火种位置（避开墙壁和出口）
        available_positions = []
        for i in range(1, self.rows-1):
            for j in range(1, self.cols-1):
                if building[i][j] in [0, 1]:  # 空地或人员位置
                    available_positions.append((i, j))
        
        if len(available_positions) < num_seeds:
            return
        
        seeds = random.sample(available_positions, num_seeds)
        
        # 计算每个火种应该产生的火焰面积
        fire_per_seed = max(1, target_fire_cells // num_seeds)
        
        # 从火种开始蔓延火焰
        for seed in seeds:
            seed_i, seed_j = seed
            self._spread_fire_from_seed(building, floor, seed_i, seed_j, fire_per_seed)
    
    def _spread_fire_from_seed(self, building, floor, start_i, start_j, target_area):
        """从火种开始蔓延火焰"""
        if building[start_i][start_j] != -1:
            building[start_i][start_j] = -1
        
        current_area = 1
        queue = deque([(start_i, start_j)])
        visited = set([(start_i, start_j)])
        
        while current_area < target_area and queue:
            i, j = queue.popleft()
            
            # 尝试向四个方向蔓延
            for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                ni, nj = i + di, j + dj
                
                if (0 <= ni < self.rows and 0 <= nj < self.cols and
                    (ni, nj) not in visited and
                    building[ni][nj] in [0, 1] and  # 空地或人员位置
                    random.random() < 0.7):  # 70%概率蔓延
                    
                    building[ni][nj] = -1
                    visited.add((ni, nj))
                    queue.append((ni, nj))
                    current_area += 1
                    
                    if current_area >= target_area:
                        break
    
    def _add_obstacles(self, building, obstacle_ratio=0.02):  # 降低障碍物密度
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

class BuildingVisualizer:
    def __init__(self, building_3d, rooms_3d, doors_3d):
        self.building_3d = building_3d
        self.rooms_3d = rooms_3d
        self.doors_3d = doors_3d
        self.floors = len(building_3d)
        self.rows = len(building_3d[0])
        self.cols = len(building_3d[0][0])
        
        # 创建更清晰的颜色映射
        self.cmap = ListedColormap([
            'white',    # 0: empty
            'blue',     # 1: people
            'green',    # 2: left exit
            'green',    # 3: right exit
            'red',      # -1: fire
            'gray',     # -3: obstacle
            'orange',   # 4: stairs
            'purple',   # 5: rescuer (unused)
            'cyan',     # 6: active sprinkler
            'lightblue', # 7: used sprinkler
            'lightgreen' # 8: available sprinkler
        ])
        
        # 创建子图 - 根据楼层数调整布局
        if self.floors <= 3:
            self.fig, self.axs = plt.subplots(1, self.floors, figsize=(5*self.floors, 5))
        else:
            rows = (self.floors + 1) // 2
            self.fig, self.axs = plt.subplots(rows, 2, figsize=(10, 5*rows))
            # 将2D轴数组展平
            self.axs = self.axs.flatten()
        
        if self.floors == 1:
            self.axs = [self.axs]
        
        self.current_time = 0
        self.rescue_operations = []
        self.people_states = {}
        self.sprinkler_status = None
        
    def update_display(self, current_time, rescue_operations=None, people_states=None, sprinkler_status=None):
        self.current_time = current_time
        if rescue_operations:
            self.rescue_operations = rescue_operations
        if people_states:
            self.people_states = people_states
        if sprinkler_status:
            self.sprinkler_status = sprinkler_status
            
        for floor in range(self.floors):
            if floor >= len(self.axs):
                break
                
            ax = self.axs[floor]
            ax.clear()
            
            # 创建显示矩阵
            display_grid = np.zeros((self.rows, self.cols))
            
            # 填充基本元素
            for i in range(self.rows):
                for j in range(self.cols):
                    cell_value = self.building_3d[floor][i][j]
                    # 将值映射到0-8的范围
                    if cell_value == -3:  # 障碍物
                        display_grid[i][j] = 5
                    elif cell_value == -1:  # 火
                        display_grid[i][j] = 4
                    elif cell_value == 0:  # 空地
                        display_grid[i][j] = 0
                    elif cell_value == 1:  # 人员
                        display_grid[i][j] = 1
                    elif cell_value == 2:  # 左出口
                        display_grid[i][j] = 2
                    elif cell_value == 3:  # 右出口
                        display_grid[i][j] = 3
                    elif cell_value == 4:  # 楼梯
                        display_grid[i][j] = 6
            
            # 显示洒水器状态
            if self.sprinkler_status:
                # 活跃洒水器
                for sprinkler_pos in self.sprinkler_status['active']:
                    if sprinkler_pos[0] == floor:
                        i, j = sprinkler_pos[1], sprinkler_pos[2]
                        display_grid[i][j] = 7  # 活跃洒水器
                
                # 已使用洒水器
                for sprinkler_pos in self.sprinkler_status['used']:
                    if sprinkler_pos[0] == floor:
                        i, j = sprinkler_pos[1], sprinkler_pos[2]
                        display_grid[i][j] = 8  # 已使用洒水器
                
                # 可用洒水器
                for sprinkler_pos in self.sprinkler_status['available']:
                    if sprinkler_pos[0] == floor:
                        i, j = sprinkler_pos[1], sprinkler_pos[2]
                        display_grid[i][j] = 9  # 可用洒水器
            
            # 显示网格
            im = ax.imshow(display_grid, cmap=self.cmap, vmin=0, vmax=9)
            
            # 添加房间边界
            for room_id, room_info in self.rooms_3d[floor].items():
                start_row = room_info["start_row"]
                end_row = room_info["end_row"]
                start_col = room_info["start_col"]
                end_col = room_info["end_col"]
                
                width = end_col - start_col + 1
                height = end_row - start_row + 1
                
                rect = patches.Rectangle(
                    (start_col - 0.5, start_row - 0.5), width, height,
                    linewidth=2, edgecolor='black', facecolor='none'
                )
                ax.add_patch(rect)
                
                # 添加房间编号
                center_row = (start_row + end_row) / 2
                center_col = (start_col + end_col) / 2
                ax.text(center_col, center_row, f"R{room_id}", 
                       ha='center', va='center', fontsize=8, color='black',
                       bbox=dict(boxstyle="round,pad=0.1", facecolor='white', alpha=0.7))
            
            # 添加门的位置
            for room_id, door_pos in self.doors_3d[floor].items():
                door_row, door_col = door_pos
                ax.plot(door_col, door_row, 'ks', markersize=6, markerfacecolor='yellow')
            
            # 添加人员状态信息
            for (f, i, j), state in self.people_states.items():
                if f == floor:
                    exposure = state.get('total_fire_exposure', 0)
                    consecutive = state.get('consecutive_fire_exposure', 0)
                    ax.text(j, i, f"E:{exposure}\nC:{consecutive}", 
                           fontsize=6, color='red', ha='center', va='center')
            
            ax.set_title(f'Floor {floor+1} - Time: {current_time:.1f}s')
            ax.set_xlabel('Column')
            ax.set_ylabel('Row')
            
            # 添加图例
            legend_elements = [
                plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='blue', markersize=10, label='People'),
                plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='red', markersize=10, label='Fire'),
                plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='green', markersize=10, label='Exit'),
                plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', markersize=10, label='Obstacle'),
                plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='orange', markersize=10, label='Stairs'),
                plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='yellow', markersize=10, label='Door'),
                plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='cyan', markersize=10, label='Active Sprinkler'),
                plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='lightblue', markersize=10, label='Used Sprinkler'),
                plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='lightgreen', markersize=10, label='Available Sprinkler')
            ]
            ax.legend(handles=legend_elements, loc='upper right', fontsize=8)
        
        # 隐藏多余的子图
        for i in range(self.floors, len(self.axs)):
            self.axs[i].set_visible(False)
        
        plt.tight_layout()
        plt.pause(0.1)
    
    def show_final_stats(self, rescue_score, rescue_timeline):
        print("\n" + "="*50)
        print("Final Rescue Statistics")
        print("="*50)
        print(f"Initial People: {rescue_score['initial_people']}")
        print(f"People Saved: {rescue_score['people_saved']}")
        print(f"People Self Rescued: {rescue_score['people_self_rescued']}")
        print(f"People Killed: {rescue_score['people_killed']}")
        print(f"Current People: {rescue_score['current_people']}")
        print(f"Rescue Rate: {rescue_score['rescue_rate']:.2%}")
        print(f"Total Rescue Time: {rescue_score['total_rescue_time']:.2f} seconds")
        
        # 显示人员类型统计
        if 'people_type_stats' in rescue_score:
            print("\n=== People Type Statistics ===")
            for person_type, stats in rescue_score['people_type_stats'].items():
                total = stats['saved'] + stats['killed'] + stats['self_rescued']
                print(f"{person_type.capitalize()}: Saved: {stats['saved']}, Killed: {stats['killed']}, Self Rescued: {stats['self_rescued']}, Total: {total}")
        
        # 显示剩余人员位置
        if rescue_score['current_people'] > 0:
            print(f"\nRemaining People Position Analysis:")
            remaining_people = []
            for floor in range(self.floors):
                for i in range(self.rows):
                    for j in range(self.cols):
                        if self.building_3d[floor][i][j] == 1:
                            remaining_people.append((floor, i, j))
            
            for pos in remaining_people:
                floor, i, j = pos
                status = self._analyze_position_status(floor, i, j)
                print(f"  Position({floor},{i},{j}): {status}")
        
        plt.show()

    def _analyze_position_status(self, floor, i, j):
        # 检查是否在火中
        if self.building_3d[floor][i][j] == -1:
            return "In Fire"
        
        # 检查周围是否有火
        has_fire_nearby = False
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0:
                    continue
                ni, nj = i + di, j + dj
                if (0 <= ni < self.rows and 0 <= nj < self.cols and
                    self.building_3d[floor][ni][nj] == -1):
                    has_fire_nearby = True
                    break
            if has_fire_nearby:
                break
        
        if has_fire_nearby:
            return "Near Fire"
        
        # 检查是否在房间内
        room_info = self._get_room_for_position(floor, i, j)
        if room_info:
            return f"In Room {room_info['room_id']}"
        
        # 检查是否在走廊
        if 12 <= i <= 14:
            return "In Corridor"
        
        return "Unknown Position"

    def _get_room_for_position(self, floor, i, j):
        for room_id, room_info in self.rooms_3d[floor].items():
            start_row = room_info["start_row"]
            end_row = room_info["end_row"]
            start_col = room_info["start_col"]
            end_col = room_info["end_col"]
            
            if (start_row <= i <= end_row and start_col <= j <= end_col):
                return {"room_id": room_id, "floor": floor}
        return None

class ImprovedRescuePlanner:
    def __init__(self, building_3d, rooms_3d, doors_3d, initial_people_count, people_type_mapping=None):
        self.original_building = copy.deepcopy(building_3d)
        self.building_3d = building_3d
        self.rooms_3d = rooms_3d
        self.doors_3d = doors_3d
        self.finder = PathFinder(building_3d)
        self.fire_model = FireSpreadModel(building_3d)
        self.smoke_model = SmokeSpreadModel(building_3d)
        
        # 添加洒水器系统
        self.sprinkler_system = SprinklerSystem(building_3d, rooms_3d)
        
        self.floors = len(building_3d)
        self.rows = len(building_3d[0]) if building_3d else 0
        self.cols = len(building_3d[0][0]) if building_3d and building_3d[0] else 0
        
        corridor_mid_row = 13
        
        # 救援人员起始位置
        self.left_rescuer_start = (0, corridor_mid_row, 0)
        
        if len(building_3d) > 0 and len(building_3d[0]) > 0:
            right_start_col = len(building_3d[0][0]) - 1
            self.right_rescuer_start = (0, corridor_mid_row, right_start_col)
        else:
            self.right_rescuer_start = (0, corridor_mid_row, 35)
        
        # 出口位置 - 所有楼层的楼梯位置都可以作为出口路径的一部分
        self.exits = []
        for floor in range(len(building_3d)):
            # 只有第一层有实际出口
            if floor == 0:
                self.exits.append((floor, corridor_mid_row, 0))  # 左侧出口
                if len(building_3d[floor]) > 0:
                    right_exit_col = len(building_3d[floor][0]) - 1
                    self.exits.append((floor, corridor_mid_row, right_exit_col))  # 右侧出口
        
        self.people_model = ImprovedIntelligentPeopleMovementModel(building_3d, rooms_3d, self.exits)
        
        # 设置人员类型映射
        if people_type_mapping:
            for pos, person_type in people_type_mapping.items():
                self.people_model.set_person_type(pos, person_type)
        
        self.fire_truck_available = True
        self.floor_rescue_status = [False] * len(building_3d)
        
        self.initial_people_count = initial_people_count
        print(f"Initial people count from generator: {self.initial_people_count}")
        
        # 添加准确的人员跟踪
        self.initial_people_positions = self._get_all_people_positions()
        print(f"Initial people positions: {len(self.initial_people_positions)}")
        
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
        self.update_interval = 2  # 每2秒更新一次环境状态
        
        # 灭火统计
        self.fires_extinguished = 0
        
        # 添加走廊救援跟踪
        self.corridor_rescues = 0
        self.corridor_people_positions = []
        
        # 添加被困人员跟踪
        self.trapped_people_positions = []
        
        # 增强高层救援优先级
        self.high_floor_rescue_priority = True
        
        # 添加救援效率改进
        self.rescue_radius = 12  # 增加救援半径
        
        # 修改人员移动更新逻辑，包含楼梯逃生和下楼
        self.stair_escape_count = 0
        self.stair_descent_count = 0
        
        # 人员类型统计
        self.people_type_stats = {
            'young': {'saved': 0, 'killed': 0, 'self_rescued': 0},
            'adult': {'saved': 0, 'killed': 0, 'self_rescued': 0},
            'elder': {'saved': 0, 'killed': 0, 'self_rescued': 0}
        }
    
    def _get_all_people_positions(self):
        """获取建筑中所有人员的初始位置"""
        people_positions = []
        for floor in range(len(self.building_3d)):
            for i in range(len(self.building_3d[floor])):
                for j in range(len(self.building_3d[floor][0])):
                    if self.building_3d[floor][i][j] == 1:
                        people_positions.append((floor, i, j))
        return people_positions
    
    def _update_trapped_people_tracking(self):
        """跟踪被困人员位置"""
        self.trapped_people_positions = []
        for floor in range(self.floors):
            for i in range(self.rows):
                for j in range(self.cols):
                    if self.building_3d[floor][i][j] == 1:
                        # 检查是否被火焰包围
                        if self.people_model._is_surrounded_by_fire(floor, i, j):
                            self.trapped_people_positions.append((floor, i, j))
                            print(f"Trapped person detected at ({floor},{i},{j})")
    
    def _update_corridor_people_tracking(self):
        """跟踪走廊中的人员位置"""
        self.corridor_people_positions = []
        corridor_rows = [12, 13, 14]  # 走廊所在行
        
        for floor in range(self.floors):
            for i in corridor_rows:
                for j in range(self.cols):
                    if self.building_3d[floor][i][j] == 1:
                        self.corridor_people_positions.append((floor, i, j))
    
    def execute_corridor_rescue(self, rescuer_pos, current_time):
        """救援走廊中的人员"""
        rescued_count = 0
        
        # 获取救援人员周围的走廊人员
        floor, x, y = rescuer_pos
        
        for corridor_pos in self.corridor_people_positions[:]:  # 使用副本遍历
            c_floor, c_x, c_y = corridor_pos
            
            if c_floor == floor and abs(c_x - x) <= self.rescue_radius and abs(c_y - y) <= self.rescue_radius:
                # 救援这个走廊人员
                if self.building_3d[c_floor][c_x][c_y] == 1:
                    # 记录人员类型统计
                    person_type = self.people_model.get_person_type(corridor_pos)
                    self.people_type_stats[person_type]['saved'] += 1
                    
                    self.building_3d[c_floor][c_x][c_y] = 0
                    rescued_count += 1
                    self.people_saved += 1
                    self.current_people_count -= 1
                    
                    # 从跟踪列表中移除
                    self.corridor_people_positions.remove(corridor_pos)
                    
                    self.rescue_timeline.append(("CORRIDOR_RESCUE", corridor_pos, current_time))
                    print(f"Corridor rescue: at time {current_time:.2f}s rescued person at {corridor_pos}")
        
        return rescued_count
    
    def execute_trapped_rescue(self, rescuer_pos, current_time):
        """救援被困人员"""
        rescued_count = 0
        
        # 获取救援人员周围的被困人员
        floor, x, y = rescuer_pos
        
        for trapped_pos in self.trapped_people_positions[:]:  # 使用副本遍历
            t_floor, t_x, t_y = trapped_pos
            
            if t_floor == floor and abs(t_x - x) <= self.rescue_radius and abs(t_y - y) <= self.rescue_radius:
                # 救援这个被困人员
                if self.building_3d[t_floor][t_x][t_y] == 1:
                    # 记录人员类型统计
                    person_type = self.people_model.get_person_type(trapped_pos)
                    self.people_type_stats[person_type]['saved'] += 1
                    
                    self.building_3d[t_floor][t_x][t_y] = 0
                    rescued_count += 1
                    self.people_saved += 1
                    self.current_people_count -= 1
                    
                    # 从跟踪列表中移除
                    self.trapped_people_positions.remove(trapped_pos)
                    
                    self.rescue_timeline.append(("TRAPPED_RESCUE", trapped_pos, current_time))
                    print(f"Trapped rescue: at time {current_time:.2f}s rescued trapped person at {trapped_pos}")
        
        return rescued_count
    
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
                
                # 增强高层房间的重要性计算
                floor_importance = (self.floors - floor) * 5  # 楼层越高，重要性越大
                
                room_info[room_key] = {
                    "floor": floor,
                    "room_id": room_id,
                    "people_count": people_count,
                    "fire_count": fire_count,
                    "area": (end_row - start_row + 1) * (end_col - start_col + 1),
                    "door_position": (floor, self.doors_3d[floor][room_id][0], self.doors_3d[floor][room_id][1]),
                    "center_position": (floor, center_row, center_col),
                    "importance": people_count * 10 + fire_count * 2 + floor_importance
                }
        
        return room_info
    
    def calculate_rescue_time(self, start_pos, room_key, room_info, elapsed_time=0):
        if room_key not in room_info:
            return float('inf')
            
        room_data = room_info[room_key]
        door_pos = room_data["door_position"]
        
        update_interval = 10
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
                    print(f"Self rescue successful: at time {elapsed_time:.2f}s")
            
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
        """使用改进的死亡检测"""
        people_to_remove = self.people_model.update_people_death_detection(current_time)
        
        for floor, i, j, state in people_to_remove:
            if self.building_3d[floor][i][j] == 1:
                # 记录人员类型统计
                person_type = self.people_model.get_person_type((floor, i, j))
                self.people_type_stats[person_type]['killed'] += 1
                
                self.building_3d[floor][i][j] = 0
                self.people_killed += 1
                self.current_people_count -= 1
                
                # 记录死亡事件
                death_reason = "In fire" if state['in_fire_count'] >= 5 else \
                              "Consecutive exposure" if state['consecutive_fire_exposure'] >= 15 else \
                              "Cumulative exposure"
                
                exposure_count = state['in_fire_count'] if death_reason == "In fire" else \
                               state['consecutive_fire_exposure'] if death_reason == "Consecutive exposure" else \
                               state['total_fire_exposure']
                
                self.rescue_timeline.append((
                    "DEATH", 
                    (floor, i, j), 
                    current_time,
                    f"{death_reason}({exposure_count})"
                ))
                
                print(f"Person died: Position({floor},{i},{j}) - {death_reason} {exposure_count} - Type: {person_type}")
                
                # 从状态字典中移除
                pos_key = (floor, i, j)
                if pos_key in self.people_model.people_states:
                    del self.people_model.people_states[pos_key]
                if pos_key in self.people_model.people_types:
                    del self.people_model.people_types[pos_key]
    
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
                    # 记录人员类型统计
                    person_type = self.people_model.get_person_type((floor, i, j))
                    self.people_type_stats[person_type]['saved'] += 1
                    
                    self.building_3d[floor][i][j] = 0
                    rescued_count += 1
                    self.people_saved += 1
                    self.current_people_count -= 1
                    self.rescue_timeline.append(("RESCUE", (floor, i, j), rescue_time))
                    print(f"Rescue executed: at time {rescue_time:.2f}s rescued person ({floor}, {i}, {j}) - room {room_key} - Type: {person_type}")
        
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
    
    def _search_and_rescue_trapped_people(self, floor, start_pos, current_time):
        """在指定楼层搜索并救援被困人员"""
        rescued_count = 0
        max_search_time = 30  # 最大搜索时间
        
        # 获取当前楼层的被困人员
        floor_trapped_people = [pos for pos in self.trapped_people_positions if pos[0] == floor]
        
        if not floor_trapped_people:
            return 0
        
        current_pos = start_pos
        search_start_time = current_time
        
        for target_pos in floor_trapped_people:
            # 检查搜索时间限制
            if current_time - search_start_time > max_search_time:
                print(f"Search time limit reached on floor {floor}")
                break
                
            # 计算到目标人员的路径
            path, travel_time = self.finder.find_path(
                current_pos, target_pos,
                has_people=False, smoke_grid=self.smoke_model.smoke_grid_3d
            )
            
            if travel_time == float('inf'):
                continue
                
            # 移动到目标位置
            current_time += travel_time
            current_pos = target_pos
            self._update_environment_to_time(current_time)
            
            # 救援目标人员
            if self.building_3d[floor][target_pos[1]][target_pos[2]] == 1:
                # 记录人员类型统计
                person_type = self.people_model.get_person_type(target_pos)
                self.people_type_stats[person_type]['saved'] += 1
                
                self.building_3d[floor][target_pos[1]][target_pos[2]] = 0
                rescued_count += 1
                self.people_saved += 1
                self.current_people_count -= 1
                
                # 从跟踪列表中移除
                if target_pos in self.trapped_people_positions:
                    self.trapped_people_positions.remove(target_pos)
                
                self.rescue_timeline.append(("TRAPPED_SEARCH_RESCUE", target_pos, current_time))
                print(f"Trapped search rescue: at time {current_time:.2f}s rescued trapped person at {target_pos}")
            
            # 短暂停留
            current_time += 1
        
        return rescued_count
    
    def _search_and_rescue_corridor_people(self, floor, start_pos, current_time):
        """在指定楼层搜索并救援走廊人员"""
        rescued_count = 0
        max_search_time = 30  # 最大搜索时间
        
        # 获取当前楼层的走廊人员
        floor_corridor_people = [pos for pos in self.corridor_people_positions if pos[0] == floor]
        
        if not floor_corridor_people:
            return 0
        
        current_pos = start_pos
        search_start_time = current_time
        
        for target_pos in floor_corridor_people:
            # 检查搜索时间限制
            if current_time - search_start_time > max_search_time:
                print(f"Search time limit reached on floor {floor}")
                break
                
            # 计算到目标人员的路径
            path, travel_time = self.finder.find_path(
                current_pos, target_pos,
                has_people=False, smoke_grid=self.smoke_model.smoke_grid_3d
            )
            
            if travel_time == float('inf'):
                continue
                
            # 移动到目标位置
            current_time += travel_time
            current_pos = target_pos
            self._update_environment_to_time(current_time)
            
            # 救援目标人员
            if self.building_3d[floor][target_pos[1]][target_pos[2]] == 1:
                # 记录人员类型统计
                person_type = self.people_model.get_person_type(target_pos)
                self.people_type_stats[person_type]['saved'] += 1
                
                self.building_3d[floor][target_pos[1]][target_pos[2]] = 0
                rescued_count += 1
                self.people_saved += 1
                self.current_people_count -= 1
                self.corridor_rescues += 1
                
                # 从跟踪列表中移除
                if target_pos in self.corridor_people_positions:
                    self.corridor_people_positions.remove(target_pos)
                
                self.rescue_timeline.append(("CORRIDOR_SEARCH_RESCUE", target_pos, current_time))
                print(f"Corridor search rescue: at time {current_time:.2f}s rescued person at {target_pos}")
            
            # 短暂停留
            current_time += 1
        
        return rescued_count
    
    def execute_complete_rescue_plan(self, left_rooms, right_rooms, room_info):
        """执行完整的救援计划，包含时间同步"""
        print(f"Starting complete rescue plan: {len(left_rooms)} rooms on left, {len(right_rooms)} rooms on right")
        
        # 重置模拟时间
        self.current_simulation_time = 0
        self.last_update_time = 0
        
        # 更新人员跟踪
        self._update_corridor_people_tracking()
        self._update_trapped_people_tracking()
        
        # 执行左侧救援路线
        left_rescuer_time = self._execute_rescue_route(
            left_rooms, room_info, self.left_rescuer_start, "Left")
        
        # 执行右侧救援路线  
        right_rescuer_time = self._execute_rescue_route(
            right_rooms, room_info, self.right_rescuer_start, "Right")
        
        total_time = max(left_rescuer_time, right_rescuer_time)
        
        # 救援结束后继续模拟一段时间，确保所有事件完成
        final_time = self._simulate_after_rescue(total_time)
        
        return final_time
    
    def _execute_rescue_route(self, room_sequence, room_info, start_pos, route_name):
        """执行单个救援路线 - 按楼层顺序救援"""
        current_time = 0
        current_pos = start_pos
        
        # 更新人员跟踪
        self._update_corridor_people_tracking()
        self._update_trapped_people_tracking()
        
        print(f"{route_name} rescue starting at position {start_pos}")
        print(f"Initial corridor people: {len(self.corridor_people_positions)}")
        print(f"Initial trapped people: {len(self.trapped_people_positions)}")
        
        # 按楼层分组房间
        rooms_by_floor = {}
        for room_key in room_sequence:
            if room_key not in room_info:
                continue
            floor = room_info[room_key]["floor"]
            if floor not in rooms_by_floor:
                rooms_by_floor[floor] = []
            rooms_by_floor[floor].append(room_key)
        
        # 按楼层顺序救援
        for floor in sorted(rooms_by_floor.keys()):
            print(f"{route_name} rescuer moving to floor {floor}")
            
            # 如果不在目标楼层，先移动到目标楼层
            if current_pos[0] != floor:
                # 找到目标楼层的楼梯位置
                stair_pos = (floor, current_pos[1], current_pos[2])
                path, travel_time = self.finder.find_path(
                    current_pos, stair_pos, 
                    has_people=False, smoke_grid=self.smoke_model.smoke_grid_3d
                )
                
                if travel_time != float('inf'):
                    # 在移动过程中检查并救援被困人员和走廊人员
                    step_interval = travel_time / max(1, len(path))
                    step_time = current_time
                    
                    for step_pos in path[1:]:  # 跳过起始位置
                        step_time += step_interval
                        self._update_environment_to_time(step_time)
                        
                        # 检查并救援当前位置的被困人员（优先）
                        trapped_rescued = self.execute_trapped_rescue(step_pos, step_time)
                        if trapped_rescued > 0:
                            print(f"{route_name} rescuer rescued {trapped_rescued} trapped people during movement")
                        
                        # 检查并救援当前位置的走廊人员
                        corridor_rescued = self.execute_corridor_rescue(step_pos, step_time)
                        if corridor_rescued > 0:
                            print(f"{route_name} rescuer rescued {corridor_rescued} corridor people during movement")
                    
                    current_time += travel_time
                    current_pos = stair_pos
                    self._update_environment_to_time(current_time)
                    print(f"{route_name} rescuer moved to floor {floor} at time {current_time:.2f}s")
            
            # 救援当前楼层的所有房间
            for room_key in rooms_by_floor[floor]:
                # 更新环境状态到当前时间
                self._update_environment_to_time(current_time)
                
                # 计算到房间的路径和时间
                path, travel_time = self.finder.find_path(
                    current_pos, room_info[room_key]["door_position"], 
                    has_people=False, smoke_grid=self.smoke_model.smoke_grid_3d
                )
                
                if travel_time == float('inf'):
                    print(f"Warning: {route_name} cannot reach room {room_key}, skipping")
                    continue
                    
                # 在移动过程中救援被困人员和走廊人员
                step_interval = travel_time / max(1, len(path))
                step_time = current_time
                
                for step_pos in path[1:]:
                    step_time += step_interval
                    self._update_environment_to_time(step_time)
                    
                    # 检查并救援被困人员（优先）
                    trapped_rescued = self.execute_trapped_rescue(step_pos, step_time)
                    if trapped_rescued > 0:
                        print(f"{route_name} rescuer rescued {trapped_rescued} trapped people en route to room {room_key}")
                    
                    # 检查并救援走廊人员
                    corridor_rescued = self.execute_corridor_rescue(step_pos, step_time)
                    if corridor_rescued > 0:
                        print(f"{route_name} rescuer rescued {corridor_rescued} corridor people en route to room {room_key}")
                
                current_time += travel_time
                current_pos = room_info[room_key]["door_position"]
                
                # 更新环境状态
                self._update_environment_to_time(current_time)
                
                # 执行房间救援
                rescued_count = self.execute_rescue_operation(room_key, room_info, current_time)
                print(f"{route_name} rescue: at time {current_time:.2f}s rescued {rescued_count} people from room {room_key}")
                
                # 短暂停留处理救援
                rescue_processing_time = max(1, rescued_count * 0.5)
                current_time += rescue_processing_time
                
                # 更新人员跟踪（房间救援可能影响人员分布）
                self._update_corridor_people_tracking()
                self._update_trapped_people_tracking()
                
                # 检查当前楼层是否还有人员
                if not self._has_people_on_floor(floor):
                    print(f"{route_name} rescuer: No more people on floor {floor}, moving to next floor")
                    break
            
            # 在当前楼层搜索并救援剩余的被困人员（优先）
            print(f"{route_name} rescuer: Searching for remaining trapped people on floor {floor}")
            floor_trapped_rescues = self._search_and_rescue_trapped_people(floor, current_pos, current_time)
            if floor_trapped_rescues > 0:
                print(f"{route_name} rescuer: Rescued {floor_trapped_rescues} additional trapped people on floor {floor}")
            
            # 在当前楼层搜索并救援剩余的走廊人员
            print(f"{route_name} rescuer: Searching for remaining corridor people on floor {floor}")
            floor_corridor_rescues = self._search_and_rescue_corridor_people(floor, current_pos, current_time)
            if floor_corridor_rescues > 0:
                print(f"{route_name} rescuer: Rescued {floor_corridor_rescues} additional corridor people on floor {floor}")
            
        return current_time
    
    def _has_people_on_floor(self, floor):
        """检查指定楼层是否还有人员"""
        for i in range(len(self.building_3d[floor])):
            for j in range(len(self.building_3d[floor][0])):
                if self.building_3d[floor][i][j] == 1:
                    return True
        return False
    
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
            
            # 更新洒水器系统
            self.sprinkler_system.update(self.smoke_model.smoke_grid_3d, time_step)
            
            # 更新人员移动和自救
            self_rescue_count, people_moved = self.people_model.update_people_movement(
                self.current_simulation_time, self.smoke_model.smoke_grid_3d
            )
            
            # 更新楼梯逃生统计
            self.stair_escape_count = self.people_model.stair_escape_count
            self.stair_descent_count = self.people_model.stair_descent_count
            
            # 记录自救事件（包括楼梯逃生和下楼）
            if self_rescue_count > 0:
                # 记录人员类型统计
                # 注意：这里简化处理，实际应该记录具体哪些人员自救
                # 这里假设自救人员按初始比例分布
                young_count = int(self_rescue_count * 0.2)
                elder_count = int(self_rescue_count * 0.2)
                adult_count = self_rescue_count - young_count - elder_count
                
                self.people_type_stats['young']['self_rescued'] += young_count
                self.people_type_stats['adult']['self_rescued'] += adult_count
                self.people_type_stats['elder']['self_rescued'] += elder_count
                
                self.people_self_rescued += self_rescue_count
                self.current_people_count -= self_rescue_count
                
                for _ in range(self_rescue_count):
                    self.rescue_timeline.append(("SELF_RESCUE", "Exit", self.current_simulation_time))
                
                print(f"Escape successful: at time {self.current_simulation_time:.2f}s, {self_rescue_count} people escaped")
            
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
        
        print("Rescue completed, starting post-rescue simulation...")
        
        for additional_time in range(0, max_additional_time, self.update_interval):
            current_time = rescue_end_time + additional_time
            
            # 更新环境
            self.fire_model.update_fire_spread(self.update_interval)
            self.smoke_model.update_smoke_spread(self.update_interval)
            
            # 更新洒水器系统
            self.sprinkler_system.update(self.smoke_model.smoke_grid_3d, self.update_interval)
            
            # 更新人员移动
            self_rescue_count, people_moved = self.people_model.update_people_movement(
                current_time, self.smoke_model.smoke_grid_3d
            )
            
            # 更新楼梯逃生统计
            self.stair_escape_count = self.people_model.stair_escape_count
            self.stair_descent_count = self.people_model.stair_descent_count
            
            # 记录自救
            if self_rescue_count > 0:
                # 记录人员类型统计
                young_count = int(self_rescue_count * 0.2)
                elder_count = int(self_rescue_count * 0.2)
                adult_count = self_rescue_count - young_count - elder_count
                
                self.people_type_stats['young']['self_rescued'] += young_count
                self.people_type_stats['adult']['self_rescued'] += adult_count
                self.people_type_stats['elder']['self_rescued'] += elder_count
                
                self.people_self_rescued += self_rescue_count
                self.current_people_count -= self_rescue_count
                for _ in range(self_rescue_count):
                    self.rescue_timeline.append(("SELF_RESCUE", "Exit", current_time))
                print(f"Post-rescue self rescue: at time {current_time:.2f}s, {self_rescue_count} people self rescued")
            
            # 更新移动计数
            if people_moved > 0:
                self.people_movement_count += people_moved
            
            # 检测死亡
            self._check_people_death(current_time)
            
            final_time = current_time
            
            # 如果没有人员剩余，提前结束
            if self.current_people_count <= 0:
                print("All people processed, ending simulation")
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
    
    def get_sprinkler_status(self):
        """获取洒水器状态"""
        return self.sprinkler_system.get_sprinkler_status()
    
    def calculate_rescue_score(self):
        total_rescued = self.people_saved + self.people_self_rescued
        total_lost = self.people_killed + self.current_people_count
        
        # 验证人员平衡
        if total_rescued + total_lost != self.initial_people_count:
            print(f"Warning: People count imbalance! Initial: {self.initial_people_count}, Counted: {total_rescued + total_lost}")
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
        
        # 楼梯逃生奖励
        if self.stair_escape_count > 0:
            stair_escape_bonus = (self.stair_escape_count / self.initial_people_count) * 8
            score += stair_escape_bonus
        
        # 灭火奖励
        if self.fires_extinguished > 0:
            fire_bonus = min(10, self.fires_extinguished * 0.5)
            score += fire_bonus
        
        # 走廊救援奖励
        if self.corridor_rescues > 0:
            corridor_bonus = min(5, self.corridor_rescues * 0.5)
            score += corridor_bonus
        
        # 被困人员救援奖励
        trapped_rescues = len([op for op in self.rescue_operations if op.get('trapped', False)])
        if trapped_rescues > 0:
            trapped_bonus = min(15, trapped_rescues * 1.0)
            score += trapped_bonus
        
        # 特殊人员救援奖励 - 优先救援老人和小孩
        special_rescues = self.people_type_stats['young']['saved'] + self.people_type_stats['elder']['saved']
        if special_rescues > 0:
            special_bonus = min(20, special_rescues * 1.5)
            score += special_bonus
        
        score = max(0, score)
        
        return {
            "score": score,
            "rescue_rate": rescue_rate,
            "initial_people": self.initial_people_count,
            "people_saved": self.people_saved,
            "people_self_rescued": self.people_self_rescued,
            "stair_escape_count": self.stair_escape_count,
            "stair_descent_count": self.stair_descent_count,
            "people_killed": self.people_killed,
            "total_rescue_time": total_rescue_time,
            "people_movement_count": self.people_movement_count,
            "rescue_operations": len(self.rescue_operations),
            "current_people": self.current_people_count,
            "successful_rescues": self.successful_rescues,
            "failed_rescues": self.failed_rescues,
            "total_processed": total_rescued + total_lost,
            "fires_extinguished": self.fires_extinguished,
            "corridor_rescues": self.corridor_rescues,
            "trapped_rescues": trapped_rescues,
            "people_type_stats": self.people_type_stats
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
        
        # 按楼层和重要性排序，优先救援高层和重要的房间
        sorted_rooms = sorted(room_keys, 
                            key=lambda x: (-room_info[x]['floor'],  # 高层优先
                                         -room_info[x]['importance']))  # 重要性优先
        
        left_rooms = []
        right_rooms = []
        
        # 将高层房间分配给两个救援队
        for i, room_key in enumerate(sorted_rooms):
            if i % 2 == 0:
                left_rooms.append(room_key)
            else:
                right_rooms.append(room_key)
        
        # 优化序列
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

class RealTimeRescueSimulator:
    def __init__(self, building_3d, rooms_3d, doors_3d, initial_people_count, people_type_mapping=None):
        self.building_3d = building_3d
        self.rooms_3d = rooms_3d
        self.doors_3d = doors_3d
        self.initial_people_count = initial_people_count
        
        self.visualizer = BuildingVisualizer(building_3d, rooms_3d, doors_3d)
        
        # 创建救援计划器
        self.rescue_planner = ImprovedRescuePlanner(
            copy.deepcopy(building_3d), rooms_3d, doors_3d, initial_people_count, people_type_mapping
        )
        
        # 获取房间信息和救援计划
        self.room_info = self.rescue_planner.get_room_info()
        self.multi_astar_planner = MultiAStarRescuePlanner(
            building_3d, rooms_3d, doors_3d, initial_people_count
        )
        self.left_rooms, self.right_rooms, _ = self.multi_astar_planner.solve()
        
        self.current_time = 0
        self.update_interval = 2  # 每2秒更新一次显示
        
    def run_simulation(self):
        print("Starting real-time rescue simulation...")
        print(f"Left rescue route: {len(self.left_rooms)} rooms")
        print(f"Right rescue route: {len(self.right_rooms)} rooms")
        
        # 初始显示
        self.visualizer.update_display(
            self.current_time, 
            people_states=self.rescue_planner.people_model.people_states,
            sprinkler_status=self.rescue_planner.get_sprinkler_status()
        )
        
        # 执行左侧救援
        left_time = self._execute_rescue_route(
            self.left_rooms, self.rescue_planner.left_rescuer_start, "Left")
        
        # 执行右侧救援
        right_time = self._execute_rescue_route(
            self.right_rooms, self.rescue_planner.right_rescuer_start, "Right")
        
        total_time = max(left_time, right_time)
        
        # 后续模拟
        final_time = self._simulate_after_rescue(total_time)
        
        # 显示最终统计
        rescue_score = self.rescue_planner.calculate_rescue_score()
        self.visualizer.show_final_stats(rescue_score, self.rescue_planner.rescue_timeline)
        
        return final_time, rescue_score
    
    def _execute_rescue_route(self, room_sequence, start_pos, route_name):
        current_time = 0
        current_pos = start_pos
        
        print(f"\n{route_name} rescue starting...")
        
        # 按楼层分组房间
        rooms_by_floor = {}
        for room_key in room_sequence:
            if room_key not in self.room_info:
                continue
            floor = self.room_info[room_key]["floor"]
            if floor not in rooms_by_floor:
                rooms_by_floor[floor] = []
            rooms_by_floor[floor].append(room_key)
        
        # 按楼层顺序救援
        for floor in sorted(rooms_by_floor.keys()):
            print(f"{route_name} rescuer moving to floor {floor}")
            
            # 如果不在目标楼层，先移动到目标楼层
            if current_pos[0] != floor:
                # 找到目标楼层的楼梯位置
                stair_pos = (floor, current_pos[1], current_pos[2])
                path, travel_time = self.rescue_planner.finder.find_path(
                    current_pos, stair_pos,
                    has_people=False, smoke_grid=self.rescue_planner.smoke_model.smoke_grid_3d
                )
                
                if travel_time != float('inf'):
                    # 模拟移动过程
                    step_time = current_time
                    for step_idx, step_pos in enumerate(path[1:], 1):
                        step_duration = travel_time * (step_idx / len(path))
                        step_time = current_time + step_duration
                        
                        # 更新救援人员位置
                        current_pos = step_pos
                        
                        # 更新环境状态
                        self._update_environment_to_time(step_time)
                        
                        # 检查并救援走廊人员
                        corridor_rescued = self.rescue_planner.execute_corridor_rescue(step_pos, step_time)
                        if corridor_rescued > 0:
                            print(f"{route_name} rescuer rescued {corridor_rescued} corridor people during movement")
                        
                        # 更新显示
                        if step_idx % 3 == 0 or step_idx == len(path) - 1:
                            self.visualizer.update_display(
                                step_time,
                                people_states=self.rescue_planner.people_model.people_states,
                                sprinkler_status=self.rescue_planner.get_sprinkler_status()
                            )
                    
                    current_time += travel_time
                    current_pos = stair_pos
                    self._update_environment_to_time(current_time)
                    print(f"{route_name} rescuer moved to floor {floor} at time {current_time:.2f}s")
            
            # 救援当前楼层的所有房间
            for room_key in rooms_by_floor[floor]:
                # 更新环境状态
                self._update_environment_to_time(current_time)
                
                # 计算路径和时间
                path, travel_time = self.rescue_planner.finder.find_path(
                    current_pos, self.room_info[room_key]["door_position"],
                    has_people=False, smoke_grid=self.rescue_planner.smoke_model.smoke_grid_3d
                )
                
                if travel_time == float('inf'):
                    print(f"Warning: Cannot reach room {room_key}")
                    continue
                
                # 模拟移动过程
                print(f"{route_name} rescue: Going to room {room_key}, estimated time: {travel_time:.2f}s")
                
                # 逐步移动并显示
                step_time = current_time
                for step_idx, step_pos in enumerate(path[1:], 1):
                    step_duration = travel_time * (step_idx / len(path))
                    step_time = current_time + step_duration
                    
                    # 更新救援人员位置
                    current_pos = step_pos
                    
                    # 更新环境状态
                    self._update_environment_to_time(step_time)
                    
                    # 检查并救援走廊人员
                    corridor_rescued = self.rescue_planner.execute_corridor_rescue(step_pos, step_time)
                    if corridor_rescued > 0:
                        print(f"{route_name} rescuer rescued {corridor_rescued} corridor people en route to room {room_key}")
                    
                    # 更新显示
                    if step_idx % 3 == 0 or step_idx == len(path) - 1:
                        self.visualizer.update_display(
                            step_time,
                            people_states=self.rescue_planner.people_model.people_states,
                            sprinkler_status=self.rescue_planner.get_sprinkler_status()
                        )
                
                current_time += travel_time
                current_pos = self.room_info[room_key]["door_position"]
                
                # 执行救援
                rescued_count = self.rescue_planner.execute_rescue_operation(
                    room_key, self.room_info, current_time)
                print(f"{route_name} rescue: at time {current_time:.2f}s rescued {rescued_count} people")
                
                # 更新显示
                self.visualizer.update_display(
                    current_time,
                    people_states=self.rescue_planner.people_model.people_states,
                    sprinkler_status=self.rescue_planner.get_sprinkler_status()
                )
                
                # 处理救援后的停留时间
                rescue_processing_time = max(1, rescued_count * 0.5)
                current_time += rescue_processing_time
                
                # 更新走廊人员跟踪
                self.rescue_planner._update_corridor_people_tracking()
                
                # 检查当前楼层是否还有人员
                if not self.rescue_planner._has_people_on_floor(floor):
                    print(f"{route_name} rescuer: No more people on floor {floor}, moving to next floor")
                    break
            
            # 在当前楼层搜索并救援剩余的走廊人员
            print(f"{route_name} rescuer: Searching for remaining corridor people on floor {floor}")
            floor_corridor_rescues = self.rescue_planner._search_and_rescue_corridor_people(floor, current_pos, current_time)
            if floor_corridor_rescues > 0:
                print(f"{route_name} rescuer: Rescued {floor_corridor_rescues} additional corridor people on floor {floor}")
        
        return current_time
    
    def _update_environment_to_time(self, target_time):
        while self.rescue_planner.current_simulation_time < target_time:
            time_step = min(self.rescue_planner.update_interval, 
                          target_time - self.rescue_planner.current_simulation_time)
            
            if time_step <= 0:
                break
            
            # 更新火灾蔓延
            self.rescue_planner.fire_model.update_fire_spread(time_step)
            
            # 更新烟雾扩散
            self.rescue_planner.smoke_model.update_smoke_spread(time_step)
            
            # 更新洒水器系统
            self.rescue_planner.sprinkler_system.update(
                self.rescue_planner.smoke_model.smoke_grid_3d, time_step)
            
            # 更新人员移动
            self_rescue_count, people_moved = self.rescue_planner.people_model.update_people_movement(
                self.rescue_planner.current_simulation_time, 
                self.rescue_planner.smoke_model.smoke_grid_3d
            )
            
            # 处理自救
            if self_rescue_count > 0:
                self.rescue_planner.people_self_rescued += self_rescue_count
                self.rescue_planner.current_people_count -= self_rescue_count
                for _ in range(self_rescue_count):
                    self.rescue_planner.rescue_timeline.append(
                        ("SELF_RESCUE", "Exit", self.rescue_planner.current_simulation_time))
                print(f"Self rescue successful: at time {self.rescue_planner.current_simulation_time:.2f}s, {self_rescue_count} people self rescued")
            
            # 更新移动计数
            if people_moved > 0:
                self.rescue_planner.people_movement_count += people_moved
            
            # 检测死亡
            self.rescue_planner._check_people_death(self.rescue_planner.current_simulation_time)
            
            # 更新时间
            self.rescue_planner.current_simulation_time += time_step
            
            # 定期更新显示
            if abs(self.rescue_planner.current_simulation_time - self.current_time) >= self.update_interval:
                self.current_time = self.rescue_planner.current_simulation_time
                self.visualizer.update_display(
                    self.current_time,
                    people_states=self.rescue_planner.people_model.people_states,
                    sprinkler_status=self.rescue_planner.get_sprinkler_status()
                )
    
    def _simulate_after_rescue(self, rescue_end_time):
        final_time = rescue_end_time
        max_additional_time = 300
        
        print("\nRescue completed, starting post-rescue simulation...")
        
        for additional_time in range(0, max_additional_time, self.rescue_planner.update_interval):
            current_time = rescue_end_time + additional_time
            
            # 更新环境
            self.rescue_planner.fire_model.update_fire_spread(self.rescue_planner.update_interval)
            self.rescue_planner.smoke_model.update_smoke_spread(self.rescue_planner.update_interval)
            
            # 更新洒水器系统
            self.rescue_planner.sprinkler_system.update(
                self.rescue_planner.smoke_model.smoke_grid_3d, self.rescue_planner.update_interval)
            
            # 更新人员移动
            self_rescue_count, people_moved = self.rescue_planner.people_model.update_people_movement(
                current_time, self.rescue_planner.smoke_model.smoke_grid_3d
            )
            
            # 处理自救
            if self_rescue_count > 0:
                self.rescue_planner.people_self_rescued += self_rescue_count
                self.rescue_planner.current_people_count -= self_rescue_count
                for _ in range(self_rescue_count):
                    self.rescue_planner.rescue_timeline.append(("SELF_RESCUE", "Exit", current_time))
                print(f"Post-rescue self rescue: at time {current_time:.2f}s, {self_rescue_count} people self rescued")
            
            # 更新移动计数
            if people_moved > 0:
                self.rescue_planner.people_movement_count += people_moved
            
            # 检测死亡
            self.rescue_planner._check_people_death(current_time)
            
            final_time = current_time
            
            # 更新显示
            if additional_time % 10 == 0:
                self.visualizer.update_display(
                    current_time,
                    people_states=self.rescue_planner.people_model.people_states,
                    sprinkler_status=self.rescue_planner.get_sprinkler_status()
                )
            
            # 如果没有人员剩余，提前结束
            if self.rescue_planner.current_people_count <= 0:
                print("All people processed, ending simulation")
                break
        
        return final_time

def print_rescue_score(rescue_score):
    print("\n=== Rescue Score ===")
    print(f"Initial People Count: {rescue_score['initial_people']}")
    print(f"People Saved: {rescue_score['people_saved']}")
    print(f"People Self Rescued: {rescue_score['people_self_rescued']}")
    print(f"Stair Escapes: {rescue_score['stair_escape_count']}")
    print(f"Stair Descents: {rescue_score['stair_descent_count']}")
    print(f"People Killed: {rescue_score['people_killed']}")
    print(f"Current People: {rescue_score['current_people']}")
    print(f"Fires Extinguished: {rescue_score.get('fires_extinguished', 0)}")
    print(f"Corridor Rescues: {rescue_score.get('corridor_rescues', 0)}")
    print(f"Rescue Rate: {rescue_score['rescue_rate']:.2%}")
    print(f"Total Rescue Time: {rescue_score['total_rescue_time']:.2f} seconds")
    
    # 打印人员类型统计
    if 'people_type_stats' in rescue_score:
        print("\n=== People Type Statistics ===")
        for person_type, stats in rescue_score['people_type_stats'].items():
            total = stats['saved'] + stats['killed'] + stats['self_rescued']
            print(f"{person_type.capitalize()}: Saved: {stats['saved']}, Killed: {stats['killed']}, Self Rescued: {stats['self_rescued']}, Total: {total}")
    
    print(f"Final Score: {rescue_score['score']:.2f}/100")
    
    # 计算人员平衡
    total_rescued = rescue_score['people_saved'] + rescue_score['people_self_rescued']
    total_lost = rescue_score['people_killed'] + rescue_score['current_people']
    balance_check = total_rescued + total_lost
    
    print(f"\nPeople Balance Check:")
    print(f"  Initial People: {rescue_score['initial_people']}")
    print(f"  Total Rescued: {total_rescued}")
    print(f"  Total Lost: {total_lost}")
    print(f"  Balance Result: {rescue_score['initial_people']} = {total_rescued} + {total_lost} = {balance_check}")
    
    if rescue_score['score'] >= 90:
        evaluation = "Excellent"
    elif rescue_score['score'] >= 80:
        evaluation = "Good"
    elif rescue_score['score'] >= 70:
        evaluation = "Average"
    elif rescue_score['score'] >= 60:
        evaluation = "Pass"
    else:
        evaluation = "Fail"
    
    print(f"Evaluation: {evaluation}")

def main():
    print("Generating multi-floor building layout...")
    generator = BuildingGenerator()
    building_3d, rooms_3d, doors_3d, people_type_mapping = generator.generate_building()
    
    print(f"Building Info: {generator.floors} floors, {generator.rooms_per_side} rooms per side")
    print(f"Building Size: {generator.rows} rows x {generator.cols} columns")
    print(f"Room Size: {generator.room_height} rows x {generator.room_width} columns")
    print(f"Corridor Width: {generator.corridor_width} rows")
    
    print(f"Initial People Count: {generator.total_people}")
    print(f"Initial Fire Count: {generator.total_fires}")
    
    # 统计人员类型
    type_count = {'young': 0, 'adult': 0, 'elder': 0}
    for person_type in people_type_mapping.values():
        type_count[person_type] += 1
    
    print(f"People Type Distribution: Young: {type_count['young']}, Adult: {type_count['adult']}, Elder: {type_count['elder']}")
    
    # 使用实时模拟器
    simulator = RealTimeRescueSimulator(
        building_3d, rooms_3d, doors_3d, generator.total_people, people_type_mapping
    )
    
    total_time, rescue_score = simulator.run_simulation()
    
    print(f"\nSimulation completed, total time: {total_time:.2f} seconds")
    print_rescue_score(rescue_score)

if __name__ == "__main__":
    main()