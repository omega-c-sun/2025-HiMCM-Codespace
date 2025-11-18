import heapq
import random
import numpy as np
from collections import defaultdict
import time
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import FancyArrowPatch

# Fix minus sign display
plt.rcParams["axes.unicode_minus"] = False


class PathFinder:
    def __init__(self, grid):
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])
        self.movement_speed = 1/6  # 6 cells/second = 1/6 seconds per cell
        self.fire_extinguish_time = 5  # 5 seconds to extinguish 1 cell of fire

    def find_path(self, start, end, has_people=False):
        """A* algorithm to find shortest path"""
        def heuristic(a, b):
            return abs(a[0]-b[0]) + abs(a[1]-b[1])  # Manhattan distance

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
                # Reconstruct path
                path = [current]
                total_time = g_score[(current, current_has_people)]
                while current in came_from:
                    current, current_has_people = came_from[(current, current_has_people)]
                    path.append(current)
                path.reverse()
                return path, total_time

            for neighbor in self._get_neighbors(current):
                nx, ny = neighbor

                # Calculate movement cost
                move_cost = self.movement_speed
                if self.grid[nx][ny] == -1 and current_has_people:  # Fire cell with people
                    move_cost += self.fire_extinguish_time

                # Skip walls
                if self.grid[nx][ny] == -2:
                    continue

                # Update people-carrying status
                new_has_people = current_has_people
                if self.grid[nx][ny] == 1:  # Person cell
                    new_has_people = True

                # Update scores
                tentative_g_score = g_score[(current, current_has_people)] + move_cost
                if tentative_g_score < g_score[(neighbor, new_has_people)]:
                    came_from[(neighbor, new_has_people)] = (current, current_has_people)
                    g_score[(neighbor, new_has_people)] = tentative_g_score
                    f_score[(neighbor, new_has_people)] = tentative_g_score + heuristic(neighbor, end)
                    heapq.heappush(open_set, (f_score[(neighbor, new_has_people)], neighbor, new_has_people))

        return [], float('inf')  # No path found

    def _get_neighbors(self, pos):
        """Get valid adjacent cells"""
        x, y = pos
        neighbors = []
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:  # Right, Down, Left, Up
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.rows and 0 <= ny < self.cols:
                neighbors.append((nx, ny))
        return neighbors


class BuildingGenerator:
    """Generate building layout with rooms, corridors, and hazards"""
    def __init__(self, rows=15, cols=36):
        self.rows = rows
        self.cols = cols

    def generate_building(self):
        # Initialize grid: -2 = wall (default)
        building = [[-2 for _ in range(self.cols)] for _ in range(self.rows)]

        # Create corridor (rows 6-8)
        hallway_start, hallway_end = 6, 8
        for i in range(hallway_start, hallway_end + 1):
            for j in range(self.cols):
                building[i][j] = 0  # 0 = empty space

        # Room door positions (row, col)
        room_doors = {
            1: (5, 6),   # Room 1 door
            2: (5, 18),  # Room 2 door
            3: (5, 30),  # Room 3 door
            4: (9, 6),   # Room 4 door
            5: (9, 18),  # Room 5 door
            6: (9, 30)   # Room 6 door
        }

        # Room boundaries (start_row, end_row, start_col, end_col)
        rooms = {
            1: {"start_row": 0, "end_row": 5, "start_col": 0, "end_col": 11},
            2: {"start_row": 0, "end_row": 5, "start_col": 12, "end_col": 23},
            3: {"start_row": 0, "end_row": 5, "start_col": 24, "end_col": 35},
            4: {"start_row": 9, "end_row": 14, "start_col": 0, "end_col": 11},
            5: {"start_row": 9, "end_row": 14, "start_col": 12, "end_col": 23},
            6: {"start_row": 9, "end_row": 14, "start_col": 24, "end_col": 35}
        }

        # Generate room interiors
        for room_id, info in rooms.items():
            room_rows = info["end_row"] - info["start_row"] + 1
            room_cols = info["end_col"] - info["start_col"] + 1
            room_grid = self._generate_room_layout(room_rows, room_cols)

            # Embed room into building
            for i in range(room_rows):
                for j in range(room_cols):
                    building[info["start_row"] + i][info["start_col"] + j] = room_grid[i][j]

            # Set door as empty space
            door_row, door_col = room_doors[room_id]
            building[door_row][door_col] = 0

        # Set rescuers' starting positions: 2 = left rescuer, 3 = right rescuer
        building[7][0] = 2
        building[7][35] = 3

        return building, rooms, room_doors

    def _generate_room_layout(self, rows=6, cols=12, fire_ratio=0.5, people_ratio=0.1):
        """Generate room interior with fires (-1) and people (1)"""
        total_cells = rows * cols
        fire_count = int(total_cells * fire_ratio)
        people_count = int(total_cells * people_ratio)

        all_positions = [(i, j) for i in range(rows) for j in range(cols)]
        fire_positions = random.sample(all_positions, fire_count)
        remaining_positions = [p for p in all_positions if p not in fire_positions]
        people_positions = random.sample(remaining_positions, min(people_count, len(remaining_positions)))

        # Initialize room: 0 = empty, -1 = fire, 1 = person
        room = [[0 for _ in range(cols)] for _ in range(rows)]
        for i, j in fire_positions:
            room[i][j] = -1
        for i, j in people_positions:
            room[i][j] = 1

        return room


class RescuePlanner:
    """Plan rescue routes for rescuers"""
    def __init__(self, building, rooms, room_doors):
        self.building = building
        self.rooms = rooms
        self.room_doors = room_doors
        self.finder = PathFinder(building)
        self.left_start = (7, 0)    # Left rescuer start
        self.right_start = (7, 35)  # Right rescuer start
        self.exits = [(7, 0), (7, 35)]  # Exit positions

    def get_room_info(self):
        """Get room details: people count, fire count, importance, etc."""
        room_info = {}
        for room_id, data in self.rooms.items():
            start_row, end_row = data["start_row"], data["end_row"]
            start_col, end_col = data["start_col"], data["end_col"]

            # Count people and fires in the room
            people_count = 0
            fire_count = 0
            for i in range(start_row, end_row + 1):
                for j in range(start_col, end_col + 1):
                    if self.building[i][j] == 1:
                        people_count += 1
                    elif self.building[i][j] == -1:
                        fire_count += 1

            room_info[room_id] = {
                "people": people_count,
                "fires": fire_count,
                "door": self.room_doors[room_id],
                "importance": people_count * 10 + fire_count * 2  # Prioritize people
            }
        return room_info

    def calculate_rescue_time(self, start_pos, room_id, room_info):
        """Calculate total time to rescue a room (to door + inside + to exit)"""
        door_pos = room_info[room_id]["door"]
        path_to_door, time_to_door = self.finder.find_path(start_pos, door_pos, has_people=False)

        # Time inside room: depends on people and fires
        room_time = (room_info[room_id]["people"] * 2 * self.finder.movement_speed +
                    room_info[room_id]["fires"] * self.finder.fire_extinguish_time)

        # Time from door to nearest exit
        exit_times = [self.finder.find_path(door_pos, exit_pos, has_people=True)[1] for exit_pos in self.exits]
        min_exit_time = min(exit_times)

        return time_to_door + room_time + min_exit_time, path_to_door

    def plan_rescue_routes(self):
        """Plan rescue routes for both rescuers"""
        room_info = self.get_room_info()
        
        # Get rescue paths for left and right rescuers
        left_rescue_path = [self.left_start]
        right_rescue_path = [self.right_start]
        left_rescue_time = 0.0
        right_rescue_time = 0.0

        # Prioritize rooms by importance
        sorted_rooms = sorted(room_info.keys(), key=lambda x: -room_info[x]["importance"])
        left_rooms = sorted_rooms[:3]  # Split rooms between rescuers
        right_rooms = sorted_rooms[3:]

        # Calculate left rescuer path
        current_pos = self.left_start
        for room_id in left_rooms:
            time_val, path = self.calculate_rescue_time(current_pos, room_id, room_info)
            left_rescue_time += time_val
            left_rescue_path.extend(path[1:])  # Avoid duplicate start
            current_pos = room_info[room_id]["door"]

        # Calculate right rescuer path
        current_pos = self.right_start
        for room_id in right_rooms:
            time_val, path = self.calculate_rescue_time(current_pos, room_id, room_info)
            right_rescue_time += time_val
            right_rescue_path.extend(path[1:])
            current_pos = room_info[room_id]["door"]

        total_rescue_time = max(left_rescue_time, right_rescue_time)
        
        return (left_rescue_path, right_rescue_path), (left_rescue_time, right_rescue_time, total_rescue_time)


class FireExtinguisher:
    """Plan fire extinguishing routes"""
    def __init__(self, building):
        self.building = building
        self.finder = PathFinder(building)
        self.left_start = (7, 0)    # Left extinguisher start
        self.right_start = (7, 35)  # Right extinguisher start

    def get_fire_locations(self):
        """Get all fire positions (-1 in grid)"""
        return [(i, j) for i in range(len(self.building)) 
                for j in range(len(self.building[0])) 
                if self.building[i][j] == -1]

    def optimize_extinguish_plan(self):
        """Split fires between rescuers and plan paths"""
        fires = self.get_fire_locations()
        if not fires:
            return [], [], 0

        # Assign fires to nearest rescuer
        left_fires = []
        right_fires = []
        for fire in fires:
            left_dist = abs(fire[0] - self.left_start[0]) + abs(fire[1] - self.left_start[1])
            right_dist = abs(fire[0] - self.right_start[0]) + abs(fire[1] - self.right_start[1])
            if left_dist <= right_dist:
                left_fires.append(fire)
            else:
                right_fires.append(fire)

        # Plan paths using nearest-neighbor heuristic
        left_path, left_time = self._plan_extinguish_path(left_fires, self.left_start)
        right_path, right_time = self._plan_extinguish_path(right_fires, self.right_start)

        return left_path, right_path, max(left_time, right_time)

    def _plan_extinguish_path(self, fires, start_pos):
        """Plan path to extinguish all fires in a group"""
        if not fires:
            return [], 0

        unvisited = fires.copy()
        current_pos = start_pos
        path = [current_pos]
        total_time = 0

        while unvisited:
            # Find nearest unvisited fire
            nearest_fire = min(unvisited, key=lambda f: abs(f[0]-current_pos[0]) + abs(f[1]-current_pos[1]))
            path_to_fire, time_to_fire = self.finder.find_path(current_pos, nearest_fire, has_people=False)

            # Update time and path
            total_time += time_to_fire + self.finder.fire_extinguish_time  # Add extinguish time
            current_pos = nearest_fire
            path.extend(path_to_fire[1:])  # Skip start (already in path)
            unvisited.remove(nearest_fire)

        # Return to start
        path_to_start, time_to_start = self.finder.find_path(current_pos, start_pos, has_people=False)
        total_time += time_to_start
        path.extend(path_to_start[1:])

        return path, total_time


# Visualization Functions (修复箭头显示问题)
def plot_combined_paths(building, rescue_paths, extinguish_paths):
    """Plot building layout with arrow-based paths for rescue and extinguish"""
    # Color mapping: walls, fire, empty, people, left rescuer, right rescuer
    cmap = ListedColormap(['#8B4513', '#FF4500', '#F5F5F5', '#1E90FF', '#32CD32', '#FFD700'])
    labels = ['Wall', 'Fire', 'Empty', 'Person', 'Left Rescuer', 'Right Rescuer']

    # Adjust grid values for colormap (shift from -2~3 to 0~5)
    mapped_building = np.array(building) + 2

    fig, ax = plt.subplots(figsize=(16, 10))
    ax.imshow(mapped_building, cmap=cmap, origin='upper')

    # 提取救援路径
    left_rescue, right_rescue = rescue_paths
    
    # 绘制左侧救援路径箭头 - 使用更简单直接的方法
    if left_rescue and len(left_rescue) >= 2:
        # 提取坐标 (注意: matplotlib中x是列，y是行)
        x_coords = [pos[1] for pos in left_rescue]  # 列坐标
        y_coords = [pos[0] for pos in left_rescue]  # 行坐标
        
        # 绘制路径线
        ax.plot(x_coords, y_coords, color='lime', linewidth=3, alpha=0.7, label='Left Rescue Path')
        
        # 在关键点添加箭头
        step = max(1, len(left_rescue) // 8)  # 减少步长，增加箭头数量
        for i in range(0, len(left_rescue)-step, step):
            if i+step < len(left_rescue):
                x1, y1 = left_rescue[i][1], left_rescue[i][0]
                x2, y2 = left_rescue[i+step][1], left_rescue[i+step][0]
                
                # 使用annotate绘制箭头
                ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                           arrowprops=dict(arrowstyle='->', color='lime', lw=2, alpha=0.8))
                
                # 添加顺序编号
                if i % (step*2) == 0:  # 每隔几个箭头添加一个编号
                    ax.text(x1, y1, str(i//step + 1), fontsize=9, ha='center', va='center',
                           bbox=dict(boxstyle="circle,pad=0.3", facecolor='lime', alpha=0.8))

    # 绘制右侧救援路径箭头
    if right_rescue and len(right_rescue) >= 2:
        x_coords = [pos[1] for pos in right_rescue]
        y_coords = [pos[0] for pos in right_rescue]
        
        ax.plot(x_coords, y_coords, color='gold', linewidth=3, alpha=0.7, label='Right Rescue Path')
        
        step = max(1, len(right_rescue) // 8)
        for i in range(0, len(right_rescue)-step, step):
            if i+step < len(right_rescue):
                x1, y1 = right_rescue[i][1], right_rescue[i][0]
                x2, y2 = right_rescue[i+step][1], right_rescue[i+step][0]
                
                ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                           arrowprops=dict(arrowstyle='->', color='gold', lw=2, alpha=0.8))
                
                if i % (step*2) == 0:
                    ax.text(x1, y1, str(i//step + 1), fontsize=9, ha='center', va='center',
                           bbox=dict(boxstyle="circle,pad=0.3", facecolor='gold', alpha=0.8))

    # 提取灭火路径
    left_ext, right_ext = extinguish_paths
    
    # 绘制左侧灭火路径箭头
    if left_ext and len(left_ext) >= 2:
        x_coords = [pos[1] for pos in left_ext]
        y_coords = [pos[0] for pos in left_ext]
        
        ax.plot(x_coords, y_coords, color='cyan', linewidth=2, alpha=0.7, linestyle='--', label='Left Extinguish Path')
        
        step = max(1, len(left_ext) // 10)
        for i in range(0, len(left_ext)-step, step):
            if i+step < len(left_ext):
                x1, y1 = left_ext[i][1], left_ext[i][0]
                x2, y2 = left_ext[i+step][1], left_ext[i+step][0]
                
                ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                           arrowprops=dict(arrowstyle='->', color='cyan', lw=1.5, alpha=0.8, linestyle='--'))
                
                if i % (step*3) == 0:
                    ax.text(x1, y1, str(i//step + 1), fontsize=8, ha='center', va='center',
                           bbox=dict(boxstyle="round,pad=0.2", facecolor='cyan', alpha=0.7))

    # 绘制右侧灭火路径箭头
    if right_ext and len(right_ext) >= 2:
        x_coords = [pos[1] for pos in right_ext]
        y_coords = [pos[0] for pos in right_ext]
        
        ax.plot(x_coords, y_coords, color='magenta', linewidth=2, alpha=0.7, linestyle='--', label='Right Extinguish Path')
        
        step = max(1, len(right_ext) // 10)
        for i in range(0, len(right_ext)-step, step):
            if i+step < len(right_ext):
                x1, y1 = right_ext[i][1], right_ext[i][0]
                x2, y2 = right_ext[i+step][1], right_ext[i+step][0]
                
                ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                           arrowprops=dict(arrowstyle='->', color='magenta', lw=1.5, alpha=0.8, linestyle='--'))
                
                if i % (step*3) == 0:
                    ax.text(x1, y1, str(i//step + 1), fontsize=8, ha='center', va='center',
                           bbox=dict(boxstyle="round,pad=0.2", facecolor='magenta', alpha=0.7))

    # 标记关键点位
    ax.scatter([0], [7], c='lime', s=200, marker='s', edgecolors='darkgreen', linewidth=2, label='Left Rescuer Start')
    ax.scatter([35], [7], c='gold', s=200, marker='s', edgecolors='darkorange', linewidth=2, label='Right Rescuer Start')
    
    # 标记灭火点
    all_fires = [(i, j) for i in range(len(building)) 
                for j in range(len(building[0])) 
                if building[i][j] == -1]
    if all_fires:
        fire_x = [pos[1] for pos in all_fires]
        fire_y = [pos[0] for pos in all_fires]
        ax.scatter(fire_x, fire_y, c='red', s=100, marker='X', label='Fire Locations')

    # 添加房间编号
    room_positions = {
        1: (2, 5), 2: (2, 18), 3: (2, 30),
        4: (12, 5), 5: (12, 18), 6: (12, 30)
    }
    for room_id, (row, col) in room_positions.items():
        ax.text(col, row, f'Room {room_id}', fontsize=9, ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

    # 添加标签和图例
    cbar = plt.colorbar(ax.imshow(mapped_building, cmap=cmap, origin='upper'), ticks=range(6))
    cbar.ax.set_yticklabels(labels)
    ax.set_title('Building Layout with Rescue and Extinguish Paths (Numbered Arrows Show Sequence)', fontsize=14, pad=20)
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    ax.grid(linestyle='--', alpha=0.5)
    
    # 改进图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='lime', edgecolor='darkgreen', label='Left Rescue Path'),
        Patch(facecolor='gold', edgecolor='darkorange', label='Right Rescue Path'),
        Patch(facecolor='cyan', edgecolor='blue', label='Left Extinguish Path'),
        Patch(facecolor='magenta', edgecolor='darkviolet', label='Right Extinguish Path'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='lime', 
                  markeredgecolor='darkgreen', markersize=10, label='Left Rescuer Start'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='gold', 
                  markeredgecolor='darkorange', markersize=10, label='Right Rescuer Start'),
        plt.Line2D([0], [0], marker='X', color='w', markerfacecolor='red', 
                  markersize=10, label='Fire Locations')
    ]
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.35, 1))
    
    plt.tight_layout()
    return fig


def plot_time_comparison(rescue_data, extinguish_data):
    """Plot bar chart comparing rescue and extinguish times"""
    left_rescue, right_rescue, total_rescue = rescue_data
    left_ext, right_ext, total_ext = extinguish_data

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Rescue time bar chart
    ax1.bar(['Left Rescuer', 'Right Rescuer', 'Total Rescue'],
            [left_rescue, right_rescue, total_rescue],
            color=['green', 'yellow', 'blue'])
    ax1.set_title('Rescue Time Comparison (seconds)')
    ax1.set_ylabel('Time (s)')
    for i, v in enumerate([left_rescue, right_rescue, total_rescue]):
        ax1.text(i, v + 0.5, f'{v:.1f}', ha='center')

    # Extinguish time bar chart
    ax2.bar(['Left Extinguisher', 'Right Extinguisher', 'Total Extinguish'],
            [left_ext, right_ext, total_ext],
            color=['blue', 'red', 'purple'])
    ax2.set_title('Extinguish Time Comparison (seconds)')
    ax2.set_ylabel('Time (s)')
    for i, v in enumerate([left_ext, right_ext, total_ext]):
        ax2.text(i, v + 0.5, f'{v:.1f}', ha='center')

    plt.tight_layout()
    return fig


def main():
    # Generate building
    print("Generating building layout...")
    generator = BuildingGenerator()
    building, rooms, room_doors = generator.generate_building()

    # Plan rescues
    print("Planning rescue routes...")
    rescue_planner = RescuePlanner(building, rooms, room_doors)
    rescue_paths, rescue_times = rescue_planner.plan_rescue_routes()
    left_rescue_path, right_rescue_path = rescue_paths
    left_rescue_time, right_rescue_time, total_rescue_time = rescue_times

    # Plan fire extinguishing
    print("Planning fire extinguishing routes...")
    fire_planner = FireExtinguisher(building)
    left_ext_path, right_ext_path, total_ext_time = fire_planner.optimize_extinguish_plan()

    # 计算灭火时间
    all_fires = fire_planner.get_fire_locations()
    left_fires = [f for f in all_fires 
                 if abs(f[0]-fire_planner.left_start[0]) + abs(f[1]-fire_planner.left_start[1]) 
                 <= abs(f[0]-fire_planner.right_start[0]) + abs(f[1]-fire_planner.right_start[1])]
    right_fires = [f for f in all_fires if f not in left_fires]
    
    left_ext_time = sum(fire_planner._plan_extinguish_path([f], fire_planner.left_start)[1] for f in left_fires)
    right_ext_time = sum(fire_planner._plan_extinguish_path([f], fire_planner.right_start)[1] for f in right_fires)

    # 输出结果
    print("\n=== Rescue Summary ===")
    print(f"Left rescuer time: {left_rescue_time:.1f}s")
    print(f"Right rescuer time: {right_rescue_time:.1f}s")
    print(f"Total rescue time: {total_rescue_time:.1f}s")

    print("\n=== Extinguish Summary ===")
    print(f"Left extinguisher time: {left_ext_time:.1f}s")
    print(f"Right extinguisher time: {right_ext_time:.1f}s")
    print(f"Total extinguish time: {total_ext_time:.1f}s")

    print(f"\n=== Total Operation Time ===")
    print(f"Total time (rescue + extinguish): {total_rescue_time + total_ext_time:.1f}s")

    # 绘图
    plot_combined_paths(building, 
                       (left_rescue_path, right_rescue_path), 
                       (left_ext_path, right_ext_path))
    plot_time_comparison((left_rescue_time, right_rescue_time, total_rescue_time),
                        (left_ext_time, right_ext_time, total_ext_time))
    plt.show()


if __name__ == "__main__":
    main()