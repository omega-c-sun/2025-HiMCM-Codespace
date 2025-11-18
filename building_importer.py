# building_importer.py
import json
import cv2
import numpy as np
from PIL import Image
import copy

class BuildingDataImporter:
    """建筑数据导入模块 - 支持多种格式输入"""
    
    def __init__(self):
        self.supported_formats = ['json', 'image', 'manual', 'bim_simple']
    
    def import_building_data(self, source, format_type, **kwargs):
        """
        主导入函数
        
        Args:
            source: 数据源（文件路径或数据字典）
            format_type: 数据格式类型 ['json', 'image', 'manual', 'bim_simple']
            **kwargs: 其他参数
        """
        if format_type not in self.supported_formats:
            raise ValueError(f"不支持的格式: {format_type}. 支持: {self.supported_formats}")
        
        if format_type == 'json':
            return self._import_from_json(source)
        elif format_type == 'image':
            return self._import_from_image(source, kwargs.get('num_floors', 3))
        elif format_type == 'manual':
            return self._import_manual(source)
        elif format_type == 'bim_simple':
            return self._import_bim_simple(source)
    
    def _import_from_json(self, file_path):
        """从JSON配置文件导入"""
        with open(file_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        return self._build_from_config(config)
    
    def _import_from_image(self, image_path, num_floors=3):
        """从图像文件导入建筑数据"""
        # 读取图像
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"无法读取图像文件: {image_path}")
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 二值化处理
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        building_3d = []
        rows, cols = binary.shape
        
        # 为每个楼层创建相似的布局
        for floor in range(num_floors):
            floor_grid = []
            for i in range(rows):
                row = []
                for j in range(cols):
                    pixel = binary[i, j]
                    
                    # 根据像素值映射到建筑元素
                    if pixel < 50:    # 黑色 - 墙/障碍物
                        row.append(-3)
                    elif pixel > 200: # 白色 - 空地
                        # 随机放置人员和火源
                        if np.random.random() < 0.08:  # 8%概率人员
                            row.append(1)
                        elif np.random.random() < 0.03:  # 3%概率火源
                            row.append(-1)
                        else:
                            row.append(0)
                    else:             # 灰色 - 特殊区域
                        if 100 < pixel < 150:  # 楼梯
                            row.append(4)
                        else:  # 走廊等
                            row.append(0)
                floor_grid.append(row)
            
            building_3d.append(floor_grid)
        
        # 生成房间和门信息
        rooms_3d, doors_3d = self._generate_room_door_info(building_3d)
        
        return building_3d, rooms_3d, doors_3d
    
    def _import_manual(self, manual_data):
        """从手动定义的数据导入"""
        if isinstance(manual_data, str):
            manual_data = json.loads(manual_data)
        
        return self._build_from_config(manual_data)
    
    def _import_bim_simple(self, bim_data):
        """从简化的BIM数据导入"""
        if isinstance(bim_data, str):
            bim_data = json.loads(bim_data)
        
        # 转换BIM数据为配置格式
        config = self._convert_bim_to_config(bim_data)
        return self._build_from_config(config)
    
    def _build_from_config(self, config):
        """从配置数据构建建筑"""
        building_3d = []
        rooms_3d = []
        doors_3d = []
        
        floors = config.get('floors', 3)
        rows = config.get('rows', 27)
        cols = config.get('cols', 20)
        
        for floor_num in range(floors):
            # 创建基础网格
            floor_grid = [[0 for _ in range(cols)] for _ in range(rows)]
            floor_rooms = {}
            floor_doors = {}
            
            # 处理每层的布局
            floor_plan = None
            for plan in config.get('floor_plans', []):
                if plan.get('floor') == floor_num:
                    floor_plan = plan
                    break
            
            if floor_plan:
                # 设置墙壁和房间
                for room in floor_plan.get('rooms', []):
                    room_id = room['id']
                    start_row, end_row = room['start_row'], room['end_row']
                    start_col, end_col = room['start_col'], room['end_col']
                    
                    # 记录房间信息
                    floor_rooms[room_id] = {
                        "start_row": start_row,
                        "end_row": end_row,
                        "start_col": start_col,
                        "end_col": end_col
                    }
                    
                    # 设置房间边界为墙
                    for i in range(start_row, end_row + 1):
                        for j in range(start_col, end_col + 1):
                            if i == start_row or i == end_row or j == start_col or j == end_col:
                                if floor_grid[i][j] == 0:  # 不覆盖已有元素
                                    floor_grid[i][j] = -3
                    
                    # 在房间内随机放置人员和火源
                    for i in range(start_row + 1, end_row):
                        for j in range(start_col + 1, end_col):
                            if np.random.random() < 0.15:  # 15%概率人员
                                floor_grid[i][j] = 1
                            elif np.random.random() < 0.05:  # 5%概率火源
                                floor_grid[i][j] = -1
            
                # 设置门
                for door in floor_plan.get('doors', []):
                    row, col = door['row'], door['col']
                    floor_grid[row][col] = 0  # 门位置可通行
                    floor_doors[door.get('room_id', 0)] = (row, col)
                
                # 设置出口（仅第一层）
                if floor_num == 0:
                    for exit_info in floor_plan.get('exits', []):
                        row, col = exit_info['row'], exit_info['col']
                        if exit_info['type'] == 'left':
                            floor_grid[row][col] = 2
                        else:
                            floor_grid[row][col] = 3
                
                # 设置楼梯
                for stair in floor_plan.get('stairs', []):
                    row, col = stair['row'], stair['col']
                    floor_grid[row][col] = 4
            
            building_3d.append(floor_grid)
            rooms_3d.append(floor_rooms)
            doors_3d.append(floor_doors)
        
        return building_3d, rooms_3d, doors_3d
    
    def _generate_room_door_info(self, building_3d):
        """从建筑网格生成房间和门信息（简化版本）"""
        rooms_3d = []
        doors_3d = []
        
        for floor_num, floor_grid in enumerate(building_3d):
            rows = len(floor_grid)
            cols = len(floor_grid[0]) if rows > 0 else 0
            
            floor_rooms = {}
            floor_doors = {}
            
            # 简化的房间检测 - 在实际应用中需要更复杂的算法
            room_id = 1
            visited = set()
            
            for i in range(rows):
                for j in range(cols):
                    if (i, j) not in visited and floor_grid[i][j] == 0:
                        # 找到连续的空地区域作为房间
                        room_cells = self._find_connected_area(floor_grid, i, j, visited)
                        if len(room_cells) > 4:  # 最小房间大小
                            min_i = min(cell[0] for cell in room_cells)
                            max_i = max(cell[0] for cell in room_cells)
                            min_j = min(cell[1] for cell in room_cells)
                            max_j = max(cell[1] for cell in room_cells)
                            
                            floor_rooms[room_id] = {
                                "start_row": min_i,
                                "end_row": max_i,
                                "start_col": min_j,
                                "end_col": max_j
                            }
                            room_id += 1
            
            # 简化的门检测 - 在走廊与房间交界处
            corridor_rows = [12, 13, 14]  # 假设的走廊位置
            for i in corridor_rows:
                if i < rows:
                    for j in range(cols):
                        if floor_grid[i][j] == 0:
                            # 检查上下是否是房间
                            if (i-1 >= 0 and floor_grid[i-1][j] == -3 and 
                                i+1 < rows and floor_grid[i+1][j] == -3):
                                # 找到可能的门位置
                                floor_doors[len(floor_doors) + 1] = (i, j)
            
            rooms_3d.append(floor_rooms)
            doors_3d.append(floor_doors)
        
        return rooms_3d, doors_3d
    
    def _find_connected_area(self, grid, start_i, start_j, visited):
        """找到连通的区域"""
        stack = [(start_i, start_j)]
        area = []
        rows, cols = len(grid), len(grid[0])
        
        while stack:
            i, j = stack.pop()
            if (i, j) in visited:
                continue
            
            visited.add((i, j))
            area.append((i, j))
            
            # 检查四个方向
            for di, dj in [(0,1), (1,0), (0,-1), (-1,0)]:
                ni, nj = i + di, j + dj
                if (0 <= ni < rows and 0 <= nj < cols and 
                    grid[ni][nj] == 0 and (ni, nj) not in visited):
                    stack.append((ni, nj))
        
        return area
    
    def _convert_bim_to_config(self, bim_data):
        """将BIM数据转换为配置格式"""
        config = {
            "floors": bim_data.get("number_of_floors", 3),
            "rows": 27,
            "cols": 20,
            "floor_plans": []
        }
        
        for floor_num in range(config["floors"]):
            floor_plan = {
                "floor": floor_num,
                "rooms": [],
                "doors": [],
                "exits": [],
                "stairs": []
            }
            
            # 处理BIM组件
            for component in bim_data.get("components", []):
                if component.get("floor") == floor_num:
                    comp_type = component.get("type", "")
                    
                    if comp_type == "Space":  # 房间
                        bbox = component.get("bounding_box", {})
                        floor_plan["rooms"].append({
                            "id": component.get("id", len(floor_plan["rooms"]) + 1),
                            "start_row": bbox.get("min_y", 0),
                            "end_row": bbox.get("max_y", 5),
                            "start_col": bbox.get("min_x", 0),
                            "end_col": bbox.get("max_x", 5)
                        })
                    
                    elif comp_type == "Door":
                        loc = component.get("location", {})
                        floor_plan["doors"].append({
                            "room_id": component.get("connects", [0])[0],
                            "row": loc.get("y", 0),
                            "col": loc.get("x", 0)
                        })
                    
                    elif comp_type == "Exit":
                        loc = component.get("location", {})
                        floor_plan["exits"].append({
                            "type": "left" if loc.get("x", 0) < config["cols"]/2 else "right",
                            "row": loc.get("y", 0),
                            "col": loc.get("x", 0)
                        })
                    
                    elif comp_type == "Stair":
                        loc = component.get("location", {})
                        floor_plan["stairs"].append({
                            "row": loc.get("y", 0),
                            "col": loc.get("x", 0)
                        })
            
            config["floor_plans"].append(floor_plan)
        
        return config

    def create_sample_config(self):
        """创建示例配置文件"""
        sample_config = {
            "floors": 3,
            "rows": 20,
            "cols": 20,
            "floor_plans": [
                {
                    "floor": 0,
                    "rooms": [
                        {"id": 1, "start_row": 1, "end_row": 5, "start_col": 1, "end_col": 8},
                        {"id": 2, "start_row": 1, "end_row": 5, "start_col": 11, "end_col": 18},
                        {"id": 3, "start_row": 7, "end_row": 11, "start_col": 1, "end_col": 8},
                        {"id": 4, "start_row": 7, "end_row": 11, "start_col": 11, "end_col": 18}
                    ],
                    "doors": [
                        {"room_id": 1, "row": 5, "col": 4},
                        {"room_id": 2, "row": 5, "col": 14},
                        {"room_id": 3, "row": 7, "col": 4},
                        {"room_id": 4, "row": 7, "col": 14}
                    ],
                    "exits": [
                        {"type": "left", "row": 10, "col": 0},
                        {"type": "right", "row": 10, "col": 19}
                    ],
                    "stairs": [
                        {"row": 10, "col": 5},
                        {"row": 10, "col": 14}
                    ]
                },
                {
                    "floor": 1,
                    "rooms": [
                        {"id": 5, "start_row": 1, "end_row": 5, "start_col": 1, "end_col": 8},
                        {"id": 6, "start_row": 1, "end_row": 5, "start_col": 11, "end_col": 18},
                        {"id": 7, "start_row": 7, "end_row": 11, "start_col": 1, "end_col": 8},
                        {"id": 8, "start_row": 7, "end_row": 11, "start_col": 11, "end_col": 18}
                    ],
                    "doors": [
                        {"room_id": 5, "row": 5, "col": 4},
                        {"room_id": 6, "row": 5, "col": 14},
                        {"room_id": 7, "row": 7, "col": 4},
                        {"room_id": 8, "row": 7, "col": 14}
                    ],
                    "stairs": [
                        {"row": 10, "col": 5},
                        {"row": 10, "col": 14}
                    ]
                },
                {
                    "floor": 2,
                    "rooms": [
                        {"id": 9, "start_row": 1, "end_row": 5, "start_col": 1, "end_col": 8},
                        {"id": 10, "start_row": 1, "end_row": 5, "start_col": 11, "end_col": 18}
                    ],
                    "doors": [
                        {"room_id": 9, "row": 5, "col": 4},
                        {"room_id": 10, "row": 5, "col": 14}
                    ],
                    "stairs": [
                        {"row": 10, "col": 5},
                        {"row": 10, "col": 14}
                    ]
                }
            ]
        }
        
        return sample_config