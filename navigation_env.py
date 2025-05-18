import heapq
import numpy as np
from datetime import datetime, timedelta
import os
from concurrent.futures import ThreadPoolExecutor
from utils import calculate_fuel_consumption, calculate_distance, calculate_angle, angle_to_grid_direction

class NavigationEnv:
    def __init__(self):
        self.grid = np.load('land_sea_grid_cartopy_downsized.npy')
        self.n_rows, self.n_cols = self.grid.shape
        self.lat_min, self.lat_max = 30.0, 38.0
        self.lon_min, self.lon_max = 120.0, 127.0
        self.start_pos = self.latlon_to_grid(37.46036, 126.52360)
        self.end_pos = self.latlon_to_grid(30.62828, 122.06400)
        self.step_time_minutes = 8.5
        self.max_steps = 500
        self.cumulative_time = 0
        self.step_count = 0
        self.tidal_data_dir = r"C:/baramproject/tidal_database_interpolated"
        self.wind_data_dir = r"C:/baramproject/wind_database_interpolated"
        self.action_space = np.array([-90, -45, 0, 45, 90], dtype=np.float64)
        self.current_direction = 0.0
        self.grid_directions = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
        self.grid_angles = np.array([0, 45, 90, 135, 180, 225, 270, 315], dtype=np.float64)
        self.f_0 = 1.0
        self.V_s = 6.68
        self.path = []
        self.tidal_cache = {}
        self.wind_cache = {}
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.tidal_future = None
        self.wind_future = None
        self.reset()

    def latlon_to_grid(self, lat, lon):
        row = int((self.lat_max - lat) / (self.lat_max - self.lat_min) * self.n_rows)
        col = int((lon - self.lon_min) / (self.lon_max - self.lon_min) * self.n_cols)
        return row, col

    def reset(self, start_time=None):
        start_date = datetime(2018, 1, 1, 0, 0)
        end_date = datetime(2018, 12, 29, 0, 0)
        if start_time is None:
            time_delta = (end_date - start_date).total_seconds()
            random_seconds = np.random.randint(0, int(time_delta / 60 / 30) + 1) * 30 * 60
            start_time = start_date + timedelta(seconds=random_seconds)
        
        self.current_pos = self.start_pos
        self.current_direction = 0.0
        self.visit_count = {}
        self.prev_action = None
        self.previous_direction = None
        self.current_time = start_time
        self.cumulative_time = 0
        self.load_tidal_data_async()
        self.load_wind_data_async()
        self.get_tidal_data()
        self.get_wind_data()
        self.map_tidal_to_grid()
        self.map_wind_to_grid()
        self.prev_distance = self.get_distance_to_end()
        self.initial_distance = self.get_distance_to_end()
        self.step_count = 0
        self.path = []
        return self._get_state()

    def get_relative_position_and_angle(self):
        rel_pos = np.array(self.end_pos, dtype=np.float64) - np.array(self.current_pos, dtype=np.float64)
        distance = calculate_distance(np.array(self.end_pos, dtype=np.float64), np.array(self.current_pos, dtype=np.float64))
        end_angle = calculate_angle(rel_pos)
        return rel_pos, distance, end_angle

    def get_distance_to_end(self):
        return calculate_distance(np.array(self.end_pos, dtype=np.float64), np.array(self.current_pos, dtype=np.float64))

    def angle_to_grid_direction(self, abs_action_angle):
        closest_idx = angle_to_grid_direction(abs_action_angle, self.grid_angles)
        return self.grid_directions[closest_idx]

    def load_tidal_data_async(self):
        time_str = self.current_time.strftime("%Y%m%d_%H%M")
        dir_file = os.path.join(self.tidal_data_dir, f"tidal_dir_{time_str}.npy")
        speed_file = os.path.join(self.tidal_data_dir, f"tidal_speed_{time_str}.npy")
        def load():
            if time_str in self.tidal_cache:
                return self.tidal_cache[time_str]
            elif os.path.exists(dir_file) and os.path.exists(speed_file):
                return {"dir": np.load(dir_file, mmap_mode='r'), "speed": np.load(speed_file, mmap_mode='r')}
            return None
        self.tidal_future = self.executor.submit(load)

    def get_tidal_data(self):
        if self.tidal_future:
            self.tidal_data = self.tidal_future.result()
            if self.tidal_data:
                time_str = self.current_time.strftime("%Y%m%d_%H%M")
                self.tidal_cache[time_str] = self.tidal_data
            self.tidal_future = None
        self.load_tidal_data_async()

    def load_wind_data_async(self):
        time_str = self.current_time.strftime("%Y%m%d_%H%M")
        dir_file = os.path.join(self.wind_data_dir, f"wind_dir_{time_str}.npy")
        speed_file = os.path.join(self.wind_data_dir, f"wind_speed_{time_str}.npy")
        def load():
            if time_str in self.wind_cache:
                return self.wind_cache[time_str]
            elif os.path.exists(dir_file) and os.path.exists(speed_file):
                return {"dir": np.load(dir_file, mmap_mode='r'), "speed": np.load(speed_file, mmap_mode='r')}
            return None
        self.wind_future = self.executor.submit(load)

    def get_wind_data(self):
        if self.wind_future:
            self.wind_data = self.wind_future.result()
            if self.wind_data:
                time_str = self.current_time.strftime("%Y%m%d_%H%M")
                self.wind_cache[time_str] = self.wind_data
            self.wind_future = None
        self.load_wind_data_async()  

    def map_tidal_to_grid(self):
        if self.tidal_data is not None:
            self.tidal_grid_dir = self.tidal_data["dir"]
            self.tidal_grid_speed = self.tidal_data["speed"]
            self.tidal_grid_valid = np.ones((self.n_rows, self.n_cols), dtype=np.bool_)
        else:
            self.tidal_grid_dir = np.zeros((self.n_rows, self.n_cols), dtype=np.float64)
            self.tidal_grid_speed = np.zeros((self.n_rows, self.n_cols), dtype=np.float64)
            self.tidal_grid_valid = np.zeros((self.n_rows, self.n_cols), dtype=np.bool_)

    def map_wind_to_grid(self):
        if self.wind_data is not None:
            self.wind_grid_dir = self.wind_data["dir"]
            self.wind_grid_speed = self.wind_data["speed"]
            self.wind_grid_valid = np.ones((self.n_rows, self.n_cols), dtype=np.bool_)
        else:
            self.wind_grid_dir = np.zeros((self.n_rows, self.n_cols), dtype=np.float64)
            self.wind_grid_speed = np.zeros((self.n_rows, self.n_cols), dtype=np.float64)
            self.wind_grid_valid = np.zeros((self.n_rows, self.n_cols), dtype=np.bool_)

    def get_neighbors(self, pos):
        """유효한 인접 위치(8방향)를 반환합니다."""
        neighbors = []
        for dr, dc in self.grid_directions:
            r, c = pos[0] + dr, pos[1] + dc
            if 0 <= r < self.n_rows and 0 <= c < self.n_cols and self.grid[r, c] == 0:
                neighbors.append((r, c))
        return neighbors

    def calculate_total_fuel(self, path):
        """경로를 따라 총 연료 소모량을 계산합니다."""
        total_fuel = 0.0
        for i in range(len(path) - 1):
            current_pos = path[i]
            next_pos = path[i + 1]
            direction = (next_pos[0] - current_pos[0], next_pos[1] - current_pos[1])
            dir_index = self.grid_directions.index(direction)
            abs_action_angle = self.grid_angles[dir_index]
            fuel = calculate_fuel_consumption(
                abs_action_angle, current_pos, self.tidal_grid_dir, self.tidal_grid_speed,
                self.tidal_grid_valid, self.wind_grid_dir, self.wind_grid_speed,
                self.wind_grid_valid, self.n_rows, self.n_cols, self.f_0, self.V_s
            )
            total_fuel += fuel
        return total_fuel

    def calculate_astar_path(self, start_time=None):
        if start_time is None:
            start_date = datetime(2018, 1, 1, 0, 0)
            end_date = datetime(2018, 12, 29, 0, 0)
            time_delta = (end_date - start_date).total_seconds()
            random_seconds = np.random.randint(0, int(time_delta / 60 / 30) + 1) * 30 * 60
            start_time = start_date + timedelta(seconds=random_seconds)
        
        # A* 데이터 구조 초기화
        open_set = []  # 우선순위 큐: (f, g, pos, parent)
        open_dict = {}  # 빠른 조회: pos -> (g, parent)
        closed_set = set()  # 방문한 노드
        
        # 시작 노드
        start_h = calculate_distance(self.start_pos, self.end_pos)
        heapq.heappush(open_set, (start_h, 0, self.start_pos, None))
        open_dict[self.start_pos] = (0, None)
        
        while open_set:
            # f 값이 가장 작은 노드 선택
            f, g, current_pos, parent = heapq.heappop(open_set)
            if current_pos == self.end_pos:
                # 경로 재구성
                path = []
                pos = current_pos
                while pos is not None:
                    path.append(pos)
                    pos = open_dict[pos][1] if pos in open_dict else None
                path = path[::-1]  # 시작점에서 끝점 순으로
                total_distance = g  # g는 총 이동 거리
                return path, total_distance
            
            if current_pos in closed_set:
                continue
            
            closed_set.add(current_pos)
            
            # 이웃 탐색
            for neighbor in self.get_neighbors(current_pos):
                if neighbor in closed_set:
                    continue
                
                # 이동 거리 계산 (직선: 1, 대각선: sqrt(2))
                dr = abs(neighbor[0] - current_pos[0])
                dc = abs(neighbor[1] - current_pos[1])
                move_cost = 1.4142135623730951 if dr == 1 and dc == 1 else 1.0  # sqrt(2) for diagonal
                
                tentative_g = g + move_cost
                h = calculate_distance(neighbor, self.end_pos)
                f = tentative_g + h
                
                if neighbor not in open_dict or tentative_g < open_dict[neighbor][0]:
                    open_dict[neighbor] = (tentative_g, current_pos)
                    heapq.heappush(open_set, (f, tentative_g, neighbor, current_pos))
        
        return None, None
    def step(self, action):
        self.step_count += 1
        rel_pos, distance, end_angle = self.get_relative_position_and_angle()
        rel_action_angle = self.action_space[action]
        abs_action_angle = (end_angle + rel_action_angle) % 360.0
        turn_penalty = 0.0
        if self.previous_direction is not None:
            angle_diff = min((abs_action_angle - self.previous_direction) % 360.0, 
                             (self.previous_direction - abs_action_angle) % 360.0)
            turn_penalty = angle_diff * 0.1
        move_dir = self.angle_to_grid_direction(abs_action_angle)
        new_pos = (self.current_pos[0] + move_dir[0], self.current_pos[1] + move_dir[1])
        
        current_fuel = calculate_fuel_consumption(abs_action_angle, self.current_pos, self.tidal_grid_dir, 
                                                  self.tidal_grid_speed, self.tidal_grid_valid, 
                                                  self.wind_grid_dir, self.wind_grid_speed, self.wind_grid_valid, 
                                                  self.n_rows, self.n_cols, self.f_0, self.V_s)
        
        moved = False
        if (0 <= new_pos[0] < self.n_rows and 0 <= new_pos[1] < self.n_cols and 
            self.grid[new_pos[0], new_pos[1]] == 0):
            self.current_pos = new_pos
            self.path.append(self.current_pos)
            moved = True
        self.previous_direction = abs_action_angle
        self.prev_action = action
        
        self.cumulative_time += self.step_time_minutes
        if self.cumulative_time >= 30:
            next_time = self.current_time + timedelta(minutes=30)
            end_date = datetime(2018, 12, 31, 23, 30)
            if next_time <= end_date:
                self.current_time = next_time
                self.get_tidal_data()
                if self.tidal_data is None:
                    print(f"No tidal data for {self.current_time}. Terminating episode.")
                    return self._get_state(), -1000, True, {}
                self.map_tidal_to_grid()
                self.get_wind_data()
                if self.wind_data is None:
                    print(f"No wind data for {self.current_time}. Terminating episode.")
                    return self._get_state(), -1000, True, {}
                self.map_wind_to_grid()
            else:
                print("Warning: Time exceeds 2018 range. Terminating episode.")
                return self._get_state(), -1000, True, {}
            self.cumulative_time -= 30
        
        state = self._get_state()
        current_distance = self.get_distance_to_end()
        
        pos_key = tuple(self.current_pos)
        self.visit_count[pos_key] = self.visit_count.get(pos_key, 0) + 1
        visit_penalty = -self.visit_count[pos_key] * 0.5
        
        distance_reward = 0.0
        if self.initial_distance > 0:
            linear_reward = (1 - current_distance / self.initial_distance) * 50.0
            change_reward = (self.prev_distance - current_distance) * 10.0
            distance_reward = linear_reward + change_reward
        
        self.prev_distance = current_distance
        goal_reward = 10000.0 if current_distance <= 1.0 else 0.0
        
        fuel_penalty = -current_fuel * 1.0
        reward = (fuel_penalty * 1.00 + distance_reward * 1.0 - turn_penalty * 1.0 + goal_reward + visit_penalty) * 0.0001
        
        if not moved:
            reward -= 2.0
        
        done = (current_distance <= 1.0) or (self.step_count >= self.max_steps)
        return state, reward, done, {}

    def _get_state(self):
        row, col = self.current_pos
        state = []
        
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                r = row + dr
                c = col + dc
                if 0 <= r < self.n_rows and 0 <= c < self.n_cols:
                    map_info = self.grid[r, c]
                    tidal_speed = self.tidal_grid_speed[r, c] if self.tidal_grid_valid[r, c] else 0.0
                    tidal_dir = self.tidal_grid_dir[r, c] if self.tidal_grid_valid[r, c] else 0.0
                    wind_speed = self.wind_grid_speed[r, c] if self.wind_grid_valid[r, c] else 0.0
                    wind_dir = self.wind_grid_dir[r, c] if self.wind_grid_valid[r, c] else 0.0
                    distance_to_end = calculate_distance(np.array(self.end_pos, dtype=np.float64), 
                                                        np.array([r, c], dtype=np.float64))
                else:
                    map_info = 1.0
                    tidal_speed = 0.0
                    tidal_dir = 0.0
                    wind_speed = 0.0
                    wind_dir = 0.0
                    distance_to_end = 0.0
                
                state.extend([map_info, tidal_speed, tidal_dir, wind_speed, wind_dir, distance_to_end])
        
        rel_pos = np.array(self.end_pos, dtype=np.float64) - np.array(self.current_pos, dtype=np.float64)
        end_angle = calculate_angle(rel_pos)
        state.append(end_angle)
        
        return np.array(state, dtype=np.float64)
    
    def calculate_straight_path(self, start_time=None):
        start_date = datetime(2018, 1, 1, 0, 0)
        end_date = datetime(2018, 12, 29, 0, 0)
        if start_time is None:
            time_delta = (end_date - start_date).total_seconds()
            random_seconds = np.random.randint(0, int(time_delta / 60 / 30) + 1) * 30 * 60
            start_time = start_date + timedelta(seconds=random_seconds)

        self.current_pos = self.start_pos
        self.current_time = start_time
        self.cumulative_time = 0
        self.load_tidal_data_async()
        self.load_wind_data_async()
        self.get_tidal_data()
        self.get_wind_data()
        self.map_tidal_to_grid()
        self.map_wind_to_grid()
        self.path = [self.current_pos]

        path = []
        current_pos = np.array(self.start_pos, dtype=np.float64)
        end_pos = np.array(self.end_pos, dtype=np.float64)
        direction = end_pos - current_pos
        distance = np.linalg.norm(direction)
        if distance == 0:
            return self.path, 0.0

        direction = direction / distance
        total_distance = 0.0
        max_steps = int(distance) + 1
        step_count = 0

        while step_count < max_steps:
            next_pos = current_pos + direction
            next_pos_int = (int(round(next_pos[0])), int(round(next_pos[1])))
            if not (0 <= next_pos_int[0] < self.n_rows and 0 <= next_pos_int[1] < self.n_cols):
                break
            if self.grid[next_pos_int[0], next_pos_int[1]] == 1:
                break
            current_pos = next_pos
            self.current_pos = next_pos_int
            path.append(self.current_pos)
            total_distance += 1.0 if step_count > 0 else 0.0
            step_count += 1
            if np.linalg.norm(np.array(self.current_pos) - end_pos) <= 1.0:
                break

        self.path = path
        return path, total_distance
    
    def calculate_path_fuel(self, path, start_time):
        if not path or len(path) < 2:
            return 0.0
        
        total_fuel = 0.0
        current_time = start_time
        cumulative_time = 0.0
        
        # 초기 환경 데이터 설정
        self.current_time = start_time
        self.get_tidal_data()
        self.map_tidal_to_grid()
        self.get_wind_data()
        self.map_wind_to_grid()

        for i in range(len(path) - 1):
            current_pos = path[i]
            next_pos = path[i + 1]
            direction = (next_pos[0] - current_pos[0], next_pos[1] - current_pos[1])
            dir_index = self.grid_directions.index(direction)
            abs_action_angle = self.grid_angles[dir_index]

            # 연료 소모량 계산
            fuel = calculate_fuel_consumption(
                abs_action_angle, current_pos, self.tidal_grid_dir,
                self.tidal_grid_speed, self.tidal_grid_valid,
                self.wind_grid_dir, self.wind_grid_speed, self.wind_grid_valid,
                self.n_rows, self.n_cols, self.f_0, self.V_s
            )
            total_fuel += fuel

            # 시간 업데이트 및 환경 데이터 갱신
            cumulative_time += self.step_time_minutes
            if cumulative_time >= 30:
                next_time = current_time + timedelta(minutes=30)
                end_date = datetime(2018, 12, 31, 23, 30)
                if next_time <= end_date:
                    current_time = next_time
                    self.current_time = current_time
                    self.get_tidal_data()
                    self.map_tidal_to_grid()
                    self.get_wind_data()
                    self.map_wind_to_grid()
                else:
                    print("Warning: Time exceeds 2018 range.")
                    break
                cumulative_time -= 30

        return total_fuel
