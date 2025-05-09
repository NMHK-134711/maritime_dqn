import numpy as np
from numba import jit, float64, int64, boolean

@jit(nopython=True)
def calculate_fuel_consumption(abs_action_angle, position, tidal_grid_dir, tidal_grid_speed, tidal_grid_valid, 
                               wind_grid_dir, wind_grid_speed, wind_grid_valid, n_rows, n_cols, f_0=1, V_s=6.68):
    row, col = position
    tidal_dir, tidal_speed = 0.0, 0.0
    if 0 <= row < n_rows and 0 <= col < n_cols and tidal_grid_valid[row, col]:
        tidal_dir = tidal_grid_dir[row, col]
        tidal_speed = tidal_grid_speed[row, col]
    wind_dir, wind_speed = 0.0, 0.0
    if 0 <= row < n_rows and 0 <= col < n_cols and wind_grid_valid[row, col]:
        wind_dir = wind_grid_dir[row, col]
        wind_speed = wind_grid_speed[row, col]
    
    tidal_dir_rad = (90.0 - tidal_dir) * np.pi / 180.0
    wind_dir_rad = (90.0 - wind_dir) * np.pi / 180.0
    action_angle_rad = (90.0 - abs_action_angle) * np.pi / 180.0
    
    theta_c = action_angle_rad - tidal_dir_rad
    theta_w = action_angle_rad - wind_dir_rad
    
    tidal_effect = (V_s - tidal_speed * np.cos(theta_c)) / V_s
    if tidal_effect <= 0:
        tidal_effect = 0.001
    f_tidal = f_0 * (tidal_effect ** 3)
    
    wind_effect = (V_s - wind_speed * np.cos(theta_w)) / V_s
    f_wind = f_0 * (wind_effect ** 2)
    
    total_fuel = f_tidal + f_wind
    return total_fuel

@jit(nopython=True)
def calculate_distance(end_pos, current_pos):
    rel_pos = end_pos - current_pos
    return np.linalg.norm(rel_pos)

@jit(nopython=True)
def calculate_angle(rel_pos):
    return (np.degrees(np.arctan2(rel_pos[1], -rel_pos[0])) % 360.0)

@jit(nopython=True)
def angle_to_grid_direction(abs_action_angle, grid_angles):
    angle_diff = np.abs(grid_angles - abs_action_angle)
    closest_idx = np.argmin(angle_diff)
    return closest_idx