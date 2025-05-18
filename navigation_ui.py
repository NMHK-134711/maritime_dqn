import sys
import os
import torch
import numpy as np
from datetime import datetime
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout
from PyQt5.uic import loadUi
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import json

from navigation_env import NavigationEnv
from dqn_agent import DQNAgent
from utils import calculate_fuel_consumption

class NavigationUI(QMainWindow):
    def __init__(self):
        super().__init__()
        ui_path = r"C:\baramproject\navigation_ui.ui"
        if not os.path.exists(ui_path):
            print(f"UI file not found at {ui_path}")
            sys.exit(1)
        loadUi(ui_path, self)
        self.setWindowTitle("Navigation Path Planner")

        # DQN 경로
        self.dqn_figure = Figure()
        self.dqn_canvas = FigureCanvas(self.dqn_figure)
        dqn_layout = QVBoxLayout(self.dqn_canvas_widget)
        dqn_layout.setContentsMargins(0, 0, 0, 0)  # 여백 제거
        dqn_layout.setSpacing(0)  # 간격 제거
        dqn_layout.addWidget(self.dqn_canvas)
        self.dqn_ax = self.dqn_figure.add_subplot(111)

        # 연료 소비량 창을 캔버스 위에 겹치게 그리기기
        self.maritime_dqn_fuel_cost.raise_()
        self.maritime_dqn_fuel_cost.setStyleSheet("background-color: rgba(255, 255, 255, 150);")

        # 직선 경로
        self.straight_figure = Figure()
        self.straight_canvas = FigureCanvas(self.straight_figure)
        straight_layout = QVBoxLayout(self.straight_canvas_widget)
        straight_layout.setContentsMargins(0, 0, 0, 0)
        straight_layout.setSpacing(0)
        straight_layout.addWidget(self.straight_canvas)
        self.straight_ax = self.straight_figure.add_subplot(111)
        self.straight_fuel_cost.raise_()
        self.straight_fuel_cost.setStyleSheet("background-color: rgba(255, 255, 255, 150);")

        # A-star 경로
        self.astar_figure = Figure()
        self.astar_canvas = FigureCanvas(self.astar_figure)
        astar_layout = QVBoxLayout(self.astar_canvas_widget)
        astar_layout.setContentsMargins(0, 0, 0, 0)
        astar_layout.setSpacing(0)
        astar_layout.addWidget(self.astar_canvas)
        self.astar_ax = self.astar_figure.add_subplot(111)
        self.astar_fuel_cost.raise_()
        self.astar_fuel_cost.setStyleSheet("background-color: rgba(255, 255, 255, 150);")

        # 모델과 환경 초기화
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state_dim = 55
        action_dim = 5
        self.agent = DQNAgent(state_dim, action_dim)
        model_path = r"C:\baramproject\trained_model\maritime_dqn_4\navigation_model.pth"
        if not os.path.exists(model_path):
            self.maritime_dqn_fuel_cost.setPlainText("Model not found!")
            return

        self.agent.policy_net.load_state_dict(torch.load(model_path, map_location=self.device))
        self.agent.policy_net.eval()

        tidal_dir = r"C:\baramproject\tidal_database_interpolated"
        wind_dir = r"C:\baramproject\wind_database_interpolated"
        if not os.path.exists(tidal_dir) or not os.path.exists(wind_dir):
            self.maritime_dqn_fuel_cost.setPlainText(f"Data directories missing: {tidal_dir} or {wind_dir}")
            return

        self.env = NavigationEnv()

        self.run_button.clicked.connect(self.run_path_planning)
        self.reset_button.clicked.connect(self.reset)

        # 초기값
        self.end_pos_lat.setText("30.62828")
        self.end_pos_lon.setText("122.06400")

    def plot_path(self, ax, canvas, grid, path, start_pos, end_pos):
        ax.clear()
        ax.imshow(grid, cmap='gray')
        path_array = np.array(path)
        if path_array.shape[0] > 1:
            ax.plot(path_array[:, 1], path_array[:, 0], 'r-', label='Path', linewidth=2, markersize=5)
        ax.plot(start_pos[1], start_pos[0], 'go', label='Start', markersize=5, alpha=0.8)
        ax.plot(end_pos[1], end_pos[0], 'bo', label='End', markersize=5, alpha=0.8)
        ax.set_xlim(0, grid.shape[1])
        ax.set_ylim(grid.shape[0], 0)  # y축 반전
        ax.set_aspect('equal')  # 비율 유지
        ax.legend()
        canvas.draw()

    def run_path_planning(self):
        try:
            year = self.comboBox.currentText()
            month = self.comboBox_2.currentText()
            day = self.comboBox_3.currentText()
            hour = self.comboBox_4.currentText()
            minute = self.comboBox_5.currentText()
            start_time_str = f"{year}-{month}-{day} {hour}:{minute}"
            start_time = datetime.strptime(start_time_str, "%Y-%m-%d %H:%M")
            start_lat = float(self.start_pos_lat.text())
            start_lon = float(self.start_pos_lon.text())
            end_lat = float(self.end_pos_lat.text())
            end_lon = float(self.end_pos_lon.text())
        except ValueError:
            self.maritime_dqn_fuel_cost.setPlainText("Invalid input format!")
            return

        try:
            datetime(year=int(year), month=int(month), day=int(day))
        except ValueError:
            self.maritime_dqn_fuel_cost.setPlainText("Invalid date!")
            return

        # 시작점과 종료점 설정
        self.env.start_pos = self.env.latlon_to_grid(start_lat, start_lon)
        self.env.end_pos = self.env.latlon_to_grid(end_lat, end_lon)

        # DQN 경로 탐색
        state = self.env.reset(start_time=start_time)
        path = [self.env.current_pos]
        done = False
        step = 0
        while not done and step < self.env.max_steps:
            action = self.agent.select_action(state, epsilon=0.0)
            next_state, reward, done, _ = self.env.step(action)
            path.append(self.env.current_pos)
            state = next_state
            step += 1
            if step % 10 == 0:
                self.plot_path(self.dqn_ax, self.dqn_canvas, self.env.grid, path, 
                            self.env.start_pos, self.env.end_pos)
                QApplication.processEvents()
        self.plot_path(self.dqn_ax, self.dqn_canvas, self.env.grid, path, 
                    self.env.start_pos, self.env.end_pos)
        dqn_fuel = self.env.calculate_path_fuel(path, start_time)
        self.maritime_dqn_fuel_cost.setPlainText(f"DQN Fuel Cost: {dqn_fuel:.2f}")

        # 직선 경로 탐색
        straight_path, _ = self.env.calculate_straight_path(start_time=start_time)
        self.plot_path(self.straight_ax, self.straight_canvas, self.env.grid, straight_path, 
                    self.env.start_pos, self.env.end_pos)
        straight_fuel = self.env.calculate_path_fuel(straight_path, start_time)
        self.straight_fuel_cost.setPlainText(f"Straight Fuel Cost: {straight_fuel:.2f}")

        # A* 경로 탐색
        astar_path, _ = self.env.calculate_astar_path(start_time=start_time)
        if astar_path:
            self.plot_path(self.astar_ax, self.astar_canvas, self.env.grid, astar_path, 
                        self.env.start_pos, self.env.end_pos)
            astar_fuel = self.env.calculate_path_fuel(astar_path, start_time)
            self.astar_fuel_cost.setPlainText(f"A* Fuel Cost: {astar_fuel:.2f}")
        else:
            self.astar_fuel_cost.setPlainText("A* Path Not Found")

    def reset(self):
        self.comboBox.setCurrentText("2018")
        self.comboBox_2.setCurrentText("01")
        self.comboBox_3.setCurrentText("01")
        self.comboBox_4.setCurrentText("00")
        self.comboBox_5.setCurrentText("00")
        self.start_pos_lat.setText("37.46036")
        self.start_pos_lon.setText("126.52360")
        self.end_pos_lat.setText("30.62828")
        self.end_pos_lon.setText("122.06400")
        self.dqn_ax.clear()
        self.straight_ax.clear()
        self.astar_ax.clear()
        self.dqn_canvas.draw()
        self.straight_canvas.draw()
        self.astar_canvas.draw()
        self.maritime_dqn_fuel_cost.setPlainText("")
        self.straight_fuel_cost.setPlainText("")
        self.astar_fuel_cost.setPlainText("")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = NavigationUI()
    window.show()
    sys.exit(app.exec_())
