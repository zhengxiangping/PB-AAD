# -*- coding: utf-8 -*-

import sys
import os
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QTextEdit,
                             QSplitter, QFileDialog, QComboBox, QSpinBox,
                             QGroupBox, QGridLayout, QSlider, QCheckBox,
                             QMessageBox)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont, QIcon
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.cm as cm


class TrafficCanvas3D(FigureCanvas):
    
    def __init__(self, parent=None, width=8, height=6, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.ax = self.fig.add_subplot(111, projection='3d')
        super(TrafficCanvas3D, self).__init__(self.fig)
        self.setParent(parent)
        
        self.ax.set_xlabel('X (Nodes)', fontsize=10)
        self.ax.set_ylabel('Y (Nodes)', fontsize=10)
        self.ax.set_zlabel('Traffic Flow', fontsize=10)
        self.ax.set_title('Traffic Data 3D Visualization', fontsize=12, pad=20)
        
        self.data = None
        
    def plot_traffic_data(self, data, time_step=0):
        self.ax.clear()
        
        if data is None:
            return
      
        if len(data.shape) == 3:
            # (time, nodes, features)
            traffic_flow = data[time_step, :, 0]
        else:
            traffic_flow = data

        num_nodes = len(traffic_flow)
        grid_size = int(np.ceil(np.sqrt(num_nodes)))
        
        x_data = []
        y_data = []
        z_data = []
        dx = dy = 0.8
        
        for i, flow in enumerate(traffic_flow):
            x = i % grid_size
            y = i // grid_size
            x_data.append(x)
            y_data.append(y)
            z_data.append(0)
        
        dz = traffic_flow
        

        colors = cm.viridis(traffic_flow / traffic_flow.max())
        

        self.ax.bar3d(x_data, y_data, z_data, dx, dy, dz, 
                     color=colors, shade=True, alpha=0.8)
        
        self.ax.set_xlabel('X (Nodes)', fontsize=10)
        self.ax.set_ylabel('Y (Nodes)', fontsize=10)
        self.ax.set_zlabel('Traffic Flow', fontsize=10)
        self.ax.set_title(f'Time Step: {time_step}', fontsize=12, pad=20)
        
        self.draw()
    
    def plot_heatmap_3d(self, data, elevation=0.5):
        self.ax.clear()
        
        if data is None:
            return
        
        num_nodes = data.shape[0]
        grid_size = int(np.ceil(np.sqrt(num_nodes)))
        
        Z = np.zeros((grid_size, grid_size))
        for i in range(num_nodes):
            x = i % grid_size
            y = i // grid_size
            if y < grid_size and x < grid_size:
                Z[y, x] = data[i]
        
        X, Y = np.meshgrid(range(grid_size), range(grid_size))
        
        surf = self.ax.plot_surface(X, Y, Z, cmap='coolwarm', 
                                    alpha=0.8, edgecolor='none')
        
        self.fig.colorbar(surf, ax=self.ax, shrink=0.5, aspect=5)
        
        self.ax.set_xlabel('X (Nodes)', fontsize=10)
        self.ax.set_ylabel('Y (Nodes)', fontsize=10)
        self.ax.set_zlabel('Traffic Flow', fontsize=10)
        self.ax.set_title('Traffic Flow Heatmap', fontsize=12, pad=20)
        
        self.draw()


class MainWindow(QMainWindow):
    
    def __init__(self):
        super().__init__()
        self.data = None
        self.current_time_step = 0
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle('AGP-Traffic 3D Visualization System')
        self.setGeometry(100, 100, 1400, 900)
        

        main_layout = QHBoxLayout(central_widget)
      
        splitter = QSplitter(Qt.Horizontal)
        
        left_panel = self.create_control_panel()
        splitter.addWidget(left_panel)
      
        center_widget = QWidget()
        center_layout = QVBoxLayout(center_widget)
        
        self.canvas = TrafficCanvas3D(self, width=8, height=6)
        center_layout.addWidget(self.canvas)
        
        time_control = self.create_time_control()
        center_layout.addWidget(time_control)
        
        splitter.addWidget(center_widget)
      
        right_panel = self.create_log_panel()
        splitter.addWidget(right_panel)
      
        splitter.setSizes([300, 800, 300])
        
        main_layout.addWidget(splitter)
        
        self.load_sample_data()
        
    def create_control_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        

        title = QLabel('Control Panel')
        title.setFont(QFont('Arial', 14, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
          data_group = QGroupBox('Data Loading')
        data_layout = QVBoxLayout()
        
        btn_load = QPushButton('Load Data File')
        btn_load.clicked.connect(self.load_data_file)
        data_layout.addWidget(btn_load)
        
        btn_sample = QPushButton('Load Sample Data')
        btn_sample.clicked.connect(self.load_sample_data)
        data_layout.addWidget(btn_sample)
        
        data_group.setLayout(data_layout)
        layout.addWidget(data_group)
        
        vis_group = QGroupBox('Visualization Settings')
        vis_layout = QGridLayout()
        
        vis_layout.addWidget(QLabel('Visualization Type:'), 0, 0)
        self.combo_vis_type = QComboBox()
        self.combo_vis_type.addItems(['3D Bar Chart', '3D Heatmap', '3D Surface'])
        self.combo_vis_type.currentIndexChanged.connect(self.update_visualization)
        vis_layout.addWidget(self.combo_vis_type, 0, 1)
        
        vis_layout.addWidget(QLabel('Color Map:'), 1, 0)
        self.combo_colormap = QComboBox()
        self.combo_colormap.addItems(['viridis', 'coolwarm', 'plasma', 'jet'])
        vis_layout.addWidget(self.combo_colormap, 1, 1)
        
        vis_layout.addWidget(QLabel('View Angle:'), 2, 0)
        self.slider_angle = QSlider(Qt.Horizontal)
        self.slider_angle.setRange(0, 360)
        self.slider_angle.setValue(45)
        self.slider_angle.valueChanged.connect(self.rotate_view)
        vis_layout.addWidget(self.slider_angle, 2, 1)
        
        vis_group.setLayout(vis_layout)
        layout.addWidget(vis_group)
        
  
        analysis_group = QGroupBox('Data Analysis')
        analysis_layout = QVBoxLayout()
        
        self.check_show_stats = QCheckBox('Show Statistics')
        self.check_show_stats.setChecked(True)
        analysis_layout.addWidget(self.check_show_stats)
        
        self.check_animation = QCheckBox('Enable Animation')
        self.check_animation.stateChanged.connect(self.toggle_animation)
        analysis_layout.addWidget(self.check_animation)
        
        btn_export = QPushButton('Export Image')
        btn_export.clicked.connect(self.export_image)
        analysis_layout.addWidget(btn_export)
        
        analysis_group.setLayout(analysis_layout)
        layout.addWidget(analysis_group)
        
   
        stats_group = QGroupBox('Statistics')
        stats_layout = QVBoxLayout()
        
        self.label_stats = QLabel('No data loaded')
        self.label_stats.setWordWrap(True)
        self.label_stats.setFont(QFont('Courier', 9))
        stats_layout.addWidget(self.label_stats)
        
        stats_group.setLayout(stats_layout)
        layout.addWidget(stats_group)
        
        layout.addStretch()
        
        return panel
    
 
    def load_sample_data(self):
        """加载示例数据"""
        try:
            # 尝试加载真实数据
            data_path = '../save_test/data_0.npz'
            if os.path.exists(data_path):
                loaded = np.load(data_path)
                self.data = np.squeeze(loaded["array2"])[:100, :, :]  # 限制时间步
                self.log(f"✓ Loaded real data: {self.data.shape}")
            else:
                # 生成示例数据
                self.data = np.random.rand(100, 50, 1) * 100
                self.log(f"✓ Generated sample data: {self.data.shape}")
            
            self.slider_time.setRange(0, self.data.shape[0] - 1)
            self.update_visualization()
            self.update_statistics()
            
        except Exception as e:
            self.log(f"✗ Error loading data: {e}")
    
    def load_data_file(self):
        """从文件加载数据"""
        filename, _ = QFileDialog.getOpenFileName(
            self, 'Open Data File', '', 'NumPy Files (*.npz *.npy);;All Files (*)')
        
        if filename:
            try:
                loaded = np.load(filename)
                if filename.endswith('.npz'):
                    # 尝试获取数据
                    keys = list(loaded.keys())
                    self.data = loaded[keys[0]]
                else:
                    self.data = loaded
                
                self.log(f"✓ Loaded data from {filename}")
                self.log(f"  Shape: {self.data.shape}")
                self.slider_time.setRange(0, self.data.shape[0] - 1)
                self.update_visualization()
                self.update_statistics()
                
            except Exception as e:
                self.log(f"✗ Error loading file: {e}")
    

Time Step: {time_step}
Nodes: {len(current_data)}
Mean: {current_data.mean():.2f}
Std: {current_data.std():.2f}
Min: {current_data.min():.2f}
Max: {current_data.max():.2f}
        """
        self.label_stats.setText(stats_text.strip())
    
    def on_time_change(self, value):
        """时间步改变"""
        self.current_time_step = value
        self.label_time.setText(f'{value} / {self.slider_time.maximum()}')
        self.update_visualization()
        self.update_statistics()
    
    def prev_time_step(self):
        """上一时间步"""
        current = self.slider_time.value()
        if current > 0:
            self.slider_time.setValue(current - 1)
    
    def next_time_step(self):
        """下一时间步"""
        current = self.slider_time.value()
        if current < self.slider_time.maximum():
            self.slider_time.setValue(current + 1)
    
    def toggle_play(self):
        """切换播放/暂停"""
        self.is_playing = not self.is_playing
        if self.is_playing:
            self.btn_play.setText('⏸ Pause')
            self.timer.start(200)  # 200ms 间隔
            self.log("▶ Animation started")
        else:
            self.btn_play.setText('▶ Play')
            self.timer.stop()
            self.log("⏸ Animation paused")
    
    def animate_step(self):
        """动画步进"""
        current = self.slider_time.value()
        if current < self.slider_time.maximum():
            self.slider_time.setValue(current + 1)
        else:
            self.slider_time.setValue(0)  # 循环播放
    
    def toggle_animation(self, state):
        """切换动画"""
        if state == Qt.Checked:
            self.log("✓ Animation enabled")
        else:
            self.log("✗ Animation disabled")
            if self.is_playing:
                self.toggle_play()
    
    def rotate_view(self, angle):
        """旋转视图"""
        if self.canvas.ax:
            self.canvas.ax.view_init(elev=30, azim=angle)
            self.canvas.draw()
    
    def export_image(self):
        """导出图像"""
        filename, _ = QFileDialog.getSaveFileName(
            self, 'Save Image', '', 
            'PNG Files (*.png);;PDF Files (*.pdf);;All Files (*)')
        
        if filename:
            try:
                self.canvas.fig.savefig(filename, dpi=300, bbox_inches='tight')
                self.log(f"✓ Image exported: {filename}")
            except Exception as e:
                self.log(f"✗ Export error: {e}")


def main():
    try:
        app = QApplication(sys.argv)
        app.setStyle('Fusion')  # 使用 Fusion 风格
        
        window = MainWindow()
        window.show()
        
        sys.exit(app.exec_())
    except Exception as e:
        import traceback
        error_msg = f"程序启动失败！\n\n错误信息：\n{str(e)}\n\n详细堆栈：\n{traceback.format_exc()}"
        print("=" * 60)
        print("ERROR: Application failed to start!")
        print("=" * 60)
        print(error_msg)
        print("=" * 60)
        print("\n可能的解决方案：")
        print("1. 安装缺失的依赖：pip install PyQt5 matplotlib numpy")
        print("2. 或运行：python install_gui_deps.py")
        print("\n按任意键退出...")
        input()
        sys.exit(1)


if __name__ == '__main__':
    main()
