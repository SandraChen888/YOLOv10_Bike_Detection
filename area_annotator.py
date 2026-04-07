import sys
import json
import os
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QListWidget, QMessageBox,
    QFileDialog, QListWidgetItem, QInputDialog
)
from PyQt5.QtCore import Qt, QPoint, QRect
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor


class ImageLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent
        self.start_point = QPoint()
        self.end_point = QPoint()
        self.drawing = False
        self.current_pixmap = None
        self.scale_factor = 1.0
        self.original_image = None

    def set_image(self, cv2_image):
        self.original_image = cv2_image.copy()
        self.update_display()

    def update_display(self, areas=None):
        if self.original_image is None:
            return
        
        display_img = self.original_image.copy()
        
        if areas:
            for area in areas:
                x1, y1, x2, y2 = area['coords']
                color = (255, 200, 0)
                cv2.rectangle(display_img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(display_img, area['name'], (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        rgb_image = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        label_size = self.size()
        scaled_pixmap = QPixmap.fromImage(qt_image).scaled(
            label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.scale_factor = scaled_pixmap.width() / w
        self.current_pixmap = scaled_pixmap
        self.setPixmap(scaled_pixmap)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.current_pixmap:
            self.start_point = event.pos()
            self.end_point = event.pos()
            self.drawing = True

    def mouseMoveEvent(self, event):
        if self.drawing and self.current_pixmap:
            self.end_point = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.drawing:
            self.drawing = False
            self.end_point = event.pos()
            
            x1 = int(min(self.start_point.x(), self.end_point.x()) / self.scale_factor)
            y1 = int(min(self.start_point.y(), self.end_point.y()) / self.scale_factor)
            x2 = int(max(self.start_point.x(), self.end_point.x()) / self.scale_factor)
            y2 = int(max(self.start_point.y(), self.end_point.y()) / self.scale_factor)
            
            if abs(x2 - x1) > 10 and abs(y2 - y1) > 10:
                self.parent_window.on_area_drawn(x1, y1, x2, y2)

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.drawing and self.current_pixmap:
            painter = QPainter(self)
            pen = QPen(QColor(255, 200, 0), 2, Qt.DashLine)
            painter.setPen(pen)
            rect = QRect(self.start_point, self.end_point)
            painter.drawRect(rect)


class AreaAnnotator(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("禁停区域标注工具 - 校园自行车违停系统")
        self.setGeometry(100, 100, 1400, 900)
        
        self.current_image_path = ""
        self.current_image = None
        self.current_scene = ""
        self.areas = []
        self.all_scenes = {}
        
        self.config_file = "scene_config.json"
        self.load_config()
        
        self.init_ui()
    
    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(15, 15, 15, 15)
        
        left_panel = QWidget()
        left_panel.setMaximumWidth(350)
        left_panel.setStyleSheet("""
            QWidget {
                background-color: white;
                border-radius: 10px;
                padding: 15px;
            }
            QPushButton {
                padding: 10px;
                border-radius: 6px;
                font-weight: bold;
                font-size: 13px;
            }
            QLineEdit {
                padding: 8px;
                border: 1px solid #ddd;
                border-radius: 6px;
                font-size: 13px;
            }
            QListWidget {
                border: 1px solid #ddd;
                border-radius: 6px;
                font-size: 13px;
            }
            QLabel {
                font-size: 14px;
                font-weight: bold;
            }
        """)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(15)
        
        title = QLabel("禁停区域标注工具")
        title.setStyleSheet("font-size: 18px; color: #2c3e50; border: none;")
        left_layout.addWidget(title)
        
        scene_label = QLabel("场景管理")
        scene_label.setStyleSheet("color: #2c3e50; margin-top: 10px;")
        left_layout.addWidget(scene_label)
        
        scene_input_layout = QHBoxLayout()
        self.scene_input = QLineEdit()
        self.scene_input.setPlaceholderText("输入场景名称")
        scene_input_layout.addWidget(self.scene_input)
        left_layout.addLayout(scene_input_layout)
        
        scene_btn_layout = QHBoxLayout()
        self.add_scene_btn = QPushButton("添加场景")
        self.add_scene_btn.setStyleSheet("background-color: #4096ff; color: white;")
        self.add_scene_btn.clicked.connect(self.add_scene)
        scene_btn_layout.addWidget(self.add_scene_btn)
        
        self.del_scene_btn = QPushButton("删除场景")
        self.del_scene_btn.setStyleSheet("background-color: #f56c6c; color: white;")
        self.del_scene_btn.clicked.connect(self.delete_scene)
        scene_btn_layout.addWidget(self.del_scene_btn)
        left_layout.addLayout(scene_btn_layout)
        
        self.scene_list = QListWidget()
        self.scene_list.itemClicked.connect(self.on_scene_selected)
        left_layout.addWidget(self.scene_list)
        
        area_label = QLabel("禁停区域")
        area_label.setStyleSheet("color: #2c3e50; margin-top: 10px;")
        left_layout.addWidget(area_label)
        
        area_btn_layout = QHBoxLayout()
        self.add_area_btn = QPushButton("添加区域")
        self.add_area_btn.setStyleSheet("background-color: #67c23a; color: white;")
        self.add_area_btn.clicked.connect(self.add_area_from_input)
        area_btn_layout.addWidget(self.add_area_btn)
        
        self.del_area_btn = QPushButton("删除区域")
        self.del_area_btn.setStyleSheet("background-color: #e6a23c; color: white;")
        self.del_area_btn.clicked.connect(self.delete_area)
        area_btn_layout.addWidget(self.del_area_btn)
        left_layout.addLayout(area_btn_layout)
        
        self.area_list = QListWidget()
        left_layout.addWidget(self.area_list)
        
        img_btn_layout = QHBoxLayout()
        self.load_img_btn = QPushButton("打开图片")
        self.load_img_btn.setStyleSheet("background-color: #909399; color: white;")
        self.load_img_btn.clicked.connect(self.load_image)
        img_btn_layout.addWidget(self.load_img_btn)
        
        self.save_btn = QPushButton("保存配置")
        self.save_btn.setStyleSheet("background-color: #f7ba1e; color: white;")
        self.save_btn.clicked.connect(self.save_config)
        img_btn_layout.addWidget(self.save_btn)
        left_layout.addLayout(img_btn_layout)
        
        export_btn = QPushButton("导出Python配置文件")
        export_btn.setStyleSheet("background-color: #9c27b0; color: white;")
        export_btn.clicked.connect(self.export_python_config)
        left_layout.addWidget(export_btn)
        
        left_layout.addStretch()
        
        tips = QLabel("使用说明:\n1. 添加/选择场景\n2. 打开该场景的图片\n3. 在图片上拖拽画矩形\n4. 输入区域名称并添加\n5. 保存配置")
        tips.setStyleSheet("color: #909399; font-size: 12px; font-weight: normal;")
        left_layout.addWidget(tips)
        
        main_layout.addWidget(left_panel)
        
        right_panel = QWidget()
        right_panel.setStyleSheet("""
            QWidget {
                background-color: white;
                border-radius: 10px;
                padding: 10px;
            }
        """)
        right_layout = QVBoxLayout(right_panel)
        
        self.image_label = ImageLabel(self)
        self.image_label.setStyleSheet("""
            QLabel {
                border: 2px dashed #ddd;
                border-radius: 8px;
                background-color: #f5f5f5;
                color: #999;
                font-size: 16px;
            }
        """)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setText("请打开图片进行标注")
        right_layout.addWidget(self.image_label)
        
        main_layout.addWidget(right_panel, 3)
        
        self.refresh_scene_list()
    
    def load_config(self):
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    self.all_scenes = json.load(f)
            except:
                self.all_scenes = {}
        else:
            self.all_scenes = {}
    
    def save_config(self):
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.all_scenes, f, ensure_ascii=False, indent=2)
            QMessageBox.information(self, "成功", f"配置已保存到 {self.config_file}")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"保存失败: {str(e)}")
    
    def export_python_config(self):
        output_file = "scene_areas.py"
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write("# scene_areas.py - 校园禁停区域配置文件\n")
                f.write("# 由区域标注工具自动生成\n\n")
                f.write("SCENE_AREAS = {\n")
                for scene_name, areas in self.all_scenes.items():
                    f.write(f'    "{scene_name}": {{\n')
                    f.write('        "illegal_areas": {\n')
                    for area in areas:
                        coords = area['coords']
                        f.write(f'            "{area["name"]}": {coords},\n')
                    f.write('        }\n')
                    f.write('    },\n')
                f.write("}\n")
            QMessageBox.information(self, "成功", f"Python配置已导出到 {output_file}")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"导出失败: {str(e)}")
    
    def refresh_scene_list(self):
        self.scene_list.clear()
        for scene_name in self.all_scenes.keys():
            self.scene_list.addItem(scene_name)
    
    def refresh_area_list(self):
        self.area_list.clear()
        for area in self.areas:
            coords = area['coords']
            item_text = f"{area['name']}: [{coords[0]}, {coords[1]}, {coords[2]}, {coords[3]}]"
            self.area_list.addItem(item_text)
    
    def add_scene(self):
        scene_name = self.scene_input.text().strip()
        if not scene_name:
            QMessageBox.warning(self, "警告", "请输入场景名称")
            return
        if scene_name in self.all_scenes:
            QMessageBox.warning(self, "警告", "该场景已存在")
            return
        
        self.all_scenes[scene_name] = []
        self.refresh_scene_list()
        self.scene_input.clear()
        QMessageBox.information(self, "成功", f"场景 '{scene_name}' 已添加")
    
    def delete_scene(self):
        current_item = self.scene_list.currentItem()
        if not current_item:
            QMessageBox.warning(self, "警告", "请先选择要删除的场景")
            return
        
        scene_name = current_item.text()
        reply = QMessageBox.question(self, "确认", f"确定删除场景 '{scene_name}' 吗？",
                                    QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            del self.all_scenes[scene_name]
            self.refresh_scene_list()
            self.areas = []
            self.refresh_area_list()
            self.current_scene = ""
            QMessageBox.information(self, "成功", f"场景 '{scene_name}' 已删除")
    
    def on_scene_selected(self, item):
        scene_name = item.text()
        self.current_scene = scene_name
        self.areas = self.all_scenes.get(scene_name, [])
        self.refresh_area_list()
        
        if self.current_image is not None:
            self.image_label.update_display(self.areas)
    
    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择图片", "", "Images (*.png *.jpg *.jpeg *.bmp)"
        )
        if file_path:
            self.current_image_path = file_path
            self.current_image = cv2.imread(file_path)
            if self.current_image is None:
                QMessageBox.critical(self, "错误", "无法读取图片")
                return
            
            self.image_label.set_image(self.current_image)
            self.image_label.update_display(self.areas)
    
    def on_area_drawn(self, x1, y1, x2, y2):
        if not self.current_scene:
            QMessageBox.warning(self, "警告", "请先选择或添加场景")
            return
        
        name, ok = QInputDialog.getText(self, "区域命名", f"请输入禁停区域名称:\n坐标: [{x1}, {y1}, {x2}, {y2}]")
        if ok and name:
            self.areas.append({
                'name': name,
                'coords': [x1, y1, x2, y2]
            })
            self.all_scenes[self.current_scene] = self.areas
            self.refresh_area_list()
            self.image_label.update_display(self.areas)
    
    def add_area_from_input(self):
        if not self.current_scene:
            QMessageBox.warning(self, "警告", "请先选择或添加场景")
            return
        
        name, ok = QInputDialog.getText(self, "区域命名", "请输入禁停区域名称:")
        if ok and name:
            coords_str, ok2 = QInputDialog.getText(
                self, "区域坐标", 
                "请输入坐标 (格式: x1,y1,x2,y2):",
                text="0,0,100,100"
            )
            if ok2:
                try:
                    coords = [int(x.strip()) for x in coords_str.split(',')]
                    if len(coords) != 4:
                        raise ValueError("需要4个坐标值")
                    self.areas.append({
                        'name': name,
                        'coords': coords
                    })
                    self.all_scenes[self.current_scene] = self.areas
                    self.refresh_area_list()
                    self.image_label.update_display(self.areas)
                except Exception as e:
                    QMessageBox.critical(self, "错误", f"坐标格式错误: {str(e)}")
    
    def delete_area(self):
        current_row = self.area_list.currentRow()
        if current_row < 0:
            QMessageBox.warning(self, "警告", "请先选择要删除的区域")
            return
        
        del self.areas[current_row]
        self.all_scenes[self.current_scene] = self.areas
        self.refresh_area_list()
        self.image_label.update_display(self.areas)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = AreaAnnotator()
    window.show()
    sys.exit(app.exec_())
