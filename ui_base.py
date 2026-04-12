import sys
import os
import cv2
import numpy as np
from datetime import datetime
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QSlider, QComboBox, QTextEdit,
    QFileDialog, QSizePolicy, QFrame, QMessageBox,
    QDialog, QTableWidget, QTableWidgetItem, QHeaderView,
    QSplitter, QGroupBox, QScrollArea
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QImage, QPixmap, QFont, QColor, QPalette
from ultralytics import YOLO
from record_manage import export_current_detection_excel
from illegal_judge import judge_illegal


# ====================== 记录查询对话框 ======================
class QueryDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("记录查询")
        self.setMinimumSize(1100, 700)
        self.current_records = []
        self._init_ui()
    
    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(18)
        layout.setContentsMargins(25, 25, 25, 25)
        
        # 标题和统计
        title_layout = QHBoxLayout()
        
        title_label = QLabel("检测记录查询")
        title_label.setStyleSheet("""
            QLabel {
                font-size: 24px;
                font-weight: bold;
                color: #2c3e50;
            }
        """)
        title_layout.addWidget(title_label)
        
        self.stats_label = QLabel("请选择查询方式")
        self.stats_label.setStyleSheet("""
            QLabel {
                font-size: 16px;
                color: #666;
                padding: 10px 18px;
                background-color: #f0f2f5;
                border-radius: 8px;
            }
        """)
        title_layout.addStretch()
        title_layout.addWidget(self.stats_label)
        
        layout.addLayout(title_layout)
        
        # 查询选项
        query_group = QGroupBox("查询选项")
        query_group.setStyleSheet("""
            QGroupBox {
                font-weight: 600;
                font-size: 17px;
                color: #2c3e50;
                border: 2px solid #e5e9f0;
                border-radius: 10px;
                margin-top: 12px;
                padding-top: 12px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 18px;
                padding: 0 6px;
            }
        """)
        
        query_layout = QVBoxLayout()
        query_group.setLayout(query_layout)
        
        # 查询按钮行
        btn_row1 = QHBoxLayout()
        btn_row1.setSpacing(14)
        
        self.btn_all = QPushButton("查询全部")
        self.btn_all.clicked.connect(lambda: self._do_query("all"))
        self.btn_all.setStyleSheet("""
            QPushButton {
                padding: 12px 28px;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                    stop:0 #4096ff, stop:1 #3088ff);
                color: white;
                border: none;
                border-radius: 10px;
                font-weight: 600;
                font-size: 16px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                    stop:0 #50a6ff, stop:1 #4096ff);
            }
        """)
        btn_row1.addWidget(self.btn_all)
        
        self.btn_time = QPushButton("按时间查询")
        self.btn_time.clicked.connect(lambda: self._do_query("time"))
        self.btn_time.setStyleSheet("""
            QPushButton {
                padding: 12px 28px;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                    stop:0 #722ed1, stop:1 #5217b8);
                color: white;
                border: none;
                border-radius: 10px;
                font-weight: 600;
                font-size: 16px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                    stop:0 #853fe1, stop:1 #6222c4);
            }
        """)
        btn_row1.addWidget(self.btn_time)
        
        self.btn_type = QPushButton("按类型查询")
        self.btn_type.clicked.connect(lambda: self._do_query("type"))
        self.btn_type.setStyleSheet("""
            QPushButton {
                padding: 12px 28px;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                    stop:0 #fa8c16, stop:1 #d46b08);
                color: white;
                border: none;
                border-radius: 10px;
                font-weight: 600;
                font-size: 16px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                    stop:0 #ff9d2a, stop:1 #eb790a);
            }
        """)
        btn_row1.addWidget(self.btn_type)
        
        btn_row2 = QHBoxLayout()
        btn_row2.setSpacing(14)
        
        self.btn_area = QPushButton("按区域查询")
        self.btn_area.clicked.connect(lambda: self._do_query("area"))
        self.btn_area.setStyleSheet("""
            QPushButton {
                padding: 12px 28px;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                    stop:0 #67c23a, stop:1 #529e2d);
                color: white;
                border: none;
                border-radius: 10px;
                font-weight: 600;
                font-size: 16px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                    stop:0 #7dd54a, stop:1 #5daf34);
            }
        """)
        btn_row2.addWidget(self.btn_area)
        
        self.btn_stats = QPushButton("统计报告")
        self.btn_stats.clicked.connect(lambda: self._do_query("stats"))
        self.btn_stats.setStyleSheet("""
            QPushButton {
                padding: 12px 28px;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                    stop:0 #f56c6c, stop:1 #d34848);
                color: white;
                border: none;
                border-radius: 10px;
                font-weight: 600;
                font-size: 16px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                    stop:0 #ff7d7d, stop:1 #e45858);
            }
        """)
        btn_row2.addWidget(self.btn_stats)
        
        query_layout.addLayout(btn_row1)
        query_layout.addSpacing(10)
        query_layout.addLayout(btn_row2)
        
        layout.addWidget(query_group)
        
        # 数据表格
        self.table = QTableWidget()
        self.table.setColumnCount(7)
        self.table.setHorizontalHeaderLabels([
            "ID", "检测时间", "检测区域", "违规类型", "是否违规", "创建时间", "更新时间"
        ])
        
        header = self.table.horizontalHeader()
        header.setStyleSheet("""
            QHeaderView::section {
                background-color: #4096ff;
                color: white;
                padding: 12px;
                border: none;
                font-weight: 600;
                font-size: 15px;
            }
        """)
        header.setSectionResizeMode(QHeaderView.Stretch)
        
        self.table.setStyleSheet("""
            QTableWidget {
                background-color: white;
                border: 1px solid #e5e9f0;
                border-radius: 10px;
                gridline-color: #f0f2f5;
                font-size: 15px;
            }
            QTableWidget::item {
                padding: 10px;
                border-bottom: 1px solid #f0f2f5;
            }
            QTableWidget::item:selected {
                background-color: #e8f4ff;
                color: #2c3e50;
            }
            QTableWidget::alternate-row {
                background-color: #f8f9fa;
            }
        """)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        
        layout.addWidget(self.table)
        
        # 底部按钮
        bottom_layout = QHBoxLayout()
        
        self.export_btn = QPushButton("导出Excel")
        self.export_btn.clicked.connect(self._export_data)
        self.export_btn.setEnabled(False)
        self.export_btn.setStyleSheet("""
            QPushButton {
                padding: 12px 35px;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                    stop:0 #f7ba1e, stop:1 #d59e1a);
                color: white;
                border: none;
                border-radius: 10px;
                font-weight: 600;
                font-size: 16px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                    stop:0 #ffcc33, stop:1 #e6ac1c);
            }
            QPushButton:disabled {
                background: #e0e0e0;
                color: #999;
            }
        """)
        bottom_layout.addWidget(self.export_btn)
        
        bottom_layout.addStretch()
        
        close_btn = QPushButton("关闭")
        close_btn.clicked.connect(self.accept)
        close_btn.setStyleSheet("""
            QPushButton {
                padding: 12px 35px;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                    stop:0 #909399, stop:1 #787b80);
                color: white;
                border: none;
                border-radius: 10px;
                font-weight: 600;
                font-size: 16px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                    stop:0 #a0a3a9, stop:1 #888b90);
            }
        """)
        bottom_layout.addWidget(close_btn)
        
        layout.addLayout(bottom_layout)
    
    def _do_query(self, query_type):
        from record_manage import (
            query_record, query_by_time_range, query_by_violation_type,
            query_by_violation_area, get_detection_statistics
        )
        
        try:
            self.current_records = []
            
            if query_type == "all":
                self.current_records = query_record()
                self.stats_label.setText(f"查询结果：共 {len(self.current_records)} 条记录")
                self._display_records(self.current_records)
                
            elif query_type == "time":
                from PyQt5.QtWidgets import QInputDialog
                start_time, ok1 = QInputDialog.getText(
                    self, "时间查询", "请输入开始时间 (格式: 2024-01-01 00:00:00):",
                    text=datetime.now().strftime("%Y-%m-%d 00:00:00")
                )
                if ok1:
                    end_time, ok2 = QInputDialog.getText(
                        self, "时间查询", "请输入结束时间 (格式: 2024-12-31 23:59:59):",
                        text=datetime.now().strftime("%Y-%m-%d 23:59:59")
                    )
                    if ok2:
                        self.current_records = query_by_time_range(start_time, end_time)
                        self.stats_label.setText(f"时间范围查询：共 {len(self.current_records)} 条记录")
                        self._display_records(self.current_records)
                        
            elif query_type == "type":
                from PyQt5.QtWidgets import QInputDialog
                violation_type, ok = QInputDialog.getItem(
                    self, "类型查询", "请选择违规类型:",
                    ["人行道", "教学楼门口", "消防通道", "绿化带", "其他", "合法"],
                    editable=True
                )
                if ok and violation_type:
                    self.current_records = query_by_violation_type(violation_type)
                    self.stats_label.setText(f"类型查询：共 {len(self.current_records)} 条记录")
                    self._display_records(self.current_records)
                    
            elif query_type == "area":
                from PyQt5.QtWidgets import QInputDialog
                violation_area, ok = QInputDialog.getText(
                    self, "区域查询", "请输入违规区域（如：教学楼A栋、一饭堂门口）:"
                )
                if ok and violation_area:
                    self.current_records = query_by_violation_area(violation_area)
                    self.stats_label.setText(f"区域查询：共 {len(self.current_records)} 条记录")
                    self._display_records(self.current_records)
                    
            elif query_type == "stats":
                stats = get_detection_statistics()
                self._show_stats(stats)
                
        except Exception as e:
            QMessageBox.warning(self, "查询错误", f"查询失败：{str(e)}")
    
    def _display_records(self, records):
        self.table.setRowCount(0)
        self.export_btn.setEnabled(len(records) > 0)
        
        for row_idx, record in enumerate(records):
            self.table.insertRow(row_idx)
            
            self.table.setItem(row_idx, 0, QTableWidgetItem(str(record.id)))
            self.table.setItem(row_idx, 1, QTableWidgetItem(
                record.detect_time.strftime("%Y-%m-%d %H:%M:%S")
            ))
            self.table.setItem(row_idx, 2, QTableWidgetItem(record.violation_area))
            self.table.setItem(row_idx, 3, QTableWidgetItem(record.violation_type))
            
            is_illegal_item = QTableWidgetItem("违规" if record.is_illegal == 1 else "合法")
            if record.is_illegal == 1:
                is_illegal_item.setForeground(QColor("#f56c6c"))
            else:
                is_illegal_item.setForeground(QColor("#52c41a"))
            is_illegal_item.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(row_idx, 4, is_illegal_item)
            
            create_time_str = record.create_time.strftime("%Y-%m-%d %H:%M:%S") if record.create_time else "-"
            self.table.setItem(row_idx, 5, QTableWidgetItem(create_time_str))
            
            update_time_str = record.update_time.strftime("%Y-%m-%d %H:%M:%S") if record.update_time else "-"
            self.table.setItem(row_idx, 6, QTableWidgetItem(update_time_str))
    
    def _show_stats(self, stats):
        stats_text = f"检测统计报告\n\n"
        stats_text += f"总检测次数：{stats.get('total_count', 0)}\n\n"
        
        stats_text += f"合法次数：{stats['legality_statistics']['合法']}\n"
        stats_text += f"违规次数：{stats['legality_statistics']['违规']}\n\n"
        
        type_stats = stats.get('type_statistics', {})
        if type_stats:
            stats_text += "按违规类型统计：\n"
            for vtype, count in sorted(type_stats.items(), key=lambda x: x[1], reverse=True):
                stats_text += f"  - {vtype}: {count} 次\n"
            stats_text += "\n"
        
        area_stats = stats.get('area_statistics', {})
        if area_stats:
            stats_text += "按检测区域统计：\n"
            for varea, count in sorted(area_stats.items(), key=lambda x: x[1], reverse=True):
                stats_text += f"  - {varea}: {count} 次\n"
        
        QMessageBox.information(self, "统计报告", stats_text)
    
    def _export_data(self):
        from record_manage import export_query_results_to_excel
        
        if not self.current_records:
            QMessageBox.information(self, "提示", "没有数据可导出！")
            return
        
        save_path, _ = QFileDialog.getSaveFileName(
            self, "导出查询结果", "查询结果.xlsx", "Excel (*.xlsx)"
        )
        if save_path:
            export_query_results_to_excel(self.current_records, save_path)
            QMessageBox.information(self, "导出成功", f"查询结果已导出至：{save_path}")


# ====================== 数据库查看对话框 ======================
class DatabaseViewDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("数据库记录查看")
        self.setMinimumSize(1200, 700)
        self._init_ui()
        self._load_data()
    
    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(18)
        layout.setContentsMargins(25, 25, 25, 25)
        
        # 标题和统计信息
        title_layout = QHBoxLayout()
        
        title_label = QLabel("检测记录数据库")
        title_label.setStyleSheet("""
            QLabel {
                font-size: 24px;
                font-weight: bold;
                color: #2c3e50;
            }
        """)
        title_layout.addWidget(title_label)
        
        self.stats_label = QLabel()
        self.stats_label.setStyleSheet("""
            QLabel {
                font-size: 16px;
                color: #666;
                padding: 10px 18px;
                background-color: #f0f2f5;
                border-radius: 8px;
            }
        """)
        title_layout.addStretch()
        title_layout.addWidget(self.stats_label)
        
        layout.addLayout(title_layout)
        
        # 筛选区域
        filter_layout = QHBoxLayout()
        
        self.refresh_btn = QPushButton("刷新数据")
        self.refresh_btn.clicked.connect(self._load_data)
        self.refresh_btn.setStyleSheet("""
            QPushButton {
                padding: 10px 24px;
                background-color: #4096ff;
                color: white;
                border: none;
                border-radius: 8px;
                font-weight: 600;
                font-size: 15px;
            }
            QPushButton:hover {
                background-color: #3088ff;
            }
        """)
        filter_layout.addWidget(self.refresh_btn)
        
        filter_layout.addStretch()
        
        self.export_btn = QPushButton("导出Excel")
        self.export_btn.clicked.connect(self._export_data)
        self.export_btn.setStyleSheet("""
            QPushButton {
                padding: 10px 24px;
                background-color: #52c41a;
                color: white;
                border: none;
                border-radius: 8px;
                font-weight: 600;
                font-size: 15px;
            }
            QPushButton:hover {
                background-color: #49ad18;
            }
        """)
        filter_layout.addWidget(self.export_btn)
        
        layout.addLayout(filter_layout)
        
        # 数据表格
        self.table = QTableWidget()
        self.table.setColumnCount(7)
        self.table.setHorizontalHeaderLabels([
            "ID", "检测时间", "检测区域", "违规类型", "是否违规", "创建时间", "更新时间"
        ])
        
        # 设置表头样式
        header = self.table.horizontalHeader()
        header.setStyleSheet("""
            QHeaderView::section {
                background-color: #4096ff;
                color: white;
                padding: 12px;
                border: none;
                font-weight: 600;
                font-size: 15px;
            }
        """)
        header.setSectionResizeMode(QHeaderView.Stretch)
        
        # 设置表格样式
        self.table.setStyleSheet("""
            QTableWidget {
                background-color: white;
                border: 1px solid #e5e9f0;
                border-radius: 10px;
                gridline-color: #f0f2f5;
                font-size: 15px;
            }
            QTableWidget::item {
                padding: 10px;
                border-bottom: 1px solid #f0f2f5;
            }
            QTableWidget::item:selected {
                background-color: #e8f4ff;
                color: #2c3e50;
            }
            QTableWidget::alternate-row {
                background-color: #f8f9fa;
            }
        """)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        
        layout.addWidget(self.table)
        
        # 关闭按钮
        close_btn = QPushButton("关闭")
        close_btn.clicked.connect(self.accept)
        close_btn.setStyleSheet("""
            QPushButton {
                padding: 12px 45px;
                background-color: #909399;
                color: white;
                border: none;
                border-radius: 8px;
                font-weight: 600;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: #82848a;
            }
        """)
        
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        btn_layout.addWidget(close_btn)
        layout.addLayout(btn_layout)
    
    def _load_data(self):
        from record_manage import query_record, get_detection_statistics
        
        try:
            records = query_record()
            stats = get_detection_statistics()
            
            # 更新统计信息
            self.stats_label.setText(
                f"📊 总计: {stats['total_count']} 条 | "
                f"✅ 合法: {stats['legality_statistics']['合法']} | "
                f"⚠️ 违规: {stats['legality_statistics']['违规']}"
            )
            
            # 清空表格
            self.table.setRowCount(0)
            
            # 填充数据
            for row_idx, record in enumerate(records):
                self.table.insertRow(row_idx)
                
                # ID
                self.table.setItem(row_idx, 0, QTableWidgetItem(str(record.id)))
                
                # 检测时间
                self.table.setItem(row_idx, 1, QTableWidgetItem(
                    record.detect_time.strftime("%Y-%m-%d %H:%M:%S")
                ))
                
                # 检测区域
                self.table.setItem(row_idx, 2, QTableWidgetItem(record.violation_area))
                
                # 违规类型
                self.table.setItem(row_idx, 3, QTableWidgetItem(record.violation_type))
                
                # 是否违规
                is_illegal_item = QTableWidgetItem("违规" if record.is_illegal == 1 else "合法")
                if record.is_illegal == 1:
                    is_illegal_item.setForeground(QColor("#f56c6c"))
                else:
                    is_illegal_item.setForeground(QColor("#52c41a"))
                is_illegal_item.setTextAlignment(Qt.AlignCenter)
                self.table.setItem(row_idx, 4, is_illegal_item)
                
                # 创建时间
                create_time_str = record.create_time.strftime("%Y-%m-%d %H:%M:%S") if record.create_time else "-"
                self.table.setItem(row_idx, 5, QTableWidgetItem(create_time_str))
                
                # 更新时间
                update_time_str = record.update_time.strftime("%Y-%m-%d %H:%M:%S") if record.update_time else "-"
                self.table.setItem(row_idx, 6, QTableWidgetItem(update_time_str))
                
        except Exception as e:
            QMessageBox.warning(self, "加载失败", f"无法加载数据库记录：{str(e)}")
    
    def _export_data(self):
        from record_manage import query_record, export_query_results_to_excel
        
        try:
            records = query_record()
            if not records:
                QMessageBox.information(self, "提示", "没有数据可导出！")
                return
            
            save_path, _ = QFileDialog.getSaveFileName(
                self, "导出数据库记录", "数据库记录.xlsx", "Excel (*.xlsx)"
            )
            if save_path:
                export_query_results_to_excel(records, save_path)
                QMessageBox.information(self, "导出成功", f"数据库记录已导出至：{save_path}")
        except Exception as e:
            QMessageBox.warning(self, "导出失败", f"导出失败：{str(e)}")


# ---------------------- 检测线程（功能不变） ----------------------
class DetectionThread(QThread):
    update_frame = pyqtSignal(np.ndarray)  # 传递检测后的帧
    update_log = pyqtSignal(str)  # 传递日志信息
    update_info = pyqtSignal(dict)  # 传递检测信息
    show_video_duration_warning = pyqtSignal(float)  # 传递视频时长警告

    def __init__(self, model_path, source_type, source_path, conf_thres, iou_thres, camera_index=0, scene="教学楼A栋"):
        super().__init__()
        self.model = YOLO(model_path)  # 加载YOLOv10模型
        self.source_type = source_type  # image/video/camera
        self.source_path = source_path
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.camera_index = camera_index  # 摄像头索引
        self.scene = scene  # 当前场景
        self.is_running = True
        self.current_frame = None  # 保存当前帧用于导出
        self.current_detections = []  # 保存当前检测结果数据（用于Excel导出）
        self.batch_results = []  # 保存批量检测结果

    def run(self):
        try:
            # 初始化数据源
            if self.source_type == "camera":
                cap = cv2.VideoCapture(self.camera_index)  # 使用指定的摄像头索引
                if not cap.isOpened():
                    self.update_log.emit(f"❌ 摄像头 {self.camera_index} 打开失败！")
                    return
                self.update_log.emit(f"✅ 成功打开摄像头 {self.camera_index}")
            elif self.source_type == "video":
                cap = cv2.VideoCapture(self.source_path)
                if not cap.isOpened():
                    self.update_log.emit(f"❌ 视频文件打开失败：{self.source_path}")
                    return
                
                # 检查视频时长是否超过5分钟（300秒）
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if fps > 0:
                    duration = frame_count / fps  # 视频时长（秒）
                    if duration > 300:  # 5分钟 = 300秒
                        self.update_log.emit(f"⚠️ 视频时长超过5分钟（{duration:.1f}秒），建议截取片段后再检测")
                        # 发送信号显示弹窗
                        self.show_video_duration_warning.emit(duration)
                        cap.release()
                        return
                    else:
                        self.update_log.emit(f"✅ 视频时长：{duration:.1f}秒，符合要求（≤5分钟）")
            elif self.source_type == "batch_image":
                # 批量处理多张图片
                import os
                total_images = len(self.source_path)
                self.update_log.emit(f"✅ 开始批量检测，共 {total_images} 张图片")
                
                success_count = 0
                fail_count = 0
                
                for i, image_path in enumerate(self.source_path, 1):
                    self.update_log.emit(f"🔍 检测第 {i}/{total_images} 张：{os.path.basename(image_path)}")
                    
                    img = cv2.imread(image_path)
                    if img is None:
                        self.update_log.emit(f"❌ 图片打开失败：{image_path}")
                        fail_count += 1
                        continue
                    
                    # 检测单张图片
                    results = self.model(img, conf=self.conf_thres, iou=self.iou_thres)
                    
                    # 转换检测结果格式
                    detect_res = []
                    for box in results[0].boxes:
                        xyxy = box.xyxy[0].cpu().numpy()
                        detect_res.append([
                            int(box.cls.item()),  # 类别ID
                            float(box.conf.item()),  # 置信度
                            int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])  # 坐标
                        ])
                    
                    # 违规判断
                    illegal_res, det_img = judge_illegal(detect_res, img, overlap_thresh=0.3, scene=self.scene)
                    
                    # 保存检测结果
                    self.batch_results.append({
                        'image_path': image_path,
                        'detected_image': det_img,
                        'illegal_results': illegal_res,
                        'bicycle_count': len(detect_res)
                    })
                    
                    # 发送信号更新预览
                    self.update_frame.emit(det_img)
                    
                    # 保存检测记录到数据库
                    try:
                        from record_manage import save_detection_record
                        import os
                        from datetime import datetime
                        
                        # 保存截图
                        screenshot_path = f"detection_screenshots/batch_{datetime.now().strftime('%Y%m%d%H%M%S')}_{i}.jpg"
                        os.makedirs(os.path.dirname(screenshot_path), exist_ok=True)
                        cv2.imwrite(screenshot_path, det_img)
                        
                        # 保存每条检测记录
                        for detection in illegal_res:
                            illegal_type = detection.get("违规类型", "合法")
                            is_illegal = 1 if illegal_type != "合法" else 0
                            save_detection_record(
                                screenshot_path=screenshot_path,
                                violation_area=self.scene,
                                violation_type=illegal_type if is_illegal else "合法",
                                is_illegal=is_illegal
                            )
                        
                        success_count += 1
                        self.update_log.emit(f"✅ 第 {i} 张图片检测完成")
                    except Exception as e:
                        self.update_log.emit(f"⚠️ 保存数据库失败：{str(e)}")
                        fail_count += 1
                
                self.update_log.emit(f"📊 批量检测完成：成功 {success_count} 张，失败 {fail_count} 张")
                return
            else:  # 单张图片
                img = cv2.imread(self.source_path)
                if img is None:
                    self.update_log.emit(f"❌ 图片文件打开失败：{self.source_path}")
                    return
                self._detect_single_image(img)
                return

            # 视频/摄像头逐帧检测
            while self.is_running and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    self.update_log.emit("📹 视频/摄像头流结束")
                    break

                # YOLOv10检测
                results = self.model(frame, conf=self.conf_thres, iou=self.iou_thres)
                
                # 转换检测结果格式为judge_illegal所需格式
                detect_res = []
                for box in results[0].boxes:
                    xyxy = box.xyxy[0].cpu().numpy()
                    detect_res.append([
                        int(box.cls.item()),  # 类别ID
                        float(box.conf.item()),  # 置信度
                        int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])  # 坐标
                    ])
                
                # 违规判断
                illegal_res, det_frame = judge_illegal(detect_res, frame, overlap_thresh=0.3, scene=self.scene)
                self.current_frame = det_frame

                # 保存检测结果数据（用于Excel导出）
                self.current_detections = []
                for i, box in enumerate(results[0].boxes):
                    xyxy = box.xyxy[0].cpu().numpy()
                    detection_info = {
                        "x1": float(xyxy[0]),
                        "y1": float(xyxy[1]),
                        "x2": float(xyxy[2]),
                        "y2": float(xyxy[3]),
                        "score": float(box.conf.item())
                    }
                    # 添加违规信息
                    if i < len(illegal_res):
                        detection_info["illegal_type"] = illegal_res[i].get("违规类型", "")
                        detection_info["overlap"] = illegal_res[i].get("重叠占比", 0.0)
                    else:
                        detection_info["illegal_type"] = "合法"
                        detection_info["overlap"] = 0.0
                    self.current_detections.append(detection_info)

                # 统计检测信息
                bike_count = len(results[0].boxes)
                illegal_count = len(illegal_res)
                infer_time = round(results[0].speed.get("inference", 0), 2)
                conf_list = [round(box.conf.item(), 2) for box in results[0].boxes]

                # 发送信号更新界面
                self.update_frame.emit(det_frame)
                self.update_info.emit({
                    "count": bike_count,
                    "illegal_count": illegal_count,
                    "time": infer_time,
                    "conf": conf_list
                })
                
                # 保存检测记录到数据库（每10帧保存一次，避免频繁操作）
                if hasattr(self, 'frame_count'):
                    self.frame_count += 1
                else:
                    self.frame_count = 0
                
                if self.frame_count % 10 == 0:  # 每10帧保存一次
                    try:
                        from record_manage import save_detection_record
                        import os
                        
                        # 保存截图
                        screenshot_path = f"detection_screenshots/video_{datetime.now().strftime('%Y%m%d%H%M%S')}_{self.frame_count}.jpg"
                        os.makedirs(os.path.dirname(screenshot_path), exist_ok=True)
                        cv2.imwrite(screenshot_path, det_frame)
                        
                        # 保存每条检测记录
                        for detection in self.current_detections:
                            illegal_type = detection.get("illegal_type", "合法")
                            is_illegal = 1 if illegal_type and illegal_type != "合法" else 0
                            save_detection_record(
                                screenshot_path=screenshot_path,
                                violation_area=self.scene,  # 使用当前场景
                                violation_type=illegal_type if is_illegal else "合法",
                                is_illegal=is_illegal
                            )
                        
                        if self.frame_count == 0:
                            self.update_log.emit(f"✅ 视频检测记录已开始保存到数据库")
                    except Exception as e:
                        if self.frame_count == 0:
                            self.update_log.emit(f"⚠️ 保存数据库失败：{str(e)}")

            cap.release()
        except Exception as e:
            self.update_log.emit(f"❌ 检测出错：{str(e)}")

    def _detect_single_image(self, img):
        """单张图片检测逻辑"""
        results = self.model(img, conf=self.conf_thres, iou=self.iou_thres)
        
        # 转换检测结果格式为judge_illegal所需格式
        detect_res = []
        for box in results[0].boxes:
            xyxy = box.xyxy[0].cpu().numpy()
            detect_res.append([
                int(box.cls.item()),  # 类别ID
                float(box.conf.item()),  # 置信度
                int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])  # 坐标
            ])
        
        # 违规判断
        illegal_res, det_img = judge_illegal(detect_res, img, overlap_thresh=0.3, scene=self.scene)
        self.current_frame = det_img

        # 保存检测结果数据（用于Excel导出）
        self.current_detections = []
        for i, box in enumerate(results[0].boxes):
            xyxy = box.xyxy[0].cpu().numpy()
            detection_info = {
                "x1": float(xyxy[0]),
                "y1": float(xyxy[1]),
                "x2": float(xyxy[2]),
                "y2": float(xyxy[3]),
                "score": float(box.conf.item())
            }
            # 添加违规信息
            if i < len(illegal_res):
                detection_info["illegal_type"] = illegal_res[i].get("违规类型", "")
                detection_info["overlap"] = illegal_res[i].get("重叠占比", 0.0)
            else:
                detection_info["illegal_type"] = "合法"
                detection_info["overlap"] = 0.0
            self.current_detections.append(detection_info)

        bike_count = len(results[0].boxes)
        illegal_count = len(illegal_res)
        infer_time = round(results[0].speed.get("inference", 0), 2)
        conf_list = [round(box.conf.item(), 2) for box in results[0].boxes]

        self.update_frame.emit(det_img)
        self.update_info.emit({
            "count": bike_count,
            "illegal_count": illegal_count,
            "time": infer_time,
            "conf": conf_list
        })
        # 保存检测记录到数据库
        try:
            from record_manage import save_detection_record
            import os
            
            # 保存截图
            screenshot_path = f"detection_screenshots/detect_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
            os.makedirs(os.path.dirname(screenshot_path), exist_ok=True)
            cv2.imwrite(screenshot_path, det_img)
            
            # 保存每条检测记录
            for detection in self.current_detections:
                illegal_type = detection.get("illegal_type", "合法")
                is_illegal = 1 if illegal_type and illegal_type != "合法" else 0
                save_detection_record(
                    screenshot_path=screenshot_path,
                    violation_area=self.scene,  # 使用当前场景
                    violation_type=illegal_type if is_illegal else "合法",
                    is_illegal=is_illegal
                )
            
            self.update_log.emit(f"✅ 检测记录已保存到数据库")
        except Exception as e:
            self.update_log.emit(f"⚠️ 保存数据库失败：{str(e)}")
        
        self.update_log.emit(f"✅ 图片检测完成，识别到 {bike_count} 辆自行车，其中 {illegal_count} 辆违规停放")

    def stop(self):
        """停止检测"""
        self.is_running = False


# ---------------------- 主界面类（重点美化样式） ----------------------
class BikeDetectionUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLOv10 自行车检测系统（毕业设计）")
        self.setGeometry(80, 50, 1500, 950)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f0f2f5;
                font-family: "Microsoft YaHei", Arial, sans-serif;
            }
            QWidget {
                font-size: 18px;
                color: #333;
            }
            QLabel {
                color: #444;
            }
            QSlider::groove:horizontal {
                border: none;
                height: 10px;
                background: #e5e9f0;
                border-radius: 5px;
            }
            QSlider::handle:horizontal {
                background: #4096ff;
                border: none;
                width: 24px;
                height: 24px;
                margin: -6px 0;
                border-radius: 12px;
            }
            QSlider::handle:horizontal:hover {
                background: #3088ff;
            }
            QComboBox {
                padding: 12px 16px;
                border: 1px solid #dcdfe6;
                border-radius: 8px;
                background-color: white;
                selection-background-color: #e8f4ff;
                font-size: 17px;
            }
            QComboBox:hover {
                border-color: #c0c4cc;
            }
            QComboBox::drop-down {
                border: none;
            }
            QTextEdit {
                border: 1px solid #dcdfe6;
                border-radius: 8px;
                padding: 12px;
                background-color: white;
                font-size: 17px;
                selection-color: #000000;
                selection-background-color: #b3d7ff;
            }
            QTextEdit:focus {
                border-color: #4096ff;
                outline: none;
            }
        """)

        self.detection_thread = None
        self.model_path = "runs/detect/train/weights/best.pt"  # 使用训练好的模型
        self.source_path = ""  # 数据源路径
        self.batch_results = []  # 保存批量检测结果
        self.current_batch_index = 0  # 当前批量检测结果索引

        # 初始化界面
        self._init_ui()

    def _init_ui(self):
        """初始化界面布局（美化版）"""
        # 中心部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(20, 20, 20, 20)

        # ========== 左侧功能区（卡片式设计） ==========
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        # 左侧卡片样式
        left_widget.setStyleSheet("""
            QWidget {
                background-color: white;
                border-radius: 12px;
                padding: 20px;
                box-shadow: 0 2px 12px rgba(0, 0, 0, 0.08);
            }
        """)
        left_widget.setMaximumWidth(500)
        main_layout.addWidget(left_widget)

        # 1. 标题栏
        title_label = QLabel("YOLOv10 检测控制台")
        title_label.setStyleSheet("""
            QLabel {
                font-size: 22px;
                font-weight: bold;
                color: #2c3e50;
                margin-bottom: 30px;
                padding-bottom: 12px;
                border-bottom: 1px solid #f0f2f5;
            }
        """)
        left_layout.addWidget(title_label)

        # 2. 数据源选择
        source_label = QLabel("📁 数据源选择")
        source_label.setStyleSheet("""
            QLabel {
                font-size: 18px;
                font-weight: 600;
                color: #2c3e50;
                margin-bottom: 12px;
            }
        """)
        left_layout.addWidget(source_label)

        self.source_combo = QComboBox()
        self.source_combo.addItems(["本地图片", "批量图片", "本地视频", "摄像头实时"])
        self.source_combo.setStyleSheet("""
            QComboBox {
                padding: 14px 18px;
                border: 1px solid #dcdfe6;
                border-radius: 8px;
                background-color: white;
                selection-background-color: #e8f4ff;
                font-size: 19px;
            }
        """)
        left_layout.addWidget(self.source_combo)
        left_layout.addSpacing(10)

        # 摄像头索引选择
        self.camera_index_layout = QHBoxLayout()
        camera_index_label = QLabel("摄像头索引：")
        camera_index_label.setStyleSheet("color: #666; font-size: 17px;")
        self.camera_index_combo = QComboBox()
        self.camera_index_combo.addItems(["0 (默认)", "1", "2", "3"])
        self.camera_index_combo.setStyleSheet("""
            QComboBox {
                padding: 14px 18px;
                border: 1px solid #dcdfe6;
                border-radius: 8px;
                background-color: white;
                selection-background-color: #e8f4ff;
                font-size: 19px;
            }
        """)
        self.camera_index_layout.addWidget(camera_index_label)
        self.camera_index_layout.addWidget(self.camera_index_combo)
        left_layout.addLayout(self.camera_index_layout)
        left_layout.addSpacing(10)

        self.select_source_btn = QPushButton("选择文件")
        self.select_source_btn.clicked.connect(self._select_source)
        self.select_source_btn.setStyleSheet("""
            QPushButton {
                padding: 12px;
                background-color: #4096ff;
                color: white;
                border: none;
                border-radius: 10px;
                font-weight: 600;
                font-size: 15px;
            }
            QPushButton:hover {
                background-color: #3088ff;
            }
            QPushButton:pressed {
                background-color: #2078ee;
            }
        """)
        left_layout.addWidget(self.select_source_btn)
        left_layout.addSpacing(30)

        # 4. 场景选择
        scene_label = QLabel("📍 场景选择")
        scene_label.setStyleSheet("""
            QLabel {
                font-size: 18px;
                font-weight: 600;
                color: #2c3e50;
                margin-bottom: 12px;
            }
        """)
        left_layout.addWidget(scene_label)

        self.scene_combo = QComboBox()
        self.scene_combo.addItems([
            "教学楼A栋",
            "教学楼B栋",
            "教学楼C栋",
            "教学楼D栋",
            "教学楼E栋",
            "教学楼F栋",
            "一饭堂门口"
        ])
        self.scene_combo.setStyleSheet("""
            QComboBox {
                padding: 14px 18px;
                border: 1px solid #dcdfe6;
                border-radius: 8px;
                background-color: white;
                selection-background-color: #e8f4ff;
                font-size: 19px;
            }
        """)
        left_layout.addWidget(self.scene_combo)
        left_layout.addSpacing(30)

        # 3. 检测参数调节
        param_label = QLabel("⚙️ 检测参数")
        param_label.setStyleSheet("""
            QLabel {
                font-size: 18px;
                font-weight: 600;
                color: #2c3e50;
                margin-bottom: 12px;
            }
        """)
        left_layout.addWidget(param_label)

        # 置信度滑块
        conf_layout = QHBoxLayout()
        conf_layout.setSpacing(12)
        conf_label = QLabel("置信度：")
        conf_label.setStyleSheet("color: #666; font-size: 17px;")
        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setRange(0, 100)
        self.conf_slider.setValue(50)  # 默认0.5
        self.conf_label = QLabel("0.5")
        self.conf_label.setStyleSheet("""
            QLabel {
                color: #4096ff;
                font-weight: 600;
                font-size: 17px;
                min-width: 50px;
                text-align: center;
            }
        """)
        self.conf_slider.valueChanged.connect(lambda v: self.conf_label.setText(str(v / 100)))
        conf_layout.addWidget(conf_label)
        conf_layout.addWidget(self.conf_slider)
        conf_layout.addWidget(self.conf_label)
        left_layout.addLayout(conf_layout)
        left_layout.addSpacing(12)

        # IoU滑块
        iou_layout = QHBoxLayout()
        iou_layout.setSpacing(12)
        iou_label = QLabel("IoU阈值：")
        iou_label.setStyleSheet("color: #666; font-size: 17px;")
        self.iou_slider = QSlider(Qt.Horizontal)
        self.iou_slider.setRange(0, 100)
        self.iou_slider.setValue(45)  # 默认0.45
        self.iou_label = QLabel("0.45")
        self.iou_label.setStyleSheet("""
            QLabel {
                color: #4096ff;
                font-weight: 600;
                font-size: 17px;
                min-width: 50px;
                text-align: center;
            }
        """)
        self.iou_slider.valueChanged.connect(lambda v: self.iou_label.setText(str(v / 100)))
        iou_layout.addWidget(iou_label)
        iou_layout.addWidget(self.iou_slider)
        iou_layout.addWidget(self.iou_label)
        left_layout.addLayout(iou_layout)
        left_layout.addSpacing(30)

        # 4. 操作按钮（两行布局）
        btn_layout = QVBoxLayout()
        btn_layout.setSpacing(0)

        # 创建统一的按钮样式
        button_style = """
            QPushButton {
                padding: 18px 28px;
                border: none;
                border-radius: 12px;
                font-weight: 600;
                font-size: 18px;
            }
            QPushButton:hover {
                transform: translateY(-2px);
            }
            QPushButton:pressed {
                transform: translateY(0px);
            }
        """

        self.start_btn = QPushButton("开始检测")
        self.start_btn.clicked.connect(self._start_detection)
        self.start_btn.setStyleSheet(button_style + """
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                    stop:0 #67c23a, stop:1 #529e2d);
                color: white;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                    stop:0 #7dd54a, stop:1 #5daf34);
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                    stop:0 #529e2d, stop:1 #468c27);
            }
            QPushButton:disabled {
                background: #b3e19d;
                color: #f5f5f5;
            }
        """)

        self.stop_btn = QPushButton("停止检测")
        self.stop_btn.clicked.connect(self._stop_detection)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet(button_style + """
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                    stop:0 #f56c6c, stop:1 #d34848);
                color: white;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                    stop:0 #ff7d7d, stop:1 #e45858);
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                    stop:0 #d34848, stop:1 #c13838);
            }
            QPushButton:disabled {
                background: #f9b4b4;
                color: #f5f5f5;
            }
        """)

        self.export_btn = QPushButton("导出结果")
        self.export_btn.clicked.connect(self._export_result)
        self.export_btn.setStyleSheet(button_style + """
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                    stop:0 #f7ba1e, stop:1 #d59e1a);
                color: white;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                    stop:0 #ffcc33, stop:1 #e6ac1c);
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                    stop:0 #d59e1a, stop:1 #c48d18);
            }
        """)

        # 添加记录查询按钮
        self.query_btn = QPushButton("记录查询")
        self.query_btn.clicked.connect(self._open_query_dialog)
        self.query_btn.setStyleSheet(button_style + """
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                    stop:0 #722ed1, stop:1 #5217b8);
                color: white;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                    stop:0 #853fe1, stop:1 #6222c4);
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                    stop:0 #5217b8, stop:1 #4210a8);
            }
        """)

        # 添加数据库查看按钮
        self.db_view_btn = QPushButton("数据库查看")
        self.db_view_btn.clicked.connect(self._open_database_view)
        self.db_view_btn.setStyleSheet(button_style + """
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                    stop:0 #fa8c16, stop:1 #d46b08);
                color: white;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                    stop:0 #ff9d2a, stop:1 #eb790a);
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                    stop:0 #d46b08, stop:1 #c35a07);
            }
        """)

        # 第一行按钮：开始检测、停止检测
        btn_row1 = QHBoxLayout()
        btn_row1.setSpacing(12)
        btn_row1.addWidget(self.start_btn)
        btn_row1.addWidget(self.stop_btn)
        
        # 第二行按钮：导出结果、记录查询、数据库查看
        btn_row2 = QHBoxLayout()
        btn_row2.setSpacing(12)
        btn_row2.addWidget(self.export_btn)
        btn_row2.addWidget(self.query_btn)
        btn_row2.addWidget(self.db_view_btn)
        
        # 添加到主布局
        btn_layout.addLayout(btn_row1)
        btn_layout.addSpacing(10)
        btn_layout.addLayout(btn_row2)
        left_layout.addLayout(btn_layout)
        left_layout.addSpacing(25)

        # 5. 模型信息（卡片式）
        model_card = QWidget()
        model_card_layout = QVBoxLayout(model_card)
        model_card.setStyleSheet("""
            QWidget {
                background-color: #f8f9fa;
                border-radius: 10px;
                padding: 18px;
            }
        """)

        model_label = QLabel("📊 模型信息")
        model_label.setStyleSheet("""
            QLabel {
                font-size: 18px;
                font-weight: 600;
                color: #2c3e50;
                margin-bottom: 10px;
            }
        """)
        model_card_layout.addWidget(model_label)

        self.model_info_label = QLabel(f"""当前模型：{self.model_path}
检测类别：自行车
运行设备：CPU/GPU（自动适配）""")
        self.model_info_label.setStyleSheet("""
            QLabel {
                font-size: 15px;
                color: #666;
                line-height: 1.6;
            }
        """)
        model_card_layout.addWidget(self.model_info_label)
        left_layout.addWidget(model_card)

        # 填充空白
        left_layout.addStretch()

        # ========== 右侧结果展示区（卡片式） ==========
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_widget.setStyleSheet("""
            QWidget {
                background-color: white;
                border-radius: 12px;
                padding: 20px;
                box-shadow: 0 2px 12px rgba(0, 0, 0, 0.08);
            }
        """)
        main_layout.addWidget(right_widget, 3)

        # 1. 检测结果预览
        result_title_layout = QHBoxLayout()
        result_label = QLabel("检测结果预览")
        result_label.setStyleSheet("""
            QLabel {
                font-size: 20px;
                font-weight: 600;
                color: #2c3e50;
            }
        """)
        result_title_layout.addWidget(result_label)
        result_title_layout.addStretch()
        right_layout.addLayout(result_title_layout)
        right_layout.addSpacing(12)

        # 预览图片导航按钮
        self.nav_layout = QHBoxLayout()
        self.prev_btn = QPushButton("上一张")
        self.prev_btn.setEnabled(False)
        self.prev_btn.setStyleSheet("""
            QPushButton {
                padding: 8px 16px;
                background-color: #4096ff;
                color: white;
                border: none;
                border-radius: 6px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #3088ff;
            }
            QPushButton:disabled {
                background-color: #c6e2ff;
                color: #a0cfff;
            }
        """)
        self.next_btn = QPushButton("下一张")
        self.next_btn.setEnabled(False)
        self.next_btn.setStyleSheet("""
            QPushButton {
                padding: 8px 16px;
                background-color: #4096ff;
                color: white;
                border: none;
                border-radius: 6px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #3088ff;
            }
            QPushButton:disabled {
                background-color: #c6e2ff;
                color: #a0cfff;
            }
        """)
        self.image_index_label = QLabel("0/0")
        self.image_index_label.setStyleSheet("""
            QLabel {
                font-size: 14px;
                color: #666;
                margin: 0 10px;
            }
        """)
        self.nav_layout.addStretch()
        self.nav_layout.addWidget(self.prev_btn)
        self.nav_layout.addWidget(self.image_index_label)
        self.nav_layout.addWidget(self.next_btn)
        right_layout.addLayout(self.nav_layout)
        right_layout.addSpacing(8)

        # 连接导航按钮的点击事件
        self.prev_btn.clicked.connect(self._prev_batch_image)
        self.next_btn.clicked.connect(self._next_batch_image)

        self.result_display = QLabel()
        self.result_display.setStyleSheet("""
            QLabel {
                border: 1px solid #e5e9f0;
                border-radius: 10px;
                background-color: #f8f9fa;
                color: #909399;
                font-size: 17px;
            }
        """)
        self.result_display.setAlignment(Qt.AlignCenter)
        self.result_display.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # 默认显示提示图
        self.result_display.setText("请选择数据源并点击开始检测")
        right_layout.addWidget(self.result_display, 6)
        right_layout.addSpacing(18)

        # 2. 检测信息面板
        info_label = QLabel("📋 检测信息")
        info_label.setStyleSheet("""
            QLabel {
                font-size: 18px;
                font-weight: 600;
                color: #2c3e50;
                margin-bottom: 10px;
            }
        """)
        right_layout.addWidget(info_label)

        self.info_panel = QTextEdit()
        self.info_panel.setReadOnly(True)
        self.info_panel.setText("自行车数量：0 辆\n单帧检测耗时：0 ms\n置信度分布：[]")
        right_layout.addWidget(self.info_panel, 1)
        right_layout.addSpacing(12)

        # 3. 日志输出
        log_label = QLabel("📝 系统日志")
        log_label.setStyleSheet("""
            QLabel {
                font-size: 18px;
                font-weight: 600;
                color: #2c3e50;
                margin-bottom: 10px;
            }
        """)
        right_layout.addWidget(log_label)

        self.log_display = QTextEdit()
        self.log_display.setReadOnly(True)
        self.log_display.setText("系统已就绪，等待检测...")
        right_layout.addWidget(self.log_display, 1)

    def _select_source(self):
        """选择图片/视频文件"""
        source_type = self.source_combo.currentText()
        if source_type == "本地图片":
            file_path, _ = QFileDialog.getOpenFileName(
                self, "选择图片", "", "Images (*.png *.jpg *.jpeg *.bmp)"
            )
        elif source_type == "批量图片":
            file_paths, _ = QFileDialog.getOpenFileNames(
                self, "选择多张图片", "", "Images (*.png *.jpg *.jpeg *.bmp)"
            )
            if file_paths:
                self.source_path = file_paths
                self.log_display.append(f"✅ 已选择 {len(file_paths)} 张图片")
            else:
                self.log_display.append("❌ 未选择任何文件")
            return
        elif source_type == "本地视频":
            file_path, _ = QFileDialog.getOpenFileName(
                self, "选择视频", "", "Videos (*.mp4 *.avi *.mov *.mkv)"
            )
        else:  # 摄像头
            file_path = "camera"
            self.log_display.append("✅ 已选择摄像头作为数据源")
            self.source_path = file_path
            return

        if file_path:
            self.source_path = file_path
            self.log_display.append(f"✅ 已选择数据源：{file_path}")
        else:
            self.log_display.append("❌ 未选择任何文件")

    def _start_detection(self):
        """启动检测"""
        # 检查数据源
        if self.source_combo.currentText() != "摄像头实时" and not self.source_path:
            self.log_display.append("❌ 请先选择图片/视频文件！")
            return

        # 获取参数
        conf_thres = self.conf_slider.value() / 100
        iou_thres = self.iou_slider.value() / 100
        source_type_map = {"本地图片": "image", "批量图片": "batch_image", "本地视频": "video", "摄像头实时": "camera"}
        source_type = source_type_map[self.source_combo.currentText()]

        # 获取摄像头索引
        camera_index = 0
        if source_type == "camera":
            # 从下拉框获取摄像头索引
            camera_index_text = self.camera_index_combo.currentText()
            camera_index = int(camera_index_text.split()[0])

        # 获取当前场景
        current_scene = self.scene_combo.currentText()

        # 启动检测线程
        self.detection_thread = DetectionThread(
            model_path=self.model_path,
            source_type=source_type,
            source_path=self.source_path,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            camera_index=camera_index,
            scene=current_scene
        )

        # 绑定信号
        self.detection_thread.update_frame.connect(self._update_frame)
        self.detection_thread.update_log.connect(self._update_log)
        self.detection_thread.update_info.connect(self._update_info)
        self.detection_thread.show_video_duration_warning.connect(self._show_video_duration_warning)
        self.detection_thread.finished.connect(self._reset_btn_state)

        # 启动线程
        self.detection_thread.start()
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.log_display.append("✅ 开始检测...")

    def _stop_detection(self):
        """停止检测"""
        if self.detection_thread and self.detection_thread.isRunning():
            self.detection_thread.stop()
            self.log_display.append("⏹️ 已停止检测")

    def _reset_btn_state(self):
        """重置按钮状态"""
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        
        # 批量检测完成后更新结果
        if self.detection_thread and self.detection_thread.source_type == "batch_image":
            self.batch_results = self.detection_thread.batch_results
            if self.batch_results:
                self.current_batch_index = 0
                self._display_batch_image()
                self.log_display.append(f"✅ 批量检测结果已加载，共 {len(self.batch_results)} 张图片")
    
    def _prev_batch_image(self):
        """显示上一张批量检测图片"""
        if self.batch_results and self.current_batch_index > 0:
            self.current_batch_index -= 1
            self._display_batch_image()
    
    def _next_batch_image(self):
        """显示下一张批量检测图片"""
        if self.batch_results and self.current_batch_index < len(self.batch_results) - 1:
            self.current_batch_index += 1
            self._display_batch_image()
    
    def _display_batch_image(self):
        """显示当前批量检测图片"""
        if not self.batch_results:
            return
        
        # 更新索引标签
        self.image_index_label.setText(f"{self.current_batch_index + 1}/{len(self.batch_results)}")
        
        # 更新导航按钮状态
        self.prev_btn.setEnabled(self.current_batch_index > 0)
        self.next_btn.setEnabled(self.current_batch_index < len(self.batch_results) - 1)
        
        # 显示当前图片
        result = self.batch_results[self.current_batch_index]
        frame = result['detected_image']
        
        # 转换为 QImage
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qt_img = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # 显示图片，保持原始比例
        display_size = self.result_display.size()
        pixmap = QPixmap.fromImage(qt_img).scaled(
            display_size, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.result_display.setFixedSize(display_size)
        self.result_display.setPixmap(pixmap)
        
        # 更新检测信息
        bicycle_count = result['bicycle_count']
        self.info_panel.setText(f"自行车数量：{bicycle_count} 辆\n检测结果：{len(result['illegal_results'])} 个")
    
    def _show_video_duration_warning(self, duration):
        """显示视频时长警告弹窗"""
        from PyQt5.QtWidgets import QMessageBox
        QMessageBox.warning(
            self,
            "视频时长警告",
            f"视频时长为 {duration:.1f} 秒，超过了5分钟的限制。\n\n建议截取5分钟以内的片段后再进行检测。",
            QMessageBox.Ok
        )

    def _update_frame(self, frame):
        """更新检测帧到界面"""
        # 转换OpenCV帧为Qt格式
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qt_img = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)

        # 使用固定大小缩放（防止循环放大）
        display_size = self.result_display.size()
        pixmap = QPixmap.fromImage(qt_img).scaled(
            display_size, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        # 保持QLabel大小固定
        self.result_display.setFixedSize(display_size)
        self.result_display.setPixmap(pixmap)

    def _update_info(self, info):
        """更新检测信息面板"""
        info_text = f"""
自行车数量：{info['count']} 辆
"""
        if 'illegal_count' in info:
            info_text += f"违规数量：{info['illegal_count']} 辆\n"
        info_text += f"单帧检测耗时：{info['time']} ms\n"
        info_text += f"置信度分布：{info['conf']}"
        self.info_panel.setText(info_text.strip())

    def _update_log(self, msg):
        """更新日志"""
        self.log_display.append(msg)

    # 导出按钮槽函数
    def on_export_clicked(self):
        try:
            # 1. 检查是否有检测结果
            if not hasattr(self, 'current_detect_list') or not self.current_detect_list:
                from PyQt5.QtWidgets import QMessageBox
                QMessageBox.information(self, "提示", "当前未检测到自行车！")
                return

            # 2. 调用导出函数 (来自 record_manage)
            # 注意：如果你想导出历史记录，用 export_to_excel()
            # 如果你想导出当前帧，用 export_current_detection_excel()
            from record_manage import export_current_detection_excel
            from PyQt5.QtWidgets import QFileDialog, QMessageBox

            # 弹出保存对话框
            path, _ = QFileDialog.getSaveFileName(self, "导出检测结果", "", "Excel 表格 (*.xlsx)")
            if path:
                export_current_detection_excel(self.current_detect_list, path)
                QMessageBox.information(self, "成功", f"Excel 已导出至：{path}")
        except Exception as e:
            QMessageBox.warning(self, "错误", f"导出失败：{str(e)}")

    def _export_result(self):
        """导出检测结果（支持图片和Excel）"""
        if not hasattr(self.detection_thread, "current_frame") or self.detection_thread.current_frame is None:
            self.log_display.append("❌ 暂无检测结果可导出！")
            return

        # 弹出选择对话框
        msg = QMessageBox(self)
        msg.setWindowTitle("选择导出类型")
        msg.setText("请选择导出格式：")
        msg.setIcon(QMessageBox.Question)
        
        btn_image = msg.addButton("导出图片", QMessageBox.AcceptRole)
        btn_excel = msg.addButton("导出Excel", QMessageBox.AcceptRole)
        btn_both = msg.addButton("同时导出", QMessageBox.AcceptRole)
        btn_cancel = msg.addButton("取消", QMessageBox.RejectRole)
        
        msg.exec_()
        
        clicked_btn = msg.clickedButton()
        
        if clicked_btn == btn_cancel:
            return
        
        # 导出图片
        if clicked_btn in [btn_image, btn_both]:
            save_path, _ = QFileDialog.getSaveFileName(
                self, "保存检测结果图片", "", "Images (*.png *.jpg *.jpeg)"
            )
            if save_path:
                cv2.imwrite(save_path, self.detection_thread.current_frame)
                self.log_display.append(f"✅ 图片已导出至：{save_path}")
        
        # 导出Excel
        if clicked_btn in [btn_excel, btn_both]:
            if not self.detection_thread.current_detections:
                self.log_display.append("❌ 暂无检测数据可导出Excel！")
                return
            
            save_path, _ = QFileDialog.getSaveFileName(
                self, "保存检测结果Excel", "", "Excel (*.xlsx)"
            )
            if save_path:
                try:
                    result_path, count = export_current_detection_excel(
                        self.detection_thread.current_detections, 
                        save_path
                    )
                    self.log_display.append(f"✅ Excel已导出至：{result_path}，共 {count} 条记录")
                except Exception as e:
                    self.log_display.append(f"❌ Excel导出失败：{str(e)}")


    def _open_database_view(self):
        """打开数据库查看对话框"""
        dialog = DatabaseViewDialog(self)
        dialog.exec_()
    
    def _open_query_dialog(self):
        """打开记录查询对话框"""
        dialog = QueryDialog(self)
        dialog.exec_()
        return

# 测试入口
if __name__ == "__main__":
    app = QApplication(sys.argv)
    # 设置全局字体
    font = QFont("Microsoft YaHei", 9)
    app.setFont(font)
    window = BikeDetectionUI()
    window.show()
    sys.exit(app.exec_())