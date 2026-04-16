#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自行车检测UI界面
"""

import sys
import os
import cv2
from datetime import datetime
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QLabel, QComboBox, QSlider, QTextEdit, QFileDialog, QMessageBox,
    QTableWidget, QTableWidgetItem, QInputDialog, QSizePolicy, QFrame, QStackedWidget,
    QHeaderView
)
from PyQt5.QtGui import QImage, QPixmap, QColor
from PyQt5.QtCore import Qt, QThread, pyqtSignal

from detection_thread import DetectionThread
from detection_page import DetectionPage
from record_page import RecordPage

class BikeDetectionUI(QMainWindow):
    """自行车检测UI类"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("校园自行车违停检测系统")
        self.setGeometry(100, 100, 1400, 800)
        
        # 模型路径
        self.model_path = "F:\\YOLOv10_Bike_Detection\\merged_bicycle_split\\runs\\detect\\bike_yellow_grid_300e\\weights\\best.pt"
        
        # 初始化界面
        self._init_ui()
        self._create_menu_bar()
    
    def _init_ui(self):
        """初始化界面布局"""
        # 中心部件
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        
        # 页面容器 - 使用QStackedWidget实现页面切换
        self.stacked_widget = QStackedWidget()
        self.main_layout.addWidget(self.stacked_widget)
        
        # 创建两个页面
        self.detection_page = DetectionPage(self)
        self.record_page = RecordPage(self)
        
        # 添加页面到QStackedWidget
        self.stacked_widget.addWidget(self.detection_page)
        self.stacked_widget.addWidget(self.record_page)
    
    def _create_menu_bar(self):
        """创建顶部导航栏（页面切换）"""
        # 创建自定义导航栏
        nav_bar = QWidget()
        nav_layout = QHBoxLayout(nav_bar)
        nav_bar.setStyleSheet("""
            QWidget {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                    stop:0 #4096ff, stop:1 #3088ff);
                padding: 0 30px;
            }
        """)
        
        # 导航按钮
        self.nav_buttons = []
        
        # 核心检测页面按钮
        detect_btn = QPushButton("核心检测")
        detect_btn.setStyleSheet(self._get_nav_button_style())
        detect_btn.clicked.connect(lambda: self._switch_page("detection"))
        nav_layout.addWidget(detect_btn)
        self.nav_buttons.append(detect_btn)
        
        # 记录管理页面按钮
        record_btn = QPushButton("记录管理")
        record_btn.setStyleSheet(self._get_nav_button_style())
        record_btn.clicked.connect(lambda: self._switch_page("record"))
        nav_layout.addWidget(record_btn)
        self.nav_buttons.append(record_btn)
        
        # 添加到菜单栏
        menubar = self.menuBar()
        menubar.setStyleSheet("""
            QMenuBar {
                background-color: #4096ff;
                color: white;
            }
        """)
        
        # 替换默认菜单栏
        self.setMenuWidget(nav_bar)
        
        # 默认选中核心检测页面
        self._switch_page("detection")
    
    def _get_nav_button_style(self):
        """获取导航按钮样式"""
        return """
            QPushButton {
                padding: 18px 35px;
                background-color: transparent;
                color: white;
                border: none;
                font-size: 18px;
                font-weight: 600;
                border-radius: 8px;
                margin: 5px;
            }
            QPushButton:hover {
                background-color: rgba(255, 255, 255, 0.15);
            }
            QPushButton:pressed {
                background-color: rgba(255, 255, 255, 0.25);
            }
        """
    
    def _get_nav_button_active_style(self):
        """获取导航按钮激活状态样式"""
        return """
            QPushButton {
                padding: 18px 35px;
                background-color: rgba(255, 255, 255, 0.25);
                color: white;
                border: none;
                font-size: 18px;
                font-weight: 600;
                border-radius: 8px;
                margin: 5px;
            }
        """
    
    def _switch_page(self, page_name):
        """切换页面"""
        # 激活对应的导航按钮
        for i, btn in enumerate(self.nav_buttons):
            if i == 0 and page_name == "detection":
                btn.setStyleSheet(self._get_nav_button_active_style())
            elif i == 1 and page_name == "record":
                btn.setStyleSheet(self._get_nav_button_active_style())
            else:
                btn.setStyleSheet(self._get_nav_button_style())
        
        # 切换到对应页面
        if page_name == "detection":
            self.stacked_widget.setCurrentIndex(0)
        elif page_name == "record":
            self.stacked_widget.setCurrentIndex(1)
    

    
    def _select_source_type(self, source_type):
        """根据来源类型选择文件"""
        if source_type == "image":
            self.source_combo.setCurrentIndex(0)  # 本地图片
            self._select_source()
        elif source_type == "video":
            self.source_combo.setCurrentIndex(2)  # 本地视频
            self._select_source()
        elif source_type == "batch":
            self.source_combo.setCurrentIndex(1)  # 批量图片
            self._select_source()
        elif source_type == "camera":
            self.source_combo.setCurrentIndex(3)  # 摄像头实时
    
    def _open_model_test(self):
        """打开模型测试对话框"""
        QMessageBox.information(self, "模型测试", "模型测试功能开发中...")
    
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
        if self.source_combo.currentText() != "摄像头实时" and not hasattr(self, 'source_path'):
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
            source_path=self.source_path if hasattr(self, 'source_path') else "",
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
        if hasattr(self, 'detection_thread') and self.detection_thread.isRunning():
            self.detection_thread.stop()
            self.log_display.append("⏹️ 已停止检测")

    def _reset_btn_state(self):
        """重置按钮状态"""
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        
        # 批量检测完成后更新结果
        if hasattr(self, 'detection_thread') and self.detection_thread.source_type == "batch_image":
            self.batch_results = self.detection_thread.batch_results
            if self.batch_results:
                self.current_batch_index = 0
                self._display_batch_image()
                self.log_display.append(f"✅ 批量检测结果已加载，共 {len(self.batch_results)} 张图片")

    def _prev_batch_image(self):
        """显示上一张批量检测图片"""
        if hasattr(self, 'batch_results') and self.batch_results and self.current_batch_index > 0:
            self.current_batch_index -= 1
            self._display_batch_image()

    def _next_batch_image(self):
        """显示下一张批量检测图片"""
        if hasattr(self, 'batch_results') and self.batch_results and self.current_batch_index < len(self.batch_results) - 1:
            self.current_batch_index += 1
            self._display_batch_image()

    def _display_batch_image(self):
        """显示当前批量检测图片"""
        if not hasattr(self, 'batch_results') or not self.batch_results:
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
        illegal_count = len(result['illegal_results'])
        legal_count = bicycle_count - illegal_count
        
        # 更新统计卡片
        self.stats_labels['总检测车辆'].setText(f"{bicycle_count} 辆")
        self.stats_labels['违规车辆数量'].setText(f"{illegal_count} 辆")
        self.stats_labels['合法车辆数量'].setText(f"{legal_count} 辆")

    def _show_video_duration_warning(self, duration):
        """显示视频时长警告弹窗"""
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
        # 更新统计卡片信息
        if hasattr(self, 'stats_labels'):
            # 总检测车辆
            if 'total' in info:
                self.stats_labels['总检测车辆'].setText(f"{info['total']} 辆")
            
            # 违规车辆数量
            if 'illegal' in info:
                self.stats_labels['违规车辆数量'].setText(f"{info['illegal']} 辆")
                # 合法车辆数量 = 总车辆 - 违规车辆
                if 'total' in info:
                    legal_count = info['total'] - info['illegal']
                    self.stats_labels['合法车辆数量'].setText(f"{legal_count} 辆")
            
            # 检测耗时
            if 'time' in info:
                self.stats_labels['检测耗时'].setText(f"{info['time']} ms")

    def _update_log(self, msg):
        """更新日志"""
        self.log_display.append(msg)

    # 导出按钮槽函数
    def on_export_clicked(self):
        try:
            # 1. 检查是否有检测结果
            if not hasattr(self, 'current_detect_list') or not self.current_detect_list:
                QMessageBox.information(self, "提示", "当前未检测到自行车！")
                return

            # 2. 调用导出函数 (来自 record_manage)
            # 注意：如果你想导出历史记录，用 export_to_excel()
            # 如果你想导出当前帧，用 export_current_detection_excel()
            from record_manage import export_current_detection_excel

            # 弹出保存对话框
            path, _ = QFileDialog.getSaveFileName(self, "导出检测结果", "", "Excel 表格 (*.xlsx)")
            if path:
                export_current_detection_excel(self.current_detect_list, path)
                QMessageBox.information(self, "成功", f"Excel 已导出至：{path}")
        except Exception as e:
            QMessageBox.warning(self, "错误", f"导出失败：{str(e)}")

    def _export_result(self):
        """导出检测结果（支持图片和Excel）"""
        if not hasattr(self, 'detection_thread') or not hasattr(self.detection_thread, "current_frame") or self.detection_thread.current_frame is None:
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
            if not hasattr(self.detection_thread, 'current_detections') or not self.detection_thread.current_detections:
                self.log_display.append("❌ 暂无检测数据可导出Excel！")
                return
            
            save_path, _ = QFileDialog.getSaveFileName(
                self, "保存检测结果Excel", "", "Excel (*.xlsx)"
            )
            if save_path:
                try:
                    from record_manage import export_current_detection_excel
                    result_path, count = export_current_detection_excel(
                        self.detection_thread.current_detections, 
                        save_path
                    )
                    self.log_display.append(f"✅ Excel已导出至：{result_path}，共 {count} 条记录")
                except Exception as e:
                    self.log_display.append(f"❌ Excel导出失败：{str(e)}")

    def _open_database_view(self):
        """打开数据库查看对话框"""
        pass

    def _open_query_dialog(self):
        """打开记录查询对话框"""
        pass
    
    def _do_query(self, query_type):
        """执行记录查询"""
        try:
            from record_manage import (
                query_record, query_by_time_range, query_by_violation_type,
                query_by_violation_area, get_detection_statistics
            )
            
            self.current_records = []
            
            if query_type == "all":
                self.current_records = query_record()
            elif query_type == "time":
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
            elif query_type == "type":
                violation_type, ok = QInputDialog.getItem(
                    self, "类型查询", "请选择违规类型:",
                    ["人行道", "教学楼门口", "消防通道", "绿化带", "其他", "合法"],
                    editable=True
                )
                if ok and violation_type:
                    self.current_records = query_by_violation_type(violation_type)
            elif query_type == "area":
                violation_area, ok = QInputDialog.getText(
                    self, "区域查询", "请输入违规区域（如：教学楼A栋、一饭堂门口）:"
                )
                if ok and violation_area:
                    self.current_records = query_by_violation_area(violation_area)
            
            # 显示记录
            self._display_records(self.current_records)
            
        except Exception as e:
            QMessageBox.warning(self, "查询错误", f"查询失败：{str(e)}")
    
    def _display_records(self, records):
        """显示记录到表格"""
        if hasattr(self, 'record_table'):
            self.record_table.setRowCount(0)
            self.export_record_btn.setEnabled(len(records) > 0)
            
            for row_idx, record in enumerate(records):
                self.record_table.insertRow(row_idx)
                
                self.record_table.setItem(row_idx, 0, QTableWidgetItem(str(record.id)))
                self.record_table.setItem(row_idx, 1, QTableWidgetItem(
                    record.detect_time.strftime("%Y-%m-%d %H:%M:%S")
                ))
                self.record_table.setItem(row_idx, 2, QTableWidgetItem(record.violation_area))
                self.record_table.setItem(row_idx, 3, QTableWidgetItem(record.violation_type))
                
                is_illegal_item = QTableWidgetItem("违规" if record.is_illegal == 1 else "合法")
                if record.is_illegal == 1:
                    is_illegal_item.setForeground(QColor("#f56c6c"))
                else:
                    is_illegal_item.setForeground(QColor("#52c41a"))
                is_illegal_item.setTextAlignment(Qt.AlignCenter)
                self.record_table.setItem(row_idx, 4, is_illegal_item)
                
                create_time_str = record.create_time.strftime("%Y-%m-%d %H:%M:%S") if record.create_time else "-"
                self.record_table.setItem(row_idx, 5, QTableWidgetItem(create_time_str))
                
                update_time_str = record.update_time.strftime("%Y-%m-%d %H:%M:%S") if record.update_time else "-"
                self.record_table.setItem(row_idx, 6, QTableWidgetItem(update_time_str))
    
    def _export_selected_records(self):
        """导出选中的记录"""
        if hasattr(self, 'record_table'):
            selected_rows = set()
            for item in self.record_table.selectedItems():
                selected_rows.add(item.row())
            
            if not selected_rows:
                QMessageBox.information(self, "提示", "请先选择要导出的记录！")
                return
            
            selected_records = [self.current_records[row] for row in selected_rows]
            
            save_path, _ = QFileDialog.getSaveFileName(
                self, "导出选中记录", "选中记录.xlsx", "Excel (*.xlsx)"
            )
            if save_path:
                try:
                    from record_manage import export_query_results_to_excel
                    export_query_results_to_excel(selected_records, save_path)
                    QMessageBox.information(self, "导出成功", f"选中记录已导出至：{save_path}")
                except Exception as e:
                    QMessageBox.warning(self, "导出错误", f"导出失败：{str(e)}")

# 测试入口
if __name__ == "__main__":
    app = QApplication(sys.argv)
    # 设置全局字体
    font = QFont("Microsoft YaHei", 9)
    app.setFont(font)
    window = BikeDetectionUI()
    window.show()
    sys.exit(app.exec_())