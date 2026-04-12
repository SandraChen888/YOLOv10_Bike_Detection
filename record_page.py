from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
                           QTableWidget, QTableWidgetItem, QHeaderView, QFrame, QInputDialog, QFileDialog, QMessageBox)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor
from datetime import datetime

class RecordPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.current_records = []
        self._init_ui()
        # 初始化加载全部记录
        self._do_query("all")
    
    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(20)
        layout.setContentsMargins(30, 30, 30, 30)
        
        # 页面标题
        title_label = QLabel("记录管理")
        title_label.setStyleSheet("""
            QLabel {
                font-size: 28px;
                font-weight: bold;
                color: #2c3e50;
                margin-bottom: 30px;
            }
        """)
        layout.addWidget(title_label)
        
        # 查询条件筛选栏
        filter_card = QWidget()
        filter_layout = QHBoxLayout(filter_card)
        filter_card.setStyleSheet("""
            QWidget {
                background-color: white;
                border-radius: 12px;
                padding: 20px;
            }
        """)
        
        # 按时间查询
        time_filter = QWidget()
        time_layout = QVBoxLayout(time_filter)
        time_label = QLabel("按时间查询")
        time_label.setStyleSheet("QLabel { font-size: 16px; font-weight: 600; color: #2c3e50; margin-bottom: 10px; }")
        time_btn = QPushButton("选择时间范围")
        time_btn.setStyleSheet("""
            QPushButton {
                padding: 10px 20px;
                background-color: #4096ff;
                color: white;
                border: none;
                border-radius: 8px;
                font-size: 15px;
            }
        """)
        time_btn.clicked.connect(lambda: self._do_query("time"))
        time_layout.addWidget(time_label)
        time_layout.addWidget(time_btn)
        
        # 按违规类型查询
        type_filter = QWidget()
        type_layout = QVBoxLayout(type_filter)
        type_label = QLabel("按违规类型")
        type_label.setStyleSheet("QLabel { font-size: 16px; font-weight: 600; color: #2c3e50; margin-bottom: 10px; }")
        type_btn = QPushButton("选择违规类型")
        type_btn.setStyleSheet("""
            QPushButton {
                padding: 10px 20px;
                background-color: #722ed1;
                color: white;
                border: none;
                border-radius: 8px;
                font-size: 15px;
            }
        """)
        type_btn.clicked.connect(lambda: self._do_query("type"))
        type_layout.addWidget(type_label)
        type_layout.addWidget(type_btn)
        
        # 按区域查询
        area_filter = QWidget()
        area_layout = QVBoxLayout(area_filter)
        area_label = QLabel("按区域查询")
        area_label.setStyleSheet("QLabel { font-size: 16px; font-weight: 600; color: #2c3e50; margin-bottom: 10px; }")
        area_btn = QPushButton("选择区域")
        area_btn.setStyleSheet("""
            QPushButton {
                padding: 10px 20px;
                background-color: #67c23a;
                color: white;
                border: none;
                border-radius: 8px;
                font-size: 15px;
            }
        """)
        area_btn.clicked.connect(lambda: self._do_query("area"))
        area_layout.addWidget(area_label)
        area_layout.addWidget(area_btn)
        
        # 查询全部
        all_filter = QWidget()
        all_layout = QVBoxLayout(all_filter)
        all_label = QLabel("全部记录")
        all_label.setStyleSheet("QLabel { font-size: 16px; font-weight: 600; color: #2c3e50; margin-bottom: 10px; }")
        all_btn = QPushButton("查询全部")
        all_btn.setStyleSheet("""
            QPushButton {
                padding: 10px 20px;
                background-color: #fa8c16;
                color: white;
                border: none;
                border-radius: 8px;
                font-size: 15px;
            }
        """)
        all_btn.clicked.connect(lambda: self._do_query("all"))
        all_layout.addWidget(all_label)
        all_layout.addWidget(all_btn)
        
        filter_layout.addWidget(time_filter)
        filter_layout.addWidget(type_filter)
        filter_layout.addWidget(area_filter)
        filter_layout.addWidget(all_filter)
        filter_layout.addStretch()
        
        layout.addWidget(filter_card)
        
        # 违规记录列表
        record_card = QWidget()
        record_layout = QVBoxLayout(record_card)
        record_card.setStyleSheet("""
            QWidget {
                background-color: white;
                border-radius: 12px;
                padding: 25px;
            }
        """)
        
        # 记录表格
        self.record_table = QTableWidget()
        self.record_table.setColumnCount(7)
        self.record_table.setHorizontalHeaderLabels([
            "ID", "检测时间", "检测区域", "违规类型", "是否违规", "创建时间", "更新时间"
        ])
        
        header = self.record_table.horizontalHeader()
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
        header.setSectionResizeMode(0)
        
        self.record_table.setStyleSheet("""
            QTableWidget {
                background-color: white;
                border: 1px solid #e5e9f0;
                border-radius: 10px;
                gridline-color: #f0f2f5;
                font-size: 14px;
            }
            QTableWidget::item {
                padding: 12px;
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
        self.record_table.setAlternatingRowColors(True)
        self.record_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.record_table.setEditTriggers(QTableWidget.NoEditTriggers)
        
        record_layout.addWidget(self.record_table, 5)
        
        # 底部操作按钮
        bottom_layout = QHBoxLayout()
        self.export_btn = QPushButton("导出选中记录")
        self.export_btn.setStyleSheet("""
            QPushButton {
                padding: 12px 24px;
                background-color: #4096ff;
                color: white;
                border: none;
                border-radius: 8px;
                font-weight: 600;
                font-size: 16px;
            }
        """)
        self.export_btn.clicked.connect(self._export_selected_records)
        self.export_btn.setEnabled(False)
        
        refresh_btn = QPushButton("刷新记录")
        refresh_btn.setStyleSheet("""
            QPushButton {
                padding: 12px 24px;
                background-color: #67c23a;
                color: white;
                border: none;
                border-radius: 8px;
                font-weight: 600;
                font-size: 16px;
            }
        """)
        refresh_btn.clicked.connect(lambda: self._do_query("all"))
        
        bottom_layout.addWidget(self.export_btn)
        bottom_layout.addWidget(refresh_btn)
        bottom_layout.addStretch()
        record_layout.addLayout(bottom_layout)
        
        layout.addWidget(record_card, 3)
        
        return self
    
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
        self.record_table.setRowCount(0)
        self.export_btn.setEnabled(len(records) > 0)
        
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
