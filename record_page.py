from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
                           QTableWidget, QTableWidgetItem, QHeaderView, QFrame, QInputDialog, QFileDialog, QMessageBox, QComboBox, QDialog, QSizePolicy)
from PyQt5.QtCore import Qt, QTimer
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
        filter_layout.addWidget(area_filter)
        filter_layout.addWidget(all_filter)
        filter_layout.addStretch()
        
        layout.addWidget(filter_card)
        
        # 违规记录列表
        record_card = QWidget()
        record_layout = QVBoxLayout(record_card)
        
        # 记录表格 - 直接指定行数和列数
        self.record_table = QTableWidget(0, 9)
        
        # 设置表头标签
        self.record_table.setHorizontalHeaderLabels([
            "ID", "检测时间", "检测区域", "违规车辆数量", "合法停放车辆数量", "是否违规", "创建时间", "更新时间", "操作"
        ])
        
        # 获取表头对象
        header = self.record_table.horizontalHeader()
        
        # 设置表头属性
        header.setMinimumHeight(50)
        header.setDefaultAlignment(Qt.AlignCenter)
        
        # 设置表头样式，确保字体颜色为黑色
        header.setStyleSheet("""
            QHeaderView::section {
                background-color: white;
                color: black;
                font-weight: bold;
                font-size: 14px;
                padding: 4px;
                border: 1px solid #dcdfe6;
            }
        """)
        
        # 设置列宽
        # 固定列宽
        self.record_table.setColumnWidth(0, 60)  # ID列
        self.record_table.setColumnWidth(3, 100)  # 违规车辆数量
        self.record_table.setColumnWidth(4, 100)  # 合法停放车辆数量
        self.record_table.setColumnWidth(8, 100)  # 操作列
        
        # 拉伸列宽
        header.setSectionResizeMode(1, QHeaderView.Stretch)  # 检测时间
        header.setSectionResizeMode(2, QHeaderView.Stretch)  # 检测区域
        header.setSectionResizeMode(5, QHeaderView.Stretch)  # 是否违规
        header.setSectionResizeMode(6, QHeaderView.Stretch)  # 创建时间
        header.setSectionResizeMode(7, QHeaderView.Stretch)  # 更新时间
        
        # 设置表格属性
        self.record_table.setShowGrid(True)
        self.record_table.setAlternatingRowColors(True)
        self.record_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.record_table.setEditTriggers(QTableWidget.NoEditTriggers)
        
        # 确保表格有足够的空间
        self.record_table.setMinimumSize(1000, 500)
        self.record_table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        # 保存表头对象
        self.header = header
        
        # 调试信息
        print(f"表头可见性: {header.isVisible()}")
        print(f"表头高度: {header.height()}")
        print(f"表头最小高度: {header.minimumHeight()}")
        print(f"表格行数: {self.record_table.rowCount()}")
        print(f"表格列数: {self.record_table.columnCount()}")
        
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
        
        delete_btn = QPushButton("删除选中记录")
        delete_btn.setStyleSheet("""
            QPushButton {
                padding: 12px 24px;
                background-color: #f56c6c;
                color: white;
                border: none;
                border-radius: 8px;
                font-weight: 600;
                font-size: 16px;
            }
        """)
        delete_btn.clicked.connect(self._delete_selected_records)
        delete_btn.setEnabled(False)
        
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
        bottom_layout.addWidget(delete_btn)
        bottom_layout.addWidget(refresh_btn)
        bottom_layout.addStretch()
        
        # 保存删除按钮引用
        self.delete_btn = delete_btn
        
        # 添加表格到布局
        record_layout.addWidget(self.record_table)
        record_layout.addLayout(bottom_layout)
        
        # 添加到主布局
        layout.addWidget(record_card)
        
        # 使用定时器延迟设置表头可见性，确保UI初始化完成
        QTimer.singleShot(100, self._ensure_header_visible)
        
    def _ensure_header_visible(self):
        """确保表头可见"""
        if hasattr(self, 'header'):
            self.header.setVisible(True)
            print(f"定时器设置后表头可见性: {self.header.isVisible()}")
            # 再次设置表头样式
            self.header.setStyleSheet("""
                QHeaderView::section {
                    background-color: white;
                    color: black;
                    font-weight: bold;
                    font-size: 14px;
                    padding: 4px;
                    border: 1px solid #dcdfe6;
                }
            """)
            print(f"定时器设置后表头样式已更新")
    
    def _do_query(self, query_type):
        """执行记录查询"""
        try:
            from record_manage import (
                query_record, query_by_time_range,
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

            elif query_type == "area":
                # 创建自定义对话框
                dialog = QDialog(self)
                dialog.setWindowTitle("区域查询")
                dialog.setGeometry(100, 100, 400, 150)
                dialog.setStyleSheet("""
                    QDialog {
                        background-color: white;
                        border-radius: 12px;
                        padding: 20px;
                    }
                """)
                
                layout = QVBoxLayout(dialog)
                
                label = QLabel("请选择违规区域：")
                label.setStyleSheet("QLabel { font-size: 18px; font-weight: 600; color: #2c3e50; margin-bottom: 15px; }")
                layout.addWidget(label)
                
                # 创建场景选择下拉框
                combo = QComboBox()
                combo.addItems([
                    "默认场景",
                    "教学楼A栋",
                    "教学楼B栋",
                    "教学楼C栋",
                    "教学楼D栋",
                    "教学楼E栋",
                    "教学楼F栋",
                    "一饭堂门口"
                ])
                combo.setStyleSheet("""
                    QComboBox {
                        padding: 14px 18px;
                        border: 2px solid #e5e9f0;
                        border-radius: 10px;
                        background-color: white;
                        selection-background-color: #e8f4ff;
                        font-size: 19px;
                    }
                    QComboBox:hover {
                        border-color: #c6e2ff;
                    }
                    QComboBox:focus {
                        border-color: #4096ff;
                    }
                """)
                layout.addWidget(combo)
                
                # 按钮布局
                btn_layout = QHBoxLayout()
                ok_btn = QPushButton("确定")
                ok_btn.setStyleSheet("""
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
                cancel_btn = QPushButton("取消")
                cancel_btn.setStyleSheet("""
                    QPushButton {
                        padding: 12px 24px;
                        background-color: #909399;
                        color: white;
                        border: none;
                        border-radius: 8px;
                        font-weight: 600;
                        font-size: 16px;
                    }
                """)
                btn_layout.addStretch()
                btn_layout.addWidget(ok_btn)
                btn_layout.addWidget(cancel_btn)
                layout.addLayout(btn_layout)
                
                # 存储选择结果
                selected_area = None
                
                def on_ok():
                    nonlocal selected_area
                    selected_area = combo.currentText()
                    dialog.close()
                
                def on_cancel():
                    dialog.close()
                
                ok_btn.clicked.connect(on_ok)
                cancel_btn.clicked.connect(on_cancel)
                
                # 显示对话框
                dialog.exec_()
                
                # 执行查询
                if selected_area and selected_area != "默认场景":
                    self.current_records = query_by_violation_area(selected_area)
            
            # 显示记录
            self._display_records(self.current_records)
            
        except Exception as e:
            QMessageBox.warning(self, "查询错误", f"查询失败：{str(e)}")
    
    def _display_records(self, records):
        """显示记录到表格"""
        self.record_table.setRowCount(0)
        self.export_btn.setEnabled(len(records) > 0)
        self.delete_btn.setEnabled(len(records) > 0)
        
        for row_idx, record in enumerate(records):
            self.record_table.insertRow(row_idx)
            
            self.record_table.setItem(row_idx, 0, QTableWidgetItem(str(record.id)))
            self.record_table.setItem(row_idx, 1, QTableWidgetItem(
                record.detect_time.strftime("%Y-%m-%d %H:%M:%S")
            ))
            self.record_table.setItem(row_idx, 2, QTableWidgetItem(record.violation_area))
            self.record_table.setItem(row_idx, 3, QTableWidgetItem(str(record.illegal_count)))
            self.record_table.setItem(row_idx, 4, QTableWidgetItem(str(record.legal_count)))
            
            is_illegal_item = QTableWidgetItem("违规" if record.is_illegal == 1 else "合法")
            if record.is_illegal == 1:
                is_illegal_item.setForeground(QColor("#f56c6c"))
            else:
                is_illegal_item.setForeground(QColor("#52c41a"))
            is_illegal_item.setTextAlignment(Qt.AlignCenter)
            self.record_table.setItem(row_idx, 5, is_illegal_item)
            
            create_time_str = record.create_time.strftime("%Y-%m-%d %H:%M:%S") if record.create_time else "-"
            self.record_table.setItem(row_idx, 6, QTableWidgetItem(create_time_str))
            
            update_time_str = record.update_time.strftime("%Y-%m-%d %H:%M:%S") if record.update_time else "-"
            self.record_table.setItem(row_idx, 7, QTableWidgetItem(update_time_str))
            
            # 添加误报反馈按钮
            feedback_btn = QPushButton("误报反馈")
            feedback_btn.setStyleSheet("""
                QPushButton {
                    padding: 6px 12px;
                    background-color: #ff7875;
                    color: white;
                    border: none;
                    border-radius: 4px;
                    font-size: 12px;
                }
                QPushButton:hover {
                    background-color: #ff4d4f;
                }
            """)
            # 绑定按钮点击事件，传递记录ID
            feedback_btn.clicked.connect(lambda _, r=record: self._handle_feedback(r))
            self.record_table.setCellWidget(row_idx, 8, feedback_btn)
    
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
    
    def _delete_selected_records(self):
        """删除选中的记录"""
        selected_rows = set()
        for item in self.record_table.selectedItems():
            selected_rows.add(item.row())
        
        if not selected_rows:
            QMessageBox.information(self, "提示", "请先选择要删除的记录！")
            return
        
        # 确认删除
        reply = QMessageBox.question(
            self, "删除确认",
            f"确定要删除选中的 {len(selected_rows)} 条记录吗？此操作不可恢复！",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            try:
                from record_manage import delete_record
                deleted_count = 0
                # 保存要删除的记录ID
                record_ids = []
                for row in selected_rows:
                    record = self.current_records[row]
                    record_ids.append(record.id)
                
                # 执行删除
                for record_id in record_ids:
                    if delete_record(record_id):
                        deleted_count += 1
                
                if deleted_count > 0:
                    QMessageBox.information(self, "删除成功", f"成功删除 {deleted_count} 条记录！")
                    # 刷新记录列表
                    self._do_query("all")
                else:
                    QMessageBox.warning(self, "删除错误", "删除失败，请稍后重试！")
                
            except Exception as e:
                QMessageBox.warning(self, "删除错误", f"删除失败：{str(e)}")

    def _handle_feedback(self, record):
        """处理误报反馈"""
        # 显示反馈对话框，让管理人员输入反馈信息
        feedback_reason, ok = QInputDialog.getMultiLineText(
            self, "误报反馈",
            f"请输入对记录 ID: {record.id} 的反馈原因：",
            ""
        )
        
        if ok and feedback_reason:
            try:
                # 保存反馈信息到数据库
                from record_manage import save_feedback
                save_status = save_feedback(
                    record_id=record.id,
                    feedback_type="误报",
                    feedback_reason=feedback_reason
                )
                
                if save_status:
                    # 显示成功消息
                    QMessageBox.information(self, "反馈成功", "误报反馈已提交，感谢您的反馈！")
                    # 刷新记录列表
                    self._do_query("all")
                else:
                    QMessageBox.warning(self, "反馈错误", "反馈保存失败，请稍后重试！")
                
            except Exception as e:
                QMessageBox.warning(self, "反馈错误", f"反馈提交失败：{str(e)}")
        elif ok and not feedback_reason:
            QMessageBox.warning(self, "提示", "请输入反馈原因！")
