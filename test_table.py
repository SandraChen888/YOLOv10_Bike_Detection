from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QTableWidget, QTableWidgetItem, QHeaderView
from PyQt5.QtCore import Qt
import sys

class TestTable(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
    
    def initUI(self):
        self.setWindowTitle('测试表格表头')
        self.setGeometry(100, 100, 1200, 600)
        
        layout = QVBoxLayout(self)
        
        # 创建表格
        table = QTableWidget(0, 9)
        
        # 设置表头
        table.setHorizontalHeaderLabels([
            "ID", "检测时间", "检测区域", "违规车辆数量", "合法停放车辆数量", "是否违规", "创建时间", "更新时间", "操作"
        ])
        
        # 设置表头属性
        header = table.horizontalHeader()
        header.setVisible(True)
        header.setMinimumHeight(50)
        header.setDefaultAlignment(Qt.AlignCenter)
        
        # 设置列宽
        table.setColumnWidth(0, 60)
        table.setColumnWidth(3, 100)
        table.setColumnWidth(4, 100)
        table.setColumnWidth(8, 100)
        
        # 拉伸列宽
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.Stretch)
        header.setSectionResizeMode(5, QHeaderView.Stretch)
        header.setSectionResizeMode(6, QHeaderView.Stretch)
        header.setSectionResizeMode(7, QHeaderView.Stretch)
        
        # 设置表格属性
        table.setShowGrid(True)
        table.setAlternatingRowColors(True)
        table.setSelectionBehavior(QTableWidget.SelectRows)
        table.setEditTriggers(QTableWidget.NoEditTriggers)
        
        # 添加到布局
        layout.addWidget(table)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = TestTable()
    ex.show()
    sys.exit(app.exec_())