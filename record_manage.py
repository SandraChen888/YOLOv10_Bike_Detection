import pandas as pd
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# ====================== MySQL 配置 ======================
DB_CONFIG = {
    "user": "root",
    "password": "123456",
    "host": "localhost",
    "port": 3306,
    "database": "bike_violation_db"
}

engine = create_engine(
    f"mysql+pymysql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}/{DB_CONFIG['database']}?charset=utf8mb4"
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# ====================== 数据表模型 ======================
class ViolationRecord(Base):
    __tablename__ = "violation_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    detect_time = Column(DateTime, nullable=False)
    screenshot_path = Column(String(255), nullable=False)
    violation_area = Column(String(100), nullable=False)
    violation_type = Column(String(50), nullable=False)
    create_time = Column(DateTime, default=datetime.now)
    update_time = Column(DateTime, default=datetime.now, onupdate=datetime.now)

# ====================== 1. 保存违规记录 (原有功能) ======================
def save_illegal_record(screenshot_path, violation_area, violation_type):
    db = SessionLocal()
    try:
        record = ViolationRecord(
            detect_time=datetime.now(),
            screenshot_path=screenshot_path,
            violation_area=violation_area,
            violation_type=violation_type
        )
        db.add(record)
        db.commit()
        return True
    except Exception as e:
        db.rollback()
        print(f"数据库保存失败: {e}")
        return False
    finally:
        db.close()

# ====================== 2. 查询历史记录 ======================
def query_record():
    db = SessionLocal()
    try:
        return db.query(ViolationRecord).all()
    finally:
        db.close()

# ====================== 3. 导出【历史违规记录】到 Excel ======================
def export_to_excel(save_path=None):
    """
    兼容你 main.py 的导入名！
    导出数据库中所有已保存的违规记录到 Excel
    """
    if not save_path:
        save_path = f"历史违规记录_{datetime.now().strftime('%Y%m%d%H%M%S')}.xlsx"

    records = query_record()
    data = []
    for r in records:
        data.append({
            "ID": r.id,
            "检测时间": r.detect_time.strftime("%Y-%m-%d %H:%M:%S"),
            "违规区域": r.violation_area,
            "违规类型": r.violation_type,
            "截图路径": r.screenshot_path
        })

    df = pd.DataFrame(data)
    df.to_excel(save_path, index=False)
    print(f"✅ 历史记录已导出: {save_path}")
    return save_path

# ====================== 🔥 4. 导出【当前检测结果】到 Excel (新增核心功能) ======================
def export_current_detection_excel(detect_list, save_path=None):
    """
    导出当前画面检测到的所有自行车到 Excel
    :param detect_list: 格式 [{"x1":.., "y1":.., "x2":.., "y2":.., "score":..}, ...]
    :return: 文件名, 数量
    """
    if not detect_list:
        return None, 0

    if not save_path:
        save_path = f"当前检测结果_{datetime.now().strftime('%Y%m%d%H%M%S')}.xlsx"

    data = []
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for idx, obj in enumerate(detect_list, 1):
        data.append({
            "序号": idx,
            "检测时间": current_time,
            "类别": "bike",
            "置信度": round(obj["score"], 2),
            "左上角X": obj["x1"],
            "左上角Y": obj["y1"],
            "右下角X": obj["x2"],
            "右下角Y": obj["y2"],
        })

    df = pd.DataFrame(data)
    df.to_excel(save_path, index=False)
    print(f"✅ 当前结果已导出: {save_path}")
    return save_path, len(data)