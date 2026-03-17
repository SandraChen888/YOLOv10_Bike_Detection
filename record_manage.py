# record_manage.py
import pandas as pd
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# ====================== MySQL 连接配置（和你 Navicat 一致）======================
DB_CONFIG = {
    "user": "root",
    "password": "123456",  # 你设置的密码
    "host": "localhost",
    "port": 3306,
    "database": "bike_violation_db"
}

# 创建数据库引擎
engine = create_engine(
    f"mysql+pymysql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}?charset=utf8mb4"
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# ====================== 数据库表模型（和你建的表完全对应）======================
class ViolationRecord(Base):
    __tablename__ = "violation_records"

    id = Column(Integer, primary_key=True, autoincrement=True, comment="主键ID")
    detect_time = Column(DateTime, nullable=False, comment="检测时间")
    screenshot_path = Column(String(255), nullable=False, comment="违规截图路径")
    violation_area = Column(String(100), nullable=False, comment="违规区域")
    violation_type = Column(String(50), nullable=False, comment="违规类型")
    create_time = Column(DateTime, default=datetime.now, comment="记录创建时间")
    update_time = Column(DateTime, default=datetime.now, onupdate=datetime.now, comment="记录更新时间")


# ====================== 1. 保存违规记录（兼容你原来的函数名 save_illegal_record）======================
def save_illegal_record(screenshot_path: str, violation_area: str, violation_type: str):
    """
    保存一条违规记录（兼容原有函数名，避免修改main.py）
    :param screenshot_path: 截图文件路径
    :param violation_area: 违规区域（如：教学楼门口）
    :param violation_type: 违规类型（如：违规停放）
    """
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
        print("✅ 违规记录保存成功！")
        return True
    except Exception as e:
        db.rollback()
        print(f"❌ 保存失败：{e}")
        return False
    finally:
        db.close()


# ====================== 2. 按条件查询记录（兼容你原来的函数名 query_record）======================
def query_record(start_time: datetime = None, end_time: datetime = None, violation_type: str = None):
    """
    按条件查询违规记录（兼容原有函数名，避免修改main.py）
    :param start_time: 开始时间（可选）
    :param end_time: 结束时间（可选）
    :param violation_type: 违规类型（可选）
    :return: 记录列表
    """
    db = SessionLocal()
    try:
        query = db.query(ViolationRecord)
        if start_time:
            query = query.filter(ViolationRecord.detect_time >= start_time)
        if end_time:
            query = query.filter(ViolationRecord.detect_time <= end_time)
        if violation_type:
            query = query.filter(ViolationRecord.violation_type == violation_type)
        records = query.all()
        return records
    finally:
        db.close()


# ====================== 3. 导出 Excel ======================
def export_to_excel(save_path: str = None):
    """
    导出所有记录到 Excel
    :param save_path: 保存路径（默认：违规记录_YYYYMMDDHHMMSS.xlsx）
    """
    if not save_path:
        save_path = f"违规记录_{datetime.now().strftime('%Y%m%d%H%M%S')}.xlsx"

    records = query_record()
    data = []
    for r in records:
        data.append({
            "ID": r.id,
            "检测时间": r.detect_time.strftime("%Y-%m-%d %H:%M:%S"),
            "截图路径": r.screenshot_path,
            "违规区域": r.violation_area,
            "违规类型": r.violation_type,
            "创建时间": r.create_time.strftime("%Y-%m-%d %H:%M:%S"),
            "更新时间": r.update_time.strftime("%Y-%m-%d %H:%M:%S")
        })

    df = pd.DataFrame(data)
    df.to_excel(save_path, index=False)
    print(f"✅ Excel 已导出到：{save_path}")
    return save_path


# ====================== 测试代码 ======================
if __name__ == "__main__":
    # 测试保存
    save_illegal_record(
        screenshot_path="./test.jpg",
        violation_area="教学楼门口",
        violation_type="违规停放"
    )

    # 测试查询
    records = query_record()
    for r in records:
        print(f"ID: {r.id}, 区域: {r.violation_area}, 类型: {r.violation_type}")

    # 测试导出
    export_to_excel()