import pandas as pd
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text
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
    illegal_count = Column(Integer, nullable=False, default=0)  # 违规车辆数量
    legal_count = Column(Integer, nullable=False, default=0)  # 合法停放车辆数量
    is_illegal = Column(Integer, nullable=False, default=1)  # 0=合法, 1=违规
    create_time = Column(DateTime, default=datetime.now)
    update_time = Column(DateTime, default=datetime.now, onupdate=datetime.now)

class FeedbackRecord(Base):
    __tablename__ = "feedback_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    record_id = Column(Integer, nullable=False)  # 关联的违规记录ID
    feedback_type = Column(String(50), nullable=False)  # 反馈类型（如"误报"、"漏报"等）
    feedback_reason = Column(Text, nullable=True)  # 反馈原因
    feedback_time = Column(DateTime, nullable=False, default=datetime.now)
    create_time = Column(DateTime, default=datetime.now)
    update_time = Column(DateTime, default=datetime.now, onupdate=datetime.now)

# 自动创建表结构
Base.metadata.create_all(bind=engine)
print("✅ 数据库表结构已创建")

# ====================== 1. 保存检测记录 (支持合法和违规) ======================
def save_detection_record(screenshot_path, violation_area, illegal_count, legal_count, is_illegal=1):
    """
    保存检测记录（支持合法和违规）
    :param screenshot_path: 截图路径
    :param violation_area: 检测区域
    :param illegal_count: 违规车辆数量
    :param legal_count: 合法停放车辆数量
    :param is_illegal: 是否违规（0=合法, 1=违规）
    :return: 是否保存成功
    """
    db = SessionLocal()
    try:
        record = ViolationRecord(
            detect_time=datetime.now(),
            screenshot_path=screenshot_path,
            violation_area=violation_area,
            illegal_count=illegal_count,
            legal_count=legal_count,
            is_illegal=is_illegal
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


# 兼容旧函数名
def save_illegal_record(illegal_res, mark_img, screenshot_path, violation_area, illegal_count=None, legal_count=None):
    """保存违规记录（兼容旧代码）"""
    # 计算违规车辆数量和合法车辆数量
    if illegal_count is None:
        illegal_count = len(illegal_res)
    if legal_count is None:
        # 由于我们没有总单车数量，这里暂时将合法数量设为 0，实际应该从检测结果中获取
        legal_count = 0
    # 确定是否违规：当违规车辆数量不为0时记为违规，当违规车辆数量为0时记为不违规
    is_illegal = 1 if illegal_count > 0 else 0
    # 保存记录
    save_status = save_detection_record(screenshot_path, violation_area, illegal_count, legal_count, is_illegal=is_illegal)
    # 返回状态和结果
    return save_status, f"保存成功：{screenshot_path}"

# ====================== 2. 查询历史记录（增强版） ======================
def query_record(start_time=None, end_time=None, violation_area=None, is_illegal=None):
    """
    查询历史检测记录（支持多条件筛选）
    :param start_time: 开始时间 (datetime对象或字符串)
    :param end_time: 结束时间 (datetime对象或字符串)
    :param violation_area: 违规区域（场景名称）
    :param is_illegal: 是否违规（0=合法, 1=违规, None=全部）
    :return: 检测记录列表
    """
    db = SessionLocal()
    try:
        query = db.query(ViolationRecord)
        
        # 按时间范围筛选
        if start_time:
            if isinstance(start_time, str):
                start_time = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
            query = query.filter(ViolationRecord.detect_time >= start_time)
        
        if end_time:
            if isinstance(end_time, str):
                end_time = datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S")
            query = query.filter(ViolationRecord.detect_time <= end_time)
        
        # 按违规区域筛选
        if violation_area:
            query = query.filter(ViolationRecord.violation_area == violation_area)
        
        # 按是否违规筛选
        if is_illegal is not None:
            query = query.filter(ViolationRecord.is_illegal == is_illegal)
        
        # 按时间倒序排列（最新的在前）
        return query.order_by(ViolationRecord.detect_time.desc()).all()
    finally:
        db.close()


def query_by_legality(is_illegal):
    """按是否违规查询检测记录"""
    return query_record(is_illegal=is_illegal)


def query_by_time_range(start_time, end_time):
    """按时间范围查询违规记录"""
    return query_record(start_time=start_time, end_time=end_time)





def query_by_violation_area(violation_area):
    """按违规区域查询违规记录"""
    return query_record(violation_area=violation_area)


def get_detection_statistics(start_time=None, end_time=None):
    """
    获取检测统计信息（包含合法和违规）
    :return: 统计字典，包含各类型检测数量
    """
    db = SessionLocal()
    try:
        query = db.query(ViolationRecord)
        
        if start_time:
            if isinstance(start_time, str):
                start_time = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
            query = query.filter(ViolationRecord.detect_time >= start_time)
        
        if end_time:
            if isinstance(end_time, str):
                end_time = datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S")
            query = query.filter(ViolationRecord.detect_time <= end_time)
        
        records = query.all()
        
        # 统计各类型检测数量
        area_count = {}
        legality_count = {0: 0, 1: 0}  # 0=合法, 1=违规
        
        for r in records:
            area_count[r.violation_area] = area_count.get(r.violation_area, 0) + 1
            legality_count[r.is_illegal] = legality_count.get(r.is_illegal, 0) + 1
        
        return {
            "total_count": len(records),
            "area_statistics": area_count,
            "legality_statistics": {
                "合法": legality_count.get(0, 0),
                "违规": legality_count.get(1, 0)
            }
        }
    finally:
        db.close()


# 兼容旧函数名
def get_violation_statistics(start_time=None, end_time=None):
    """获取违规统计信息（兼容旧代码）"""
    return get_detection_statistics(start_time, end_time)


def delete_record(record_id):
    """删除指定ID的违规记录"""
    db = SessionLocal()
    try:
        record = db.query(ViolationRecord).filter(ViolationRecord.id == record_id).first()
        if record:
            db.delete(record)
            db.commit()
            return True
        return False
    except Exception as e:
        db.rollback()
        print(f"删除记录失败: {e}")
        return False
    finally:
        db.close()


def clear_all_records():
    """清空所有违规记录（慎用）"""
    db = SessionLocal()
    try:
        db.query(ViolationRecord).delete()
        db.commit()
        return True
    except Exception as e:
        db.rollback()
        print(f"清空记录失败: {e}")
        return False
    finally:
        db.close()

# ====================== 7. 保存反馈记录 ======================
def save_feedback(record_id, feedback_type, feedback_reason=None):
    """
    保存反馈记录
    :param record_id: 关联的违规记录ID
    :param feedback_type: 反馈类型（如"误报"、"漏报"等）
    :param feedback_reason: 反馈原因（可选）
    :return: 是否保存成功
    """
    db = SessionLocal()
    try:
        feedback = FeedbackRecord(
            record_id=record_id,
            feedback_type=feedback_type,
            feedback_reason=feedback_reason,
            feedback_time=datetime.now()
        )
        db.add(feedback)
        db.commit()
        return True
    except Exception as e:
        db.rollback()
        print(f"保存反馈失败: {e}")
        return False
    finally:
        db.close()

# ====================== 8. 查询反馈记录 ======================
def query_feedback(record_id=None):
    """
    查询反馈记录
    :param record_id: 关联的违规记录ID（可选）
    :return: 反馈记录列表
    """
    db = SessionLocal()
    try:
        query = db.query(FeedbackRecord)
        
        if record_id:
            query = query.filter(FeedbackRecord.record_id == record_id)
        
        # 按时间倒序排列（最新的在前）
        return query.order_by(FeedbackRecord.feedback_time.desc()).all()
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
            "违规车辆数量": r.illegal_count,
            "合法停放车辆数量": r.legal_count,
            "是否违规": "是" if r.is_illegal == 1 else "否",
            "截图路径": r.screenshot_path
        })

    df = pd.DataFrame(data)
    df.to_excel(save_path, index=False)
    print(f"✅ 历史记录已导出: {save_path}")
    return save_path

# ====================== 🔥 4. 导出【当前检测结果】到 Excel (增强版) ======================
def export_current_detection_excel(detect_list, save_path=None):
    """
    导出当前画面检测到的所有自行车到 Excel（包含违规信息）
    :param detect_list: 格式 [{"x1":.., "y1":.., "x2":.., "y2":.., "score":.., "illegal_type":.., "overlap":..}, ...]
    :return: 文件名, 数量
    """
    if not detect_list:
        return None, 0

    if not save_path:
        save_path = f"当前检测结果_{datetime.now().strftime('%Y%m%d%H%M%S')}.xlsx"

    data = []
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for idx, obj in enumerate(detect_list, 1):
        # 判断是否为违规
        illegal_type = obj.get("illegal_type", "合法")
        is_illegal = "是" if illegal_type and illegal_type != "合法" else "否"
        overlap = obj.get("overlap", 0.0)
        
        data.append({
            "序号": idx,
            "检测时间": current_time,
            "类别": "bike",
            "置信度": round(obj["score"], 2),
            "是否违规": is_illegal,
            "违规类型": illegal_type if is_illegal == "是" else "无",
            "重叠比例": f"{overlap:.2%}" if overlap > 0 else "0%",
            "左上角X": int(obj["x1"]),
            "左上角Y": int(obj["y1"]),
            "右下角X": int(obj["x2"]),
            "右下角Y": int(obj["y2"]),
        })

    df = pd.DataFrame(data)
    df.to_excel(save_path, index=False)
    print(f"✅ 当前结果已导出: {save_path}")
    return save_path, len(data)


# ====================== 5. 导出【查询结果】到 Excel ======================
def export_query_results_to_excel(records, save_path=None):
    """
    将查询结果导出到Excel
    :param records: 违规记录列表（ViolationRecord对象列表）
    :param save_path: 保存路径
    :return: 保存的文件路径
    """
    if not records:
        print("⚠️ 没有记录可导出")
        return None

    if not save_path:
        save_path = f"查询结果_{datetime.now().strftime('%Y%m%d%H%M%S')}.xlsx"

    data = []
    for r in records:
        data.append({
            "ID": r.id,
            "检测时间": r.detect_time.strftime("%Y-%m-%d %H:%M:%S"),
            "违规区域": r.violation_area,
            "违规车辆数量": r.illegal_count,
            "合法停放车辆数量": r.legal_count,
            "是否违规": "是" if r.is_illegal == 1 else "否",
            "截图路径": r.screenshot_path,
            "创建时间": r.create_time.strftime("%Y-%m-%d %H:%M:%S") if r.create_time else "",
        })

    df = pd.DataFrame(data)
    df.to_excel(save_path, index=False)
    print(f"✅ 查询结果已导出: {save_path}")
    return save_path


# ====================== 6. 导出【统计报告】到 Excel ======================
def export_statistics_to_excel(statistics, save_path=None):
    """
    将统计信息导出到Excel（包含多个sheet）
    :param statistics: 统计信息字典
    :param save_path: 保存路径
    :return: 保存的文件路径
    """
    if not save_path:
        save_path = f"违规统计报告_{datetime.now().strftime('%Y%m%d%H%M%S')}.xlsx"

    with pd.ExcelWriter(save_path, engine='openpyxl') as writer:
        # Sheet 1: 总体统计
        summary_data = {
            "统计项": ["总检测次数"],
            "数值": [statistics.get("total_count", 0)]
        }
        pd.DataFrame(summary_data).to_excel(writer, sheet_name='总体统计', index=False)
        
        # Sheet 2: 按区域统计
        area_stats = statistics.get("area_statistics", {})
        if area_stats:
            area_data = {
                "违规区域": list(area_stats.keys()),
                "检测次数": list(area_stats.values())
            }
            pd.DataFrame(area_data).to_excel(writer, sheet_name='按区域统计', index=False)

    print(f"✅ 统计报告已导出: {save_path}")
    return save_path