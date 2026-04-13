# 查询数据库中的记录，查看当前的记录情况
from record_manage import query_record

# 查询所有记录
records = query_record()

# 打印记录信息
print("数据库中的记录：")
print("-" * 100)
for record in records:
    is_illegal_str = "是" if record.is_illegal == 1 else "否"
    print(f"ID: {record.id}")
    print(f"检测时间: {record.detect_time}")
    print(f"截图路径: {record.screenshot_path}")
    print(f"违规区域: {record.violation_area}")
    print(f"违规车辆数量: {record.illegal_count}")
    print(f"合法停放车辆数量: {record.legal_count}")
    print(f"是否违规: {is_illegal_str}")
    print(f"创建时间: {record.create_time}")
    print(f"更新时间: {record.update_time}")
    print("-" * 100)
