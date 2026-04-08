#!/usr/bin/env python3
# 测试数据库保存功能

from record_manage import save_detection_record, query_record, get_detection_statistics
from datetime import datetime
import os

print("=== 测试数据库保存功能 ===")

# 测试保存合法记录
print("\n1. 测试保存合法记录...")
screenshot_path = "test_screenshots/test_legal.jpg"
os.makedirs(os.path.dirname(screenshot_path), exist_ok=True)
with open(screenshot_path, 'w') as f:
    f.write("test")

result1 = save_detection_record(
    screenshot_path=screenshot_path,
    violation_area="教学楼A栋",
    violation_type="合法",
    is_illegal=0
)
print(f"保存合法记录: {'成功' if result1 else '失败'}")

# 测试保存违规记录
print("\n2. 测试保存违规记录...")
screenshot_path2 = "test_screenshots/test_illegal.jpg"
with open(screenshot_path2, 'w') as f:
    f.write("test")

result2 = save_detection_record(
    screenshot_path=screenshot_path2,
    violation_area="教学楼B栋",
    violation_type="人行道",
    is_illegal=1
)
print(f"保存违规记录: {'成功' if result2 else '失败'}")

# 测试查询记录
print("\n3. 测试查询记录...")
records = query_record()
print(f"查询到 {len(records)} 条记录")
for i, record in enumerate(records):
    print(f"  记录{i+1}: ID={record.id}, 时间={record.detect_time}, 区域={record.violation_area}, 类型={record.violation_type}, 是否违规={record.is_illegal}")

# 测试统计功能
print("\n4. 测试统计功能...")
statistics = get_detection_statistics()
print(f"总记录数: {statistics['total_count']}")
print(f"类型统计: {statistics['type_statistics']}")
print(f"区域统计: {statistics['area_statistics']}")
print(f"合法性统计: {statistics['legality_statistics']}")

print("\n=== 测试完成 ===")
