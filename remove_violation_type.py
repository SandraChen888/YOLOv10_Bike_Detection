# 从数据库表中删除违规类型字段
import pymysql

# 数据库连接信息
db_config = {
    "host": "localhost",
    "user": "root",
    "password": "123456",
    "database": "bike_violation_db"
}

# 连接数据库
conn = pymysql.connect(**db_config)
cursor = conn.cursor()

try:
    # 删除 violation_type 字段
    cursor.execute("ALTER TABLE violation_records DROP COLUMN violation_type")
    
    # 提交更改
    conn.commit()
    print("✅ 成功删除违规类型字段！")
    
    # 查看更新后的表结构
    cursor.execute("DESCRIBE violation_records")
    print("\n更新后的表结构：")
    for row in cursor.fetchall():
        print(row)
        
except Exception as e:
    print(f"❌ 删除违规类型字段失败：{str(e)}")
    conn.rollback()
finally:
    # 关闭连接
    cursor.close()
    conn.close()
