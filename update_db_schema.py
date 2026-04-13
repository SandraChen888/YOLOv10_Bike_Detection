# 更新数据库表结构
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
    # 添加 scene 字段
    cursor.execute("ALTER TABLE violation_records ADD COLUMN scene VARCHAR(100) NOT NULL DEFAULT '默认场景' AFTER violation_area")
    
    # 添加 illegal_count 字段
    cursor.execute("ALTER TABLE violation_records ADD COLUMN illegal_count INT NOT NULL DEFAULT 0 AFTER violation_type")
    
    # 添加 legal_count 字段
    cursor.execute("ALTER TABLE violation_records ADD COLUMN legal_count INT NOT NULL DEFAULT 0 AFTER illegal_count")
    
    # 提交更改
    conn.commit()
    print("✅ 数据库表结构更新成功！")
    
    # 查看更新后的表结构
    cursor.execute("DESCRIBE violation_records")
    print("\n更新后的表结构：")
    for row in cursor.fetchall():
        print(row)
        
except Exception as e:
    print(f"❌ 数据库表结构更新失败：{str(e)}")
    conn.rollback()
finally:
    # 关闭连接
    cursor.close()
    conn.close()
