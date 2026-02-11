# record_manage.py
import pandas as pd
import cv2
import os
from datetime import datetime

# 初始化记录保存路径
record_dir = "save_record"
img_save_dir = os.path.join(record_dir, "illegal_screenshots")
excel_path = os.path.join(record_dir, "bike_illegal_record.xlsx")
os.makedirs(img_save_dir, exist_ok=True)


def save_illegal_record(illegal_res, mark_img, image_name):
    """
    保存违规记录（Excel+截图）
    :param illegal_res: 违规结果（来自judge_illegal）
    :param mark_img: 标记违规后的图片
    :param image_name: 原始图片名
    :return: 保存状态、记录ID
    """
    try:
        # 生成唯一记录ID（时间戳）
        record_id = datetime.now().strftime("%Y%m%d%H%M%S")
        # 保存违规截图
        screenshot_name = f"{record_id}_{image_name}"
        cv2.imwrite(os.path.join(img_save_dir, screenshot_name), mark_img)

        # 整理Excel记录数据
        excel_data = []
        for res in illegal_res:
            excel_data.append({
                "记录ID": record_id,
                "检测时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "单车编号": res["单车编号"],
                "置信度": round(res["置信度"], 2),
                "检测框(左上x)": res["检测框"][0],
                "检测框(左上y)": res["检测框"][1],
                "检测框(右下x)": res["检测框"][2],
                "检测框(右下y)": res["检测框"][3],
                "违规类型": res["违规类型"],
                "重叠占比": round(res["重叠占比"], 2),
                "违规截图": screenshot_name
            })

        # 写入Excel（追加模式，无文件则新建）
        df = pd.DataFrame(excel_data)
        if os.path.exists(excel_path):
            df.to_excel(excel_path, mode="a", header=False, index=False)
        else:
            df.to_excel(excel_path, index=False)

        return True, record_id
    except Exception as e:
        return False, str(e)


def query_record(query_cond=None):
    """
    查询违规记录（按条件，如违规类型、时间）
    :param query_cond: 查询条件，如{"违规类型": "人行道"}
    :return: 查询结果（DataFrame）
    """
    if not os.path.exists(excel_path):
        return pd.DataFrame()
    df = pd.read_excel(excel_path)
    # 按条件筛选
    if query_cond and isinstance(query_cond, dict):
        for key, val in query_cond.items():
            if key in df.columns:
                df = df[df[key] == val]
    return df


# 测试代码
if __name__ == "__main__":
    # 模拟违规结果
    test_illegal = [
        {"单车编号": 1, "置信度": 0.85, "检测框": [200, 300, 300, 450], "违规类型": "人行道", "重叠占比": 0.45}]
    test_img = cv2.imread("illegal_mark.jpg")
    # 保存记录
    status, rid = save_illegal_record(test_illegal, test_img, "test_bike.jpg")
    if status:
        print(f"记录保存成功，记录ID：{rid}")
        # 查询记录
        res = query_record({"违规类型": "人行道"})
        print(f"查询结果：\n{res}")
    else:
        print(f"保存失败：{rid}")