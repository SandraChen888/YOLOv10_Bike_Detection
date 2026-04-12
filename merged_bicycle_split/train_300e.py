from ultralytics import YOLO

if __name__ == '__main__':
    # 全新从头开始训练
    model = YOLO('yolov10n.pt')

    model.train(
        data='merged_bicycle_split/bicycle_grid.yaml',
        epochs=300,
        batch=16,        # 安全不爆显存
        imgsz=640,
        device=0,
        amp=False,
        patience=0,
        save_period=50,  # 50轮存一次，省空间
        plots=False,     # 不生成图表，防止磁盘爆满
        exist_ok=True,   # 覆盖旧文件，不占额外空间
        name='bike_yellow_grid_300e'
    )