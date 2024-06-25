import os
import shutil
import time

# 设置要监控的目录
directory_to_monitor = '/2310274046/FakeNewDetection/MUSER/bioclassifier/roberta-large-finetuned-BIO'

# 设置定时任务间隔（10分钟）
interval = 16 * 60

while True:
    # 获取目录中的所有文件夹
    folders = [f for f in os.listdir(directory_to_monitor) if os.path.isdir(os.path.join(directory_to_monitor, f))]

    # 遍历文件夹
    for folder in folders:
        # 检查文件夹名是否包含'79'
        if  ('run' not in folder):
            # 获取文件夹的完整路径
            folder_path = os.path.join(directory_to_monitor, folder)
            # 删除文件夹中的所有数据
            shutil.rmtree(folder_path)
            print(f"Deleted all data in {folder_path}")

    # 等待指定的时间间隔
    time.sleep(interval)
