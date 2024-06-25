import random

# 读取train.jsonl文件的所有行
with open('/2310274046/FakeNewDetection/MUSER/bioclassifier/data/train.jsonl', 'r', encoding='utf-8') as file:
    lines = file.readlines()

# 随机抽出64000条数据
random_lines = random.sample(lines, 64000)

# 覆盖train.jsonl文件
with open('/2310274046/FakeNewDetection/MUSER/bioclassifier/train.jsonl', 'w', encoding='utf-8') as file:
    for line in random_lines:
        file.write(line)

print("操作完成。")
