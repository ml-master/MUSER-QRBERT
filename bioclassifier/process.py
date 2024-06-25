import json
# # 定义映射关系
# tag_to_id = {"B": 0, "I": 1, "O": 2}

# # 打开JSONL文件
# with open('/content/drive/MyDrive/MUSER/TokenClassification/train_bio.jsonl', 'r', encoding='utf-8') as file:
#     # 打开输出文件，用于写入转换后的数据
#     with open('/content/drive/MyDrive/MUSER/TokenClassification/data_digitlabel/train.jsonl', 'w', encoding='utf-8') as output_file:
#         # 逐行读取
#         for line in file:
#             # 解析每一行的JSON数据
#             data = json.loads(line)
#             # 获取doc_bio_tags字段
#             doc_bio_tags = data.get('doc_bio_tags')
#             if doc_bio_tags:
#                 # 使用列表推导式和映射关系进行转换
#                 converted_tags = [tag_to_id[tag] for tag in doc_bio_tags]
#                 # 将转换后的数字数组更新到数据中
#                 data['doc_bio_tags'] = converted_tags
#                 # 将转换后的数据写入输出文件
#                 output_file.write(json.dumps(data) + '\n')


import json

# 读取train.jsonl文件的所有行
with open('/content/drive/MyDrive/MUSER/TokenClassification/data_digitlabel/train.jsonl', 'r', encoding='utf-8') as file:
    lines = file.readlines()

# 确定验证集的起始和结束索引
start_index = len(lines) - 8000
end_index = len(lines)

# 提取验证集数据
validation_data = lines[start_index:end_index]

# 写入验证集到validation.jsonl文件
with open('/content/drive/MyDrive/MUSER/TokenClassification/data_digitlabel/validation.jsonl', 'w', encoding='utf-8') as file:
    for line in validation_data:
        file.write(line)

# 删除train.jsonl中的验证集数据
train_data = lines[:start_index]

# 将剩余的训练数据写回到train.jsonl文件
with open('/content/drive/MyDrive/MUSER/TokenClassification/data_digitlabel/train.jsonl', 'w', encoding='utf-8') as file:
    for line in train_data:
        file.write(line)

print("操作完成。")

