import json

document = []
doc_bio_tags = []
# 打开文件
with open('/2310274046/FakeNewDetection/MUSER/multi_retriever/data/train.txt', 'r', encoding='utf-8') as file, open('train_bio.jsonl', 'w', encoding='utf-8') as output_file:
    for line in file:
        line = line.strip()
        if not line:
            assert len(document) == len(doc_bio_tags), 'length error!'
            output_file.write(json.dumps({"document": document, "doc_bio_tags": doc_bio_tags}) + '\n')
            document = []
            doc_bio_tags = []
            continue

        parts = line.split('\t')
        # 如果分隔后长度不是2，可以根据实际情况处理，这里仅打印
        if len(parts) != 2:
            print("这行数据的格式不正确:", line)
        else:
            string1, string2 = parts
            document.append(string1)
            if 'O' in string2:
                doc_bio_tags.append('O')
            elif 'B' in string2:
                doc_bio_tags.append('B')
            else:
                doc_bio_tags.append('I')