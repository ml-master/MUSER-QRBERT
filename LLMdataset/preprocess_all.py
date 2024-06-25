# 把所有样本都结合起来起来

import json
import re
import glob

# path = "/2310274046/FakeNewDetection/MUSER/CJYdataset/gossipcop_v3-1_style_based_fake.json"
# test_json = "/2310274046/FakeNewDetection/MUSER/CJYdataset/cjy_test_datasets_v31.json"
# with open(path, 'r', encoding='utf-8') as infile, open(test_json, 'w', encoding='utf-8') as outfile:
#     infile_json = json.load(infile)
#     keys=infile_json.keys()
#     for key in keys:
#         line = infile_json[key]
#         origin_text = line['origin_text'].replace('\n', '')
#         generated_text = line['generated_text'].replace('\n', '')
#         for i in (0, 1):
#             if i:
#                 if ("fake" in line['origin_label'] or "illegitimate" in line['origin_label']):
#                     json_data = {'text': origin_text, 'label': 0}
#                     outfile.write(json.dumps(json_data) + '\n')
#                 else:
#                     json_data = {'text': origin_text, 'label': 1}
#                     outfile.write(json.dumps(json_data) + '\n')
#             else:
#                 if ("fake" in line['generated_label'] or "illegitimate" in line['generated_label']):
#                     json_data = {'text': generated_text, 'label': 0}
#                     outfile.write(json.dumps(json_data) + '\n')
#                 else:
#                     json_data = {'text': generated_text, 'label': 1}
#                     outfile.write(json.dumps(json_data) + '\n')

# path = "/2310274046/FakeNewDetection/MUSER/CJYdataset/gossipcop_v3-2_content_based_fake.json"
# test_json = "/2310274046/FakeNewDetection/MUSER/CJYdataset/cjy_test_datasets_v32.json"
# with open(path, 'r', encoding='utf-8') as infile, open(test_json, 'w', encoding='utf-8') as outfile:
#     infile_json = json.load(infile)
#     keys=infile_json.keys()
#     for key in keys:
#         line = infile_json[key]
#         origin_text = line['origin_text'].replace('\n', '')
#         generated_text = re.sub(r'[\n\[\]]', '', line['generated_text_glm4'])
#         for i in (0, 1):
#             if i:
#                 if ("fake" in line['origin_label'] or "illegitimate" in line['origin_label']):
#                     json_data = {'text': origin_text, 'label': 0}
#                     outfile.write(json.dumps(json_data) + '\n')
#                 else:
#                     json_data = {'text': origin_text, 'label': 1}
#                     outfile.write(json.dumps(json_data) + '\n')
#             else:
#                 json_data = {'text': generated_text, 'label': 0}
#                 outfile.write(json.dumps(json_data) + '\n')

# path = "/2310274046/FakeNewDetection/MUSER/CJYdataset/gossipcop_v3-3_integration_based_fake_tn200.json"
# test_json = "/2310274046/FakeNewDetection/MUSER/CJYdataset/cjy_test_datasets_v33.json"
# with open(path, 'r', encoding='utf-8') as infile, open(test_json, 'w', encoding='utf-8') as outfile:
#     infile_json = json.load(infile)
#     keys = infile_json.keys()
#     for key in keys:
#         line = infile_json[key]
#         origin_text1 = line['doc_1_text'].replace('\n', '')
#         origin_text2 = line['doc_2_text'].replace('\n', '')
#         generated_text = line['generated_text'].replace('\n', '')
#         for i in (0, 1, 2):
#             if i == 1:
#                 json_data = {'text': origin_text1, 'label': 1}
#                 outfile.write(json.dumps(json_data) + '\n')
#             elif i == 2:
#                 json_data = {'text': origin_text2, 'label': 0}
#                 outfile.write(json.dumps(json_data) + '\n')
#             else:
#                 json_data = {'text': generated_text, 'label': 0}
#                 outfile.write(json.dumps(json_data) + '\n')


# path = "/2310274046/FakeNewDetection/MUSER/CJYdataset/gossipcop_v3-4_story_based_fake.json"
# test_json = "/2310274046/FakeNewDetection/MUSER/CJYdataset/cjy_test_datasets_v34.json"
# with open(path, 'r', encoding='utf-8') as infile, open(test_json, 'w', encoding='utf-8') as outfile:
#     infile_json = json.load(infile)
#     keys = infile_json.keys()
#     for key in keys:
#         line = infile_json[key]
#         origin_text = line['origin_text'].replace('\n', '')
#         generated_text = line['generated_text'].replace('\n', '')
#         for i in (0, 1):
#             if i == 1:
#                 if ("fake" in line['origin_label'] or "illegitimate" in line['origin_label']):
#                     json_data = {'text': origin_text, 'label': 0}
#                     outfile.write(json.dumps(json_data) + '\n')
#                 else:
#                     json_data = {'text': origin_text, 'label': 1}
#                     outfile.write(json.dumps(json_data) + '\n')
#             else:
#                 json_data = {'text': generated_text, 'label': 0}
#                 outfile.write(json.dumps(json_data) + '\n')


# path = "/2310274046/FakeNewDetection/MUSER/CJYdataset/gossipcop_v3-5_style_based_legitimate.json"
# test_json = "/2310274046/FakeNewDetection/MUSER/CJYdataset/cjy_test_datasets_v35.json"
# with open(path, 'r', encoding='utf-8') as infile, open(test_json, 'w', encoding='utf-8') as outfile:
#     infile_json = json.load(infile)
#     keys = infile_json.keys()
#     for key in keys:
#         line = infile_json[key]
#         origin_text = line['origin_text'].replace('\n', '')
#         generated_text = line['generated_text_t015'].replace('\n', '')
#         for i in (0, 1):
#             if i == 1:
#                 json_data = {'text': origin_text, 'label': 1}
#                 outfile.write(json.dumps(json_data) + '\n')
#             else:
#                 json_data = {'text': generated_text, 'label': 1}
#                 outfile.write(json.dumps(json_data) + '\n')


# path = "/2310274046/FakeNewDetection/MUSER/CJYdataset/gossipcop_v3-7_integration_based_legitimate_tn300.json"
# test_json = "/2310274046/FakeNewDetection/MUSER/CJYdataset/cjy_test_datasets_v37.json"
# with open(path, 'r', encoding='utf-8') as infile, open(test_json, 'w', encoding='utf-8') as outfile:
#     infile_json = json.load(infile)
#     keys = infile_json.keys()
#     for key in keys:
#         line = infile_json[key]
#         origin_text1 = line['doc_1_text'].replace('\n', '')
#         origin_text2 = line['doc_2_text'].replace('\n', '')
#         generated_text = line['generated_text_t01'].replace('\n', '')
#         for i in (0, 1, 2):
#             if i == 1:
#                 json_data = {'text': origin_text1, 'label': 1}
#                 outfile.write(json.dumps(json_data) + '\n')
#             elif i == 2:
#                 json_data = {'text': origin_text2, 'label': 1}
#                 outfile.write(json.dumps(json_data) + '\n')
#             else:
#                 json_data = {'text': generated_text, 'label': 1}
#                 outfile.write(json.dumps(json_data) + '\n')


# files = glob.glob('cjy_test_datasets_v*.json')
# with open('cjy_test_datasets_all.json', 'w') as outfile:
#     for filename in files:
#         with open("/2310274046/FakeNewDetection/MUSER/CJYdataset/"+filename, 'r', encoding='utf-8') as infile:
#             for line in infile.readlines():
#                 dic=json.loads(line)
#                 outfile.write(json.dumps(dic) + '\n')

# 只收集机器生成的新闻样本

# path = "/2310274046/FakeNewDetection/MUSER/CJYdataset/gossipcop_v3-1_style_based_fake.json"
# test_json = "/2310274046/FakeNewDetection/MUSER/CJYdataset/cjy_test_datasets_onlygen_v31.json"
# with open(path, 'r', encoding='utf-8') as infile, open(test_json, 'w', encoding='utf-8') as outfile:
#     infile_json = json.load(infile)
#     keys = infile_json.keys()
#     for key in keys:
#         line = infile_json[key]
#         generated_text = line['generated_text'].replace('\n', '')
#         if ("fake" in line['generated_label'] or "illegitimate" in line['generated_label']):
#             json_data = {'text': generated_text, 'label': 0}
#             outfile.write(json.dumps(json_data) + '\n')
#         else:
#             json_data = {'text': generated_text, 'label': 1}
#             outfile.write(json.dumps(json_data) + '\n')
#
# path = "/2310274046/FakeNewDetection/MUSER/CJYdataset/gossipcop_v3-2_content_based_fake.json"
# test_json = "/2310274046/FakeNewDetection/MUSER/CJYdataset/cjy_test_datasets_onlygen_v32.json"
# with open(path, 'r', encoding='utf-8') as infile, open(test_json, 'w', encoding='utf-8') as outfile:
#     infile_json = json.load(infile)
#     keys = infile_json.keys()
#     for key in keys:
#         line = infile_json[key]
#         generated_text = re.sub(r'[\n\[\]]', '', line['generated_text_glm4'])
#         json_data = {'text': generated_text, 'label': 0}
#         outfile.write(json.dumps(json_data) + '\n')
#
# path = "/2310274046/FakeNewDetection/MUSER/CJYdataset/gossipcop_v3-3_integration_based_fake_tn200.json"
# test_json = "/2310274046/FakeNewDetection/MUSER/CJYdataset/cjy_test_datasets_onlygen_v33.json"
# with open(path, 'r', encoding='utf-8') as infile, open(test_json, 'w', encoding='utf-8') as outfile:
#     infile_json = json.load(infile)
#     keys = infile_json.keys()
#     for key in keys:
#         line = infile_json[key]
#         generated_text = line['generated_text'].replace('\n', '')
#         json_data = {'text': generated_text, 'label': 0}
#         outfile.write(json.dumps(json_data) + '\n')
#
# path = "/2310274046/FakeNewDetection/MUSER/CJYdataset/gossipcop_v3-4_story_based_fake.json"
# test_json = "/2310274046/FakeNewDetection/MUSER/CJYdataset/cjy_test_datasets_onlygen_v34.json"
# with open(path, 'r', encoding='utf-8') as infile, open(test_json, 'w', encoding='utf-8') as outfile:
#     infile_json = json.load(infile)
#     keys = infile_json.keys()
#     for key in keys:
#         line = infile_json[key]
#         generated_text = line['generated_text'].replace('\n', '')
#         json_data = {'text': generated_text, 'label': 0}
#         outfile.write(json.dumps(json_data) + '\n')
#
# path = "/2310274046/FakeNewDetection/MUSER/CJYdataset/gossipcop_v3-5_style_based_legitimate.json"
# test_json = "/2310274046/FakeNewDetection/MUSER/CJYdataset/cjy_test_datasets_onlygen_v35.json"
# with open(path, 'r', encoding='utf-8') as infile, open(test_json, 'w', encoding='utf-8') as outfile:
#     infile_json = json.load(infile)
#     keys = infile_json.keys()
#     for key in keys:
#         line = infile_json[key]
#         generated_text = line['generated_text_t015'].replace('\n', '')
#         json_data = {'text': generated_text, 'label': 1}
#         outfile.write(json.dumps(json_data) + '\n')
#
# path = "/2310274046/FakeNewDetection/MUSER/CJYdataset/gossipcop_v3-7_integration_based_legitimate_tn300.json"
# test_json = "/2310274046/FakeNewDetection/MUSER/CJYdataset/cjy_test_datasets_onlygen_v37.json"
# with open(path, 'r', encoding='utf-8') as infile, open(test_json, 'w', encoding='utf-8') as outfile:
#     infile_json = json.load(infile)
#     keys = infile_json.keys()
#     for key in keys:
#         line = infile_json[key]
#         origin_text1 = line['doc_1_text'].replace('\n', '')
#         origin_text2 = line['doc_2_text'].replace('\n', '')
#         generated_text = line['generated_text_t01'].replace('\n', '')
#         json_data = {'text': generated_text, 'label': 1}
#         outfile.write(json.dumps(json_data) + '\n')
#
# files = glob.glob('cjy_test_datasets_onlygen_v*.json')
# with open('cjy_test_datasets_onlygen.json', 'w') as outfile:
#     for filename in files:
#         with open("/2310274046/FakeNewDetection/MUSER/CJYdataset/" + filename, 'r', encoding='utf-8') as infile:
#             for line in infile.readlines():
#                 dic = json.loads(line)
#                 outfile.write(json.dumps(dic) + '\n')


# 使用经过新闻摘要预训练的模型生成claim

# import requests
# API_URL = "https://api-inference.huggingface.co/models/chinhon/pegasus-multi_news-summarizer_01"
# headers = {"Authorization": "Bearer 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"}
# def query(payload):
#     response = requests.post(API_URL, headers=headers, json=payload)
#     return response.json()
# output = query({
#     "inputs": data['text'],
# })
# print(output)

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import csv
import json

device = "cuda" if torch.cuda.is_available() else "cpu"
# tokenizer = AutoTokenizer.from_pretrained("chinhon/pegasus-multi_news-summarizer_01")
# model = AutoModelForSeq2SeqLM.from_pretrained("chinhon/pegasus-multi_news-summarizer_01").to(device)
tokenizer = AutoTokenizer.from_pretrained("/2310274046/FakeNewDetection/MUSER/CJYdataset")
model = AutoModelForSeq2SeqLM.from_pretrained("/2310274046/FakeNewDetection/MUSER/CJYdataset").to(device)

input_json = '/2310274046/FakeNewDetection/MUSER/CJYdataset/cjy_test_datasets_all.json'
output_json = '/2310274046/FakeNewDetection/MUSER/CJYdataset/cjy_all_claim_label.jsonl'
count=0
with open(input_json, 'r', encoding='utf-8') as inputjson, open(output_json, 'w', encoding='utf-8') as output_json:
    for line in inputjson.readlines():
        data = json.loads(line)
        batch = tokenizer(data['text'], truncation=True, padding="longest", return_tensors="pt").to(device)
        translated = model.generate(**batch)
        tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
        output_json.write(json.dumps({'claim': tgt_text[0], 'label': data['label']}) + '\n')
        count+=1
        if count%100==0:
            print(f'{count} claims has generated')

print(f'cjy\'s test gossipcop_claims data saved to {output_txt}')
