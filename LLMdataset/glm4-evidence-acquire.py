from zhipuai import ZhipuAI

client = ZhipuAI(api_key="---.---")
response = client.chat.completions.create(
    model="glm-4",
    messages=[
        {
            "role": "user",
            "content": "寻找给你一篇新闻如下所示：Lindsey Vonn, the American Olympic gold medallist, is single and struggling to find a date. Despite her fame, beauty, and charm, Vonn has been unable to find a partner, and she recently appeared on a talk show to find a Valentine. The hosts of the show, Natalie Morales and Kit Hoover, made it their mission to find Vonn a partner and even tried to set her up with celebrities like Jon Hamm and Brad Pitt. However, Vonn is not interested in any of them, and she has rejected several potential partners, including Rob Gronkowski and Drake. She is currently focusing on her dogs, which she says she doesn't need a man for.请在网络上寻找支持或反对该新闻的文本证据。假如你认为这篇文章的内容是真的，那就只给出支持的文本证据。若为假新闻则给出反对的文本证据。要求使用英文给出结果，只返回证据文本结果即可，不要有其他说明"
        }
    ],
    top_p=0.7,
    temperature=0.9,
    stream=False,
    max_tokens=2000,
)
print(response.choices[0].message.content)

# from zhipuai import ZhipuAI
#
# client = ZhipuAI(api_key="----.---")  # 请填写您自己的APIKey
#
# tools = [
#     {
#         "type": "function",
#         "function": {
#             "name": "query_train_info",
#             "description": "根据用户提供的信息，查询对应的车次",
#             "parameters": {
#                 "type": "object",
#                 "properties": {
#                     "departure": {
#                         "type": "string",
#                         "description": "出发城市或车站",
#                     },
#                     "destination": {
#                         "type": "string",
#                         "description": "目的地城市或车站",
#                     },
#                     "date": {
#                         "type": "string",
#                         "description": "要查询的车次日期",
#                     },
#                 },
#                 "required": ["departure", "destination", "date"],
#             },
#         }
#     }
# ]
# messages = [
#     {
#         "role": "user",
#         "content": "你能帮我查询2024年1月1日从北京南站到上海的火车票吗？"
#     }
# ]
# response = client.chat.completions.create(
#     model="glm-4",  # 填写需要调用的模型名称
#     messages=messages,
#     tools=tools,
#     tool_choice="auto",
# )
# print(response.choices[0].message)
