# MUSER

## 建立环境

* Python3.7 
* 使用pip下载requirement.txt中所有依赖.
```shell script
pip3 install -r requirements.txt
```

## 根据语料库数据建立索引:

可以参考`MUSER_Colab.ipynb`文件

```
python multi_step_retriever.py --index example_docs.jsonl
```

运行结果示例

![image-20240625132918341](D:\homework\machineLearning\code\MUSER\multi_retriever\image-20240625132918341.png)

## 训练自然语言推理模型

* 运行*train.py*，并给定对于的基础模型结构以及是否使用checkpoint技术:
```shell script
python train.py --bert_type bert-large --check_point 1
```
训练后的模型（具有最佳验证性能）将保存在 *output/*文件夹下，取最佳准确率模型作为测试模型

![image-20240625132448968](D:\homework\machineLearning\code\MUSER\output\image-20240625132448968.png)

## 验证模型性能
* 使用如下命令进行测试，测试模型与训练模型架构需要保持一致:
```shell script
python test_trained_model.py --bert_type bert-large
```

需要下载模型并将其放在 `output/nli_model` 文件夹:

 - <a href="https://drive.google.com/drive/folders/1-aPX4HBxe8U3ErzOyoYfs-V5lpmGsVWw?usp=share_link">NLI model</a>

运行结果示例：

![image-20240625132634569](D:\homework\machineLearning\code\MUSER\output\image-20240625132634569.png)


## 参数设置

| |PolitiFact| Gossipcop|
|-|-|-|
| Sequence_length | 512|512 |
| Max_encoder_length | 512|512 |
| Min_decoder_length | 64|64 |
| Max_decoder_length | 128|128 |
| Embedding_dimension | 200| 200|
| k(number of paragraphs retrieved) |30 |30 |
| MSR| 0.3| 0.3|
|$lambda$ |0.9 |0.9 |
| Retrieve_steps | 2| 3|
| Batch_size |64 |64 |
| Maximum_epochs |10 |10 |
| Vocabulary_size | 30522|30522 |
| Learning_rate | 1e-5| 1e-5|





