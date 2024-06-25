# Labels
label_list = ["B", "I", "O"]
lbl2idx = {"B": 0, "I": 1, "O": 2}
idx2label = {0: "B", 1: "I", 2: "O"}

task = "BIO"
model_checkpoint = "FacebookAI/roberta-large"
batch_size = 8

import datasets
from datasets import load_dataset,load_from_disk,Features,Value,ClassLabel,Sequence
# datasets_ = load_dataset(path="json",data_dir="/2310274046/FakeNewDetection/MUSER/bioclassifier/data")
context_feat = Features({'document': Sequence(feature=Value(dtype='string', id=None), length=-1, id=None),
                         'doc_bio_tags':Sequence(feature=ClassLabel(num_classes=3, names=[0,1,2], names_file=None, id=None), length=-1, id=None)})
# datasets_ = load_dataset('json',data_files={'train':['/2310274046/FakeNewDetection/MUSER/bioclassifier/data/train.jsonl'],
#                                              'test':['/2310274046/FakeNewDetection/MUSER/bioclassifier/data/test.jsonl'],
#                                              'validation':['/2310274046/FakeNewDetection/MUSER/bioclassifier/data/validation.jsonl']},features=context_feat)
datasets_ = load_dataset(path="json",data_dir="/2310274046/FakeNewDetection/MUSER/bioclassifier/data",features=context_feat)

print(datasets_)

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint,add_prefix_space=True)

import os
import torch,gc
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb=100"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["document"], truncation=True, is_split_into_words=True)

    labels = []
    for i, label in enumerate(examples["doc_bio_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                label_ids.append(-100)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            else:
                label_ids.append(label[word_idx] if True else -100)
            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

tokenized_datasets = datasets_.map(tokenize_and_align_labels, batched=True)

from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer

gc.collect()
torch.cuda.empty_cache()

model = AutoModelForTokenClassification.from_pretrained(model_checkpoint, num_labels=3).to(device)
# model = AutoModelForTokenClassification.from_pretrained('/2310274046/FakeNewDetection/MUSER/bioclassifier/model', num_labels=3,from_tf=True).to(device)
model_name = model_checkpoint.split("/")[-1]
args = TrainingArguments(
    f"/2310274046/FakeNewDetection/MUSER/bioclassifier/{model_name}-finetuned-BIO",
    evaluation_strategy = "epoch",
    learning_rate=2e-4,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=20,
    weight_decay=0.01,
    hub_model_id="leehongzhong/roberta-large-finetuned-bio"
)
from transformers import DataCollatorForTokenClassification

data_collator = DataCollatorForTokenClassification(tokenizer)

import numpy as np
# from datasets import load_metric
import evaluate
# metric = load_metric("seqeval")
metric = evaluate.load("seqeval")
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

trainer = Trainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)
trainer.train()

with torch.no_grad():
    trainer.evaluate()

predictions, labels, _ = trainer.predict(tokenized_datasets["test"])
predictions = np.argmax(predictions, axis=2)

# Remove ignored index (special tokens)
true_predictions = [
    [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
    for prediction, label in zip(predictions, labels)
]
true_labels = [
    [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
    for prediction, label in zip(predictions, labels)
]

results = metric.compute(predictions=true_predictions, references=true_labels)
print(results)