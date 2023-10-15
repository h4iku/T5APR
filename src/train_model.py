from math import ceil

import torch
from datasets import concatenate_datasets, load_dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)


print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)

### Load model and tokenizer
model_checkpoint = "Salesforce/codet5-small"

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

### Load data
dataset_names = {
    "Python": "h4iku/coconut_python2010_preprocessed",
    "Java": "h4iku/coconut_java2006_preprocessed",
    "JavaScript": "h4iku/coconut_javascript2010_preprocessed",
    "C": "h4iku/coconut_c2005_preprocessed",
}

raw_datasets = {
    prefix: load_dataset(data, split="train") for prefix, data in dataset_names.items()
}
print(raw_datasets)

### Tokenize data
max_input_length = 512
max_target_length = 256


def preprocess_function(prefix, examples):
    inputs = [
        f"{prefix} {ex.strip()} :".replace(tokenizer.eos_token, tokenizer.unk_token)
        for ex in examples["rem"]
    ]
    targets = [
        ex.strip().replace(tokenizer.eos_token, tokenizer.unk_token)
        for ex in examples["add"]
    ]
    contexts = [
        " ".join(ex.split()).replace(tokenizer.eos_token, tokenizer.unk_token)
        for ex in examples["context"]
    ]

    inputs_contexts = [f"{src} {ctx}" for src, ctx in zip(inputs, contexts)]

    model_inputs = tokenizer(
        inputs_contexts, max_length=max_input_length, truncation=True
    )
    model_inputs["inputs_only"] = tokenizer(inputs).input_ids

    labels = tokenizer(targets)

    model_inputs["labels"] = labels["input_ids"]

    return model_inputs


tokenized_datasets = [
    raw_dataset.map(
        lambda ex: preprocess_function(prefix, ex),
        batched=True,
        remove_columns=raw_dataset.column_names,
    )
    for prefix, raw_dataset in raw_datasets.items()
]


### Filter data
def allowed_length(example):
    """Filter out long samples and samples with empty target"""
    return (
        len(example["inputs_only"]) <= max_input_length
        and 2 < len(example["labels"]) <= max_target_length
    )


filtered_datasets = [
    td.filter(allowed_length).remove_columns(["inputs_only"])
    for td in tokenized_datasets
]


concatenated_dataset = concatenate_datasets(filtered_datasets)
print(concatenated_dataset)


### Training
train_batch_size = 8
eval_batch_size = 16
model_name = model_checkpoint.split("/")[-1]
epochs = 1
lr = 1e-4
output_dir = f"{model_name}-multi"

checkpoints_each_epoch = 5
epoch_steps = ceil(len(concatenated_dataset) / train_batch_size)
train_steps = epoch_steps * epochs
save_steps = epoch_steps // checkpoints_each_epoch

args = Seq2SeqTrainingArguments(
    output_dir,
    learning_rate=lr,
    per_device_train_batch_size=train_batch_size,
    per_device_eval_batch_size=eval_batch_size,
    save_total_limit=5,
    max_steps=train_steps,
    save_steps=save_steps,
    predict_with_generate=True,
    fp16=True,  # fp16 isn't fully supported on 10 series cards.
    lr_scheduler_type="constant",
)

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

### Setting generate hyperparameters
model.config.max_length = max_target_length
model.config.min_length = 0
model.config.early_stopping = True
model.config.num_beams = 5


trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=concatenated_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

trainer.train(resume_from_checkpoint=False)
